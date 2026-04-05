from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any

import orjson


class TraceStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = self._load_records()

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def top_warm_starts(
        self,
        *,
        task_family: str | None,
        metric_name: str | None,
        family_signature: str | None = None,
        task_embedding: list[float] | None = None,
        task_embedding_is_real: bool = False,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        best_by_fingerprint: dict[str, dict[str, Any]] = {}
        for record in self.records:
            if record.get("record_type") != "compile_winner":
                continue
            fingerprint = str(record.get("winning_topology_fingerprint") or "")
            if not fingerprint:
                continue
            similarity_score = _warm_start_similarity_score(
                query_task_family=task_family,
                query_metric_name=metric_name,
                query_family_signature=family_signature,
                query_embedding=task_embedding,
                query_embedding_is_real=task_embedding_is_real,
                record=record,
            )
            if similarity_score <= 0:
                continue
            rank_score = similarity_score * float(record.get("winning_score", 0.0))
            ranked_record = dict(record)
            ranked_record["warm_start_similarity_score"] = similarity_score
            ranked_record["warm_start_rank_score"] = rank_score
            previous = best_by_fingerprint.get(fingerprint)
            if previous is None or (
                float(ranked_record.get("warm_start_rank_score", 0.0)),
                float(ranked_record.get("winning_score", 0.0)),
            ) > (
                float(previous.get("warm_start_rank_score", 0.0)),
                float(previous.get("winning_score", 0.0)),
            ):
                best_by_fingerprint[fingerprint] = ranked_record
        ranked = sorted(
            best_by_fingerprint.values(),
            key=lambda record: (
                float(record.get("warm_start_rank_score", 0.0)),
                float(record.get("warm_start_similarity_score", 0.0)),
                float(record.get("winning_score", 0.0)),
            ),
            reverse=True,
        )
        return ranked[:limit]

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = b"\n".join(orjson.dumps(record) for record in self.records)
        self.path.write_bytes(content + (b"\n" if self.records else b""))

    def _load_records(self) -> list[dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self.path.read_bytes().splitlines():
            if not line.strip():
                continue
            try:
                records.append(orjson.loads(line))
            except orjson.JSONDecodeError:
                continue
        return records


def _warm_start_similarity_score(
    *,
    query_task_family: str | None,
    query_metric_name: str | None,
    query_family_signature: str | None,
    query_embedding: list[float] | None,
    query_embedding_is_real: bool,
    record: dict[str, Any],
) -> float:
    score = 0.0
    if query_task_family and record.get("task_family") == query_task_family:
        score += 3
    if query_metric_name and record.get("metric_name") == query_metric_name:
        score += 2
    if _has_family_signature_prefix_overlap(query_family_signature, record.get("family_signature")):
        score += 1
    record_embedding = _record_task_embedding(record)
    if (
        query_embedding_is_real
        and _record_task_embedding_is_real(record)
        and query_embedding
        and record_embedding
        and len(query_embedding) == len(record_embedding)
    ):
        score += 2.0 * _cosine_similarity(query_embedding, record_embedding)
    return score


def _has_family_signature_prefix_overlap(query_signature: Any, candidate_signature: Any) -> bool:
    query_prefix = _signature_prefix_tokens(query_signature)
    candidate_prefix = _signature_prefix_tokens(candidate_signature)
    if not query_prefix or not candidate_prefix:
        return False
    overlap = 0
    for query_token, candidate_token in zip(query_prefix, candidate_prefix, strict=False):
        if query_token != candidate_token:
            break
        overlap += 1
    return overlap >= min(2, len(query_prefix), len(candidate_prefix))


def _signature_prefix_tokens(signature: Any) -> list[str]:
    prefix = str(signature or "").split("|", maxsplit=1)[0].strip()
    if not prefix:
        return []
    return [token for token in prefix.split("-") if token]


def _record_task_embedding(record: dict[str, Any]) -> list[float]:
    task_features = record.get("task_features") or {}
    embedding = task_features.get("task_embedding") or []
    if not isinstance(embedding, list):
        return []
    return [float(value) for value in embedding]


def _record_task_embedding_is_real(record: dict[str, Any]) -> bool:
    task_features = record.get("task_features") or {}
    return bool(task_features.get("task_embedding_is_real", False))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    cosine = sum(lhs * rhs for lhs, rhs in zip(left, right, strict=False)) / (left_norm * right_norm)
    return max(0.0, min(1.0, cosine))
