from __future__ import annotations

from statistics import mean
from pathlib import Path

from topoprompt.ir import prompt_token_count
from topoprompt.schemas import CandidateEvaluation, PromptProgram, TaskSpec


def extract_transfer_features(
    *,
    task_spec: TaskSpec,
    program: PromptProgram,
    candidate: CandidateEvaluation | None = None,
    metric_name: str | None = None,
    task_embedding: list[float] | None = None,
    task_embedding_is_real: bool = False,
) -> dict:
    task_features = _task_features_payload(
        task_spec=task_spec,
        metric_name=metric_name,
        task_embedding=task_embedding,
        task_embedding_is_real=task_embedding_is_real,
    )
    features = {
        "record_type": "candidate_feature",
        "task_features": task_features,
        "topology_features": {
            "num_nodes": len(program.nodes),
            "num_edges": len(program.edges),
            "route_nodes": sum(1 for node in program.nodes if node.node_type.value == "route"),
            "has_verify": any(node.node_type.value == "verify" for node in program.nodes),
            "has_plan": any(node.node_type.value == "plan" for node in program.nodes),
            "has_decompose": any(node.node_type.value == "decompose" for node in program.nodes),
            "prompt_token_count": prompt_token_count(program),
            "fewshot_count": sum(
                1 for node in program.nodes for module in node.prompt_modules if module.role == "fewshot"
            ),
        },
    }
    if candidate is not None:
        features["candidate_metrics"] = {
            "score": candidate.score,
            "mean_invocations": candidate.mean_invocations,
            "mean_tokens": candidate.mean_tokens,
            "parse_failure_rate": candidate.parse_failure_rate,
            "mean_example_score": mean(candidate.example_scores) if candidate.example_scores else 0.0,
        }
    return features


def extract_compile_winner_record(
    *,
    task_spec: TaskSpec,
    candidate: CandidateEvaluation,
    metric_name: str,
    output_dir: str | Path | None = None,
    task_embedding: list[float] | None = None,
    task_embedding_is_real: bool = False,
) -> dict:
    return {
        "record_type": "compile_winner",
        "task_family": task_spec.task_family,
        "metric_name": metric_name,
        "task_id": task_spec.task_id,
        "task_description": task_spec.description,
        "winning_topology_fingerprint": candidate.topology_fingerprint,
        "winning_score": candidate.score,
        "winning_search_score": candidate.search_score,
        "family_signature": candidate.family_signature,
        "task_features": _task_features_payload(
            task_spec=task_spec,
            metric_name=metric_name,
            task_embedding=task_embedding,
            task_embedding_is_real=task_embedding_is_real,
        ),
        "program": candidate.program.model_dump(mode="json"),
        "output_dir": str(Path(output_dir)) if output_dir is not None else None,
    }


def _task_features_payload(
    *,
    task_spec: TaskSpec,
    metric_name: str | None,
    task_embedding: list[float] | None,
    task_embedding_is_real: bool,
) -> dict:
    return {
        "task_family": task_spec.task_family,
        "metric_name": metric_name,
        "description_length": len(task_spec.description.split()),
        "input_schema_keys": sorted(task_spec.input_schema),
        "output_schema_keys": sorted(task_spec.output_schema),
        "task_embedding": list(task_embedding or []),
        "task_embedding_is_real": task_embedding_is_real and bool(task_embedding),
    }
