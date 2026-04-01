from __future__ import annotations

import re
from typing import Any, Callable

from topoprompt.schemas import Example


MetricFn = Callable[[Any, Example], float]


def metric_for_name(name: str | None) -> MetricFn:
    normalized = (name or "exact_match").lower()
    if normalized in {"exact_match", "accuracy"}:
        return exact_match_metric
    if normalized in {"numeric", "gsm8k"}:
        return numeric_metric
    if normalized in {"multiple_choice", "mmlu", "bbh"}:
        return multiple_choice_metric
    if normalized in {"ifeval", "instruction_following"}:
        return ifeval_metric
    return exact_match_metric


def exact_match_metric(prediction: Any, example: Example) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(example.target) else 0.0


def numeric_metric(prediction: Any, example: Example) -> float:
    pred_value = _extract_number(prediction)
    target_value = _extract_number(example.target)
    if pred_value is None or target_value is None:
        return exact_match_metric(prediction, example)
    return 1.0 if abs(pred_value - target_value) < 1e-9 else 0.0


def multiple_choice_metric(prediction: Any, example: Example) -> float:
    pred = _normalize_text(prediction)
    target = _normalize_text(example.target)
    if pred == target:
        return 1.0
    if pred[:1] == target[:1]:
        return 1.0
    return 0.0


def ifeval_metric(prediction: Any, example: Example) -> float:
    text = str(prediction or "")
    checks = 0
    passed = 0
    required_phrase = example.input.get("required_phrase") or example.metadata.get("required_phrase")
    if required_phrase:
        checks += 1
        passed += 1 if required_phrase.lower() in text.lower() else 0
    forbidden_phrase = example.metadata.get("forbidden_phrase")
    if forbidden_phrase:
        checks += 1
        passed += 1 if forbidden_phrase.lower() not in text.lower() else 0
    if example.target is not None:
        checks += 1
        passed += 1 if _normalize_text(text) == _normalize_text(example.target) else 0
    return passed / checks if checks else exact_match_metric(prediction, example)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _extract_number(value: Any) -> float | None:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))

