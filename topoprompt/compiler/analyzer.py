from __future__ import annotations

import json
import re

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.config import TopoPromptConfig
from topoprompt.compiler.seeds import SEED_LIBRARY
from topoprompt.compiler.task_priors import heuristic_task_analysis as _heuristic_task_analysis, normalize_task_family
from topoprompt.schemas import Example, TaskAnalysis, TaskSpec


ANALYZER_SYSTEM_PROMPT = """You are the TopoPrompt task analyzer.
Your job is to infer what prompt-program structures are plausible for a task
AND to write concise, task-specific instruction text for each node type that
would appear in those structures.
You are not solving the task itself.
You must output strict JSON only.

Guidelines:
- Prefer simple structures unless the task clearly requires more.
- Recommend routing only when the input distribution appears heterogeneous.
- Recommend verification only when errors are costly or the task is constraint-heavy.
- Recommend decomposition only when the examples show multi-part reasoning.
- Keep the number of suggested seed templates small.
- In node_instructions, write 1-2 sentence instruction strings that are
  specific to the task — not generic placeholders.  The text should tell the
  model exactly what to do for THIS task, e.g. for a math task the solve node
  should say something like "Solve the arithmetic word problem step by step,
  tracking all intermediate quantities and units." not "Solve the task."
"""


def analyze_task(
    *,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_name: str,
    backend: LLMBackend,
    config: TopoPromptConfig,
) -> TaskAnalysis:
    representative = examples[: config.data.representative_examples_for_analysis]
    examples_json = json.dumps(
        [{"input": example.input, "target": example.target} for example in representative],
        sort_keys=True,
    )
    user_prompt = (
        f"Task description:\n{task_spec.description}\n\n"
        f"Metric:\n{metric_name}\n\n"
        f"Representative examples:\n{examples_json}\n\n"
        "Available seed templates:\n"
        + "\n".join(f"- {seed}" for seed in SEED_LIBRARY)
        + "\n\nReturn JSON with these fields:\n"
        "- task_family\n- output_format\n- needs_reasoning\n- needs_verification\n- needs_decomposition\n"
        "- input_heterogeneity\n- candidate_routes\n- initial_seed_templates\n- analyzer_confidence\n- rationale\n"
        "- node_instructions (object): task-specific instruction text for each relevant node type.\n"
        "  Keys must be node type strings from: direct, plan, decompose, solve, verify, critique, route, format, finalize.\n"
        "  Each value is a 1-2 sentence instruction string specific to this task.\n"
        "  Include at least: direct, solve. Include others only if relevant to the task.\n"
        "  Example for a math task: {\"direct\": \"Solve the arithmetic problem and return the numeric answer.\",\n"
        "  \"solve\": \"Work through the word problem step by step, tracking intermediate values and units.\",\n"
        "  \"verify\": \"Check that the numeric result is dimensionally consistent and satisfies the problem constraints.\",\n"
        "  \"plan\": \"Write a short plan listing the quantities to find and the operations needed.\"}"
    )
    schema = {
        "type": "object",
        "properties": {
            "task_family": {"type": "string"},
            "output_format": {"type": "string"},
            "needs_reasoning": {"type": "boolean"},
            "needs_verification": {"type": "boolean"},
            "needs_decomposition": {"type": "boolean"},
            "input_heterogeneity": {"type": "string"},
            "candidate_routes": {"type": "array", "items": {"type": "object"}},
            "initial_seed_templates": {"type": "array", "items": {"type": "string"}},
            "analyzer_confidence": {"type": "number"},
            "rationale": {"type": "string"},
            "node_instructions": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": (
                    "Task-specific instruction text keyed by node type string "
                    "(direct, plan, decompose, solve, verify, critique, format, finalize). "
                    "Each value is a 1-2 sentence instruction specific to this task."
                ),
            },
        },
        "required": [
            "task_family",
            "output_format",
            "needs_reasoning",
            "needs_verification",
            "needs_decomposition",
            "input_heterogeneity",
            "candidate_routes",
            "initial_seed_templates",
            "analyzer_confidence",
            "rationale",
        ],
    }
    try:
        response = backend.generate_structured(
            system_prompt=ANALYZER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=schema,
            model=config.model.name,
            temperature=0.0,
            max_output_tokens=config.model.max_output_tokens,
        )
        analysis = TaskAnalysis.model_validate(response.structured or json.loads(response.text))
        analysis.task_family = normalize_task_family(analysis.task_family)
        analysis.initial_seed_templates = list(dict.fromkeys(analysis.initial_seed_templates))[:5]
        _validate_analysis(analysis)
        return stabilize_task_analysis(
            task_spec=task_spec,
            examples=representative,
            metric_name=metric_name,
            analysis=analysis,
        )
    except Exception:
        analysis = heuristic_task_analysis(
            task_spec=task_spec,
            examples=representative,
            metric_name=metric_name,
        )
        return stabilize_task_analysis(
            task_spec=task_spec,
            examples=representative,
            metric_name=metric_name,
            analysis=analysis,
        )


def heuristic_task_analysis(
    *,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_name: str | None = None,
) -> TaskAnalysis:
    return _heuristic_task_analysis(task_spec=task_spec, examples=examples, metric_name=metric_name)


def stabilize_task_analysis(
    *,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_name: str,
    analysis: TaskAnalysis,
) -> TaskAnalysis:
    refined = analysis.model_copy(deep=True)
    description = task_spec.description.lower()
    normalized_metric = metric_name.lower()
    target_samples = [str(example.target).strip() for example in examples if example.target is not None][:8]

    family_hint = _infer_task_family_hint(
        description=description,
        metric_name=normalized_metric,
        target_samples=target_samples,
    )
    if family_hint and _should_override_family(
        current=refined.task_family,
        desired=family_hint,
        confidence=refined.analyzer_confidence,
    ):
        refined.task_family = family_hint

    output_format_hint = _infer_output_format_hint(
        description=description,
        metric_name=normalized_metric,
        task_family=refined.task_family,
        target_samples=target_samples,
    )
    if output_format_hint and _should_override_output_format(
        current=refined.output_format,
        desired=output_format_hint,
    ):
        refined.output_format = output_format_hint

    if refined.task_family == "math_reasoning":
        refined.needs_reasoning = True
        refined.needs_verification = True
    elif refined.task_family == "classification" and _targets_look_like_labels(target_samples):
        refined.needs_reasoning = False
        refined.needs_verification = False

    if not refined.initial_seed_templates:
        refined.initial_seed_templates = heuristic_task_analysis(
            task_spec=task_spec,
            examples=examples,
            metric_name=metric_name,
        ).initial_seed_templates
    return refined


def _infer_task_family_hint(*, description: str, metric_name: str, target_samples: list[str]) -> str | None:
    if metric_name in {"numeric", "gsm8k"} or _looks_like_math_task(description):
        return "math_reasoning"
    if _looks_like_summarization_task(description):
        return "summarization"
    if _looks_like_classification_task(description, target_samples):
        return "classification"
    return None


def _infer_output_format_hint(
    *,
    description: str,
    metric_name: str,
    task_family: str,
    target_samples: list[str],
) -> str | None:
    if "json" in description:
        return "json"
    if _description_requests_bullets(description):
        return "bullet_points"
    if metric_name in {"numeric", "gsm8k"} or "numeric answer" in description or "final numeric answer" in description:
        return "numeric"
    if task_family == "classification" and _targets_look_like_labels(target_samples):
        return "label"
    if task_family == "summarization":
        return "paragraph"
    return None


def _should_override_family(*, current: str | None, desired: str, confidence: float) -> bool:
    normalized = (current or "").strip().lower()
    if normalized == desired:
        return False
    if normalized in {"", "mixed", "other", "general", "factual_qa"}:
        return True
    return confidence < 0.35


def _should_override_output_format(*, current: str | None, desired: str) -> bool:
    normalized = (current or "").strip().lower()
    if normalized == desired:
        return False
    if desired in {"json", "bullet_points"}:
        return True
    return normalized in {"", "text", "short_answer", "paragraph", "response", "answer"}


def _looks_like_math_task(description: str) -> bool:
    return any(
        keyword in description
        for keyword in [
            "math",
            "arithmetic",
            "algebra",
            "geometry",
            "word problem",
            "calculate",
            "grade-school",
            "numeric answer",
        ]
    )


def _looks_like_summarization_task(description: str) -> bool:
    return any(keyword in description for keyword in ["summarize", "summarise", "summary", "highlights", "tl;dr"])


def _looks_like_classification_task(description: str, target_samples: list[str]) -> bool:
    if any(keyword in description for keyword in ["classify", "classification", "label", "sentiment"]):
        return True
    return _targets_look_like_labels(target_samples)


def _targets_look_like_labels(target_samples: list[str]) -> bool:
    if not target_samples:
        return False
    normalized = [_normalize_text(sample) for sample in target_samples if sample]
    if not normalized:
        return False
    unique = set(normalized)
    return len(unique) <= 10 and all(len(sample.split()) <= 3 for sample in unique)


def _description_requests_bullets(description: str) -> bool:
    return any(keyword in description for keyword in ["bullet", "bulleted", "bullet-point", "bullet point"])


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _validate_analysis(analysis: TaskAnalysis) -> None:
    if not 0.0 <= analysis.analyzer_confidence <= 1.0:
        raise ValueError("Analyzer confidence must be in [0, 1].")
    if len(analysis.candidate_routes) > 3:
        raise ValueError("Candidate routes must contain at most three routes.")
    invalid = [seed for seed in analysis.initial_seed_templates if seed not in SEED_LIBRARY]
    if invalid:
        raise ValueError(f"Invalid seed templates from analyzer: {invalid}")
