from __future__ import annotations

import json

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.config import TopoPromptConfig
from topoprompt.compiler.seeds import SEED_LIBRARY
from topoprompt.compiler.task_priors import heuristic_task_analysis as _heuristic_task_analysis, normalize_task_family
from topoprompt.schemas import Example, TaskAnalysis, TaskSpec


ANALYZER_SYSTEM_PROMPT = """You are the TopoPrompt task analyzer.
Your job is to infer what prompt-program structures are plausible for a task.
You are not solving the task itself.
You must output strict JSON only.

Guidelines:
- Prefer simple structures unless the task clearly requires more.
- Recommend routing only when the input distribution appears heterogeneous.
- Recommend verification only when errors are costly or the task is constraint-heavy.
- Recommend decomposition only when the examples show multi-part reasoning.
- Keep the number of suggested seed templates small.
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
        "- input_heterogeneity\n- candidate_routes\n- initial_seed_templates\n- analyzer_confidence\n- rationale"
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
        return analysis
    except Exception:
        return heuristic_task_analysis(task_spec=task_spec, examples=representative, metric_name=metric_name)


def heuristic_task_analysis(
    *,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_name: str | None = None,
) -> TaskAnalysis:
    return _heuristic_task_analysis(task_spec=task_spec, examples=examples, metric_name=metric_name)


def _validate_analysis(analysis: TaskAnalysis) -> None:
    if not 0.0 <= analysis.analyzer_confidence <= 1.0:
        raise ValueError("Analyzer confidence must be in [0, 1].")
    if len(analysis.candidate_routes) > 3:
        raise ValueError("Candidate routes must contain at most three routes.")
    invalid = [seed for seed in analysis.initial_seed_templates if seed not in SEED_LIBRARY]
    if invalid:
        raise ValueError(f"Invalid seed templates from analyzer: {invalid}")
