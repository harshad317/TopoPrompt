from __future__ import annotations

import json
import re

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.config import TopoPromptConfig
from topoprompt.compiler.seeds import SEED_LIBRARY
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
        _validate_analysis(analysis)
        return analysis
    except Exception:
        return heuristic_task_analysis(task_spec=task_spec, examples=representative)


def heuristic_task_analysis(*, task_spec: TaskSpec, examples: list[Example]) -> TaskAnalysis:
    combined = " ".join(json.dumps(example.input, sort_keys=True) for example in examples).lower()
    description = task_spec.description.lower()
    needs_reasoning = bool(re.search(r"\d+\s*[\+\-\*/]", combined)) or "math" in description or "reason" in description
    needs_verification = needs_reasoning or "constraint" in description or "instruction" in description
    needs_decomposition = " and " in combined and len(combined) > 100
    heterogeneity = "high" if any(word in combined for word in ["option", "choice"]) and needs_reasoning else "medium" if needs_reasoning else "low"
    seeds = ["direct_finalize", "plan_solve_finalize" if needs_reasoning else "solve_verify_finalize"]
    if needs_verification:
        seeds.append("solve_verify_finalize")
    if heterogeneity != "low":
        seeds.append("route_direct_or_solve_finalize")
    return TaskAnalysis(
        task_family="math_reasoning" if needs_reasoning else "instruction_following" if "instruction" in description else "factual_qa",
        output_format="json" if "json" in description else "short_answer",
        needs_reasoning=needs_reasoning,
        needs_verification=needs_verification,
        needs_decomposition=needs_decomposition,
        input_heterogeneity=heterogeneity,
        candidate_routes=(
            [
                {"label": "direct", "description": "Use for direct factual items."},
                {"label": "solve", "description": "Use for reasoning items."},
            ]
            if heterogeneity != "low"
            else []
        ),
        initial_seed_templates=list(dict.fromkeys(seed for seed in seeds if seed)),
        analyzer_confidence=0.51,
        rationale="Heuristic fallback analysis.",
    )


def _validate_analysis(analysis: TaskAnalysis) -> None:
    if not 0.0 <= analysis.analyzer_confidence <= 1.0:
        raise ValueError("Analyzer confidence must be in [0, 1].")
    if len(analysis.candidate_routes) > 3:
        raise ValueError("Candidate routes must contain at most three routes.")
    invalid = [seed for seed in analysis.initial_seed_templates if seed not in SEED_LIBRARY]
    if invalid:
        raise ValueError(f"Invalid seed templates from analyzer: {invalid}")

