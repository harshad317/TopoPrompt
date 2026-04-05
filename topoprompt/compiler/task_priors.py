from __future__ import annotations

import json
import re
from typing import Any, Sequence

from topoprompt.compiler.seeds import SEED_LIBRARY
from topoprompt.schemas import Example, RouteCandidate, TaskAnalysis, TaskSpec


CANONICAL_TASK_FAMILIES = {
    "classification",
    "code",
    "extraction",
    "factual_qa",
    "generation",
    "instruction_following",
    "math_reasoning",
    "mixed",
    "other",
    "reasoning",
    "summarization",
}

CLASSIFICATION_KEYWORDS = (
    "classif",
    "label",
    "category",
    "categorize",
    "sentiment",
    "toxic",
    "spam",
    "safety",
)
EXTRACTION_KEYWORDS = (
    "extract",
    "entity",
    "entities",
    "field",
    "fields",
    "slot",
    "schema",
    "structured",
    "json",
    "tagging",
)
GENERATION_KEYWORDS = (
    "write",
    "generate",
    "draft",
    "compose",
    "rewrite",
    "translate",
    "respond",
    "caption",
    "headline",
    "email",
    "story",
)
SUMMARY_KEYWORDS = ("summarize", "summary", "tldr", "abstract", "condense")
CODE_KEYWORDS = (
    "code",
    "coding",
    "program",
    "programming",
    "function",
    "script",
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "sql",
    "regex",
    "unit test",
    "debug",
    "refactor",
    "implement",
)
REASONING_KEYWORDS = (
    "reason",
    "reasoning",
    "logic",
    "logical",
    "deduction",
    "infer",
    "proof",
    "prove",
    "multi-step",
    "multistep",
    "analyze",
    "explain",
)
CONSTRAINT_KEYWORDS = (
    "constraint",
    "constraints",
    "follow",
    "instruction",
    "format",
    "schema",
    "must",
    "exactly",
    "strict",
    "return only",
)
NUMERIC_KEYWORDS = (
    "math",
    "arithmetic",
    "calculate",
    "calculation",
    "compute",
    "equation",
    "word problem",
    "numeric",
)
LABEL_LIKE_TARGETS = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "yes",
    "no",
    "true",
    "false",
    "positive",
    "negative",
    "neutral",
    "spam",
    "ham",
    "toxic",
    "safe",
    "entailment",
    "contradiction",
    "unknown",
}


def normalize_task_family(value: str | None) -> str:
    text = re.sub(r"[\s/-]+", "_", str(value or "").strip().lower()).strip("_")
    if not text:
        return "other"
    if text in CANONICAL_TASK_FAMILIES:
        return text
    if any(keyword in text for keyword in ("math", "arithmetic", "numeric", "calculation")):
        return "math_reasoning"
    if any(keyword in text for keyword in ("code", "coding", "program", "function", "sql", "regex")):
        return "code"
    if any(keyword in text for keyword in ("summary", "summariz", "tldr", "abstract")):
        return "summarization"
    if any(keyword in text for keyword in ("extract", "entity", "field", "slot", "structured")):
        return "extraction"
    if any(keyword in text for keyword in ("classif", "label", "multiple_choice", "categor")):
        return "classification"
    if any(keyword in text for keyword in ("generate", "write", "draft", "compose", "rewrite", "translation")):
        return "generation"
    if any(keyword in text for keyword in ("instruction", "constraint", "follow", "format")):
        return "instruction_following"
    if any(keyword in text for keyword in ("reason", "logic", "deduction", "proof")):
        return "reasoning"
    if any(keyword in text for keyword in ("qa", "question_answer", "knowledge", "fact")):
        return "factual_qa"
    if "mixed" in text:
        return "mixed"
    return "other"


def heuristic_task_analysis(
    *,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_name: str | None = None,
) -> TaskAnalysis:
    description = task_spec.description.strip()
    metric_hint = _normalize_metric_hint(metric_name)
    input_texts = [_stringify_value(example.input) for example in examples]
    target_texts = [_stringify_value(example.target) for example in examples if example.target is not None]
    combined_inputs = " ".join(input_texts).lower()
    combined_targets = " ".join(target_texts).lower()
    prompt_context = f"{description.lower()} {combined_inputs}".strip()
    combined = f"{prompt_context} {combined_targets}".strip()

    choices_present = any(
        isinstance(example.input, dict) and isinstance(example.input.get("choices"), list) and example.input.get("choices")
        for example in examples
    )
    input_shapes = {tuple(sorted(example.input.keys())) for example in examples if isinstance(example.input, dict)}
    structured_output = _looks_structured_output(
        output_schema=task_spec.output_schema,
        targets=[example.target for example in examples],
        description=description,
    )
    question_like = any(
        isinstance(example.input, dict) and any(key in example.input for key in ("question", "prompt", "query"))
        for example in examples
    )
    avg_input_words = _average_word_count(input_texts)
    avg_target_words = _average_word_count(target_texts)

    looks_code = _contains_any(prompt_context, CODE_KEYWORDS) or _looks_like_code_blob(combined_inputs) or _looks_like_code_blob(combined_targets)
    looks_summarization = _contains_any(prompt_context, SUMMARY_KEYWORDS) or (
        avg_input_words >= 40 and 0 < avg_target_words < avg_input_words * 0.6
    )
    looks_extraction = _contains_any(prompt_context, EXTRACTION_KEYWORDS) or structured_output
    looks_instruction_following = metric_hint == "ifeval" or _contains_any(prompt_context, CONSTRAINT_KEYWORDS)
    math_signal = _contains_any(prompt_context, NUMERIC_KEYWORDS) or bool(re.search(r"\d+\s*[\+\-\*/]", combined_inputs))
    reasoning_signal = math_signal or _contains_any(prompt_context, REASONING_KEYWORDS) or bool(re.search(r"\bwhy\b|\bhow\b|\bexplain\b", combined_inputs))

    target_labels = {
        _normalize_label_text(example.target)
        for example in examples
        if example.target is not None and _normalize_label_text(example.target)
    }
    target_prefix_labels = {
        _leading_label_text(example.target)
        for example in examples
        if example.target is not None and _leading_label_text(example.target)
    }
    label_like_targets = bool(target_labels) and target_labels.issubset(LABEL_LIKE_TARGETS)
    label_like_prefixes = bool(target_prefix_labels) and target_prefix_labels.issubset(LABEL_LIKE_TARGETS)
    looks_classification = (
        metric_hint == "multiple_choice"
        or choices_present
        or _contains_any(prompt_context, CLASSIFICATION_KEYWORDS)
        or label_like_targets
        or label_like_prefixes
    )
    looks_generation = (
        (_contains_any(prompt_context, GENERATION_KEYWORDS) or avg_target_words >= 24)
        and not structured_output
        and not looks_classification
    )

    family_signals = {
        "code": looks_code,
        "summarization": looks_summarization,
        "extraction": looks_extraction,
        "generation": looks_generation,
        "instruction_following": looks_instruction_following,
        "classification": looks_classification,
        "math_reasoning": math_signal,
        "reasoning": reasoning_signal,
    }
    strong_family_count = sum(
        int(family_signals[family])
        for family in ("code", "summarization", "extraction", "generation", "instruction_following", "classification")
    )

    if strong_family_count >= 2 and not looks_code and not looks_summarization:
        task_family = "mixed"
    elif looks_code:
        task_family = "code"
    elif looks_summarization:
        task_family = "summarization"
    elif looks_extraction:
        task_family = "extraction"
    elif looks_classification:
        task_family = "classification"
    elif looks_generation:
        task_family = "generation"
    elif looks_instruction_following and not reasoning_signal:
        task_family = "instruction_following"
    elif math_signal:
        task_family = "math_reasoning"
    elif reasoning_signal:
        task_family = "reasoning"
    elif question_like or metric_hint == "exact_match":
        task_family = "factual_qa"
    else:
        task_family = "other"

    output_format = _infer_output_format(
        task_family=task_family,
        description=description,
        structured_output=structured_output,
        choices_present=choices_present,
    )
    needs_reasoning = reasoning_signal or task_family in {"code", "math_reasoning", "reasoning"}
    needs_verification = (
        task_family in {"code", "instruction_following"}
        or math_signal
        or (task_family == "extraction" and output_format == "json")
        or _contains_any(combined, ("verify", "verification", "check", "validate", "constraint"))
    )
    needs_decomposition = (
        task_family in {"math_reasoning", "reasoning", "code"}
        and (
            _contains_any(combined, ("decompose", "subquestion", "break down"))
            or (" and " in combined_inputs and avg_input_words >= 20)
            or (" then " in combined_inputs and avg_input_words >= 20)
        )
    )

    heterogeneity_score = 0
    if len(input_shapes) > 1:
        heterogeneity_score += 2
    if choices_present:
        heterogeneity_score += 1
    if needs_reasoning and task_family in {"classification", "factual_qa", "reasoning", "math_reasoning"}:
        heterogeneity_score += 1
    if _contains_any(combined_inputs, ("table", "chart", "passage", "options", "code block")):
        heterogeneity_score += 1
    if heterogeneity_score >= 3:
        input_heterogeneity = "high"
    elif heterogeneity_score >= 1:
        input_heterogeneity = "medium"
    else:
        input_heterogeneity = "low"

    candidate_routes = _recommend_candidate_routes(
        task_family=task_family,
        input_heterogeneity=input_heterogeneity,
        needs_reasoning=needs_reasoning,
        needs_decomposition=needs_decomposition,
    )

    initial_seed_templates = _recommend_seed_templates(
        task_family=task_family,
        output_format=output_format,
        needs_reasoning=needs_reasoning,
        needs_verification=needs_verification,
        needs_decomposition=needs_decomposition,
        input_heterogeneity=input_heterogeneity,
    )
    rationale = (
        "Heuristic fallback inferred "
        f"task_family={task_family}, output_format={output_format}, "
        f"and seeds={', '.join(initial_seed_templates) or 'direct_finalize'}."
    )

    return TaskAnalysis(
        task_family=task_family,
        output_format=output_format,
        needs_reasoning=needs_reasoning,
        needs_verification=needs_verification,
        needs_decomposition=needs_decomposition,
        input_heterogeneity=input_heterogeneity,
        candidate_routes=candidate_routes,
        initial_seed_templates=initial_seed_templates,
        analyzer_confidence=0.46,
        rationale=rationale,
    )


def _recommend_candidate_routes(
    *,
    task_family: str,
    input_heterogeneity: str,
    needs_reasoning: bool,
    needs_decomposition: bool,
) -> list[RouteCandidate]:
    """Return task-family-specific route candidates.

    Generic "direct vs solve" labels are replaced with semantically meaningful
    branch descriptions so the route LLM makes more accurate decisions at
    inference time.  Routing is only recommended when input heterogeneity
    warrants it.
    """
    if input_heterogeneity == "low":
        return []

    if task_family == "math_reasoning":
        routes = [
            RouteCandidate(
                label="direct",
                description=(
                    "Use for simple arithmetic or single-step calculations where "
                    "the answer can be read off immediately (e.g. 'What is 12 × 4?')."
                ),
            ),
            RouteCandidate(
                label="solve",
                description=(
                    "Use for multi-step word problems that require setting up "
                    "equations, tracking intermediate quantities, or reasoning "
                    "across multiple sentences."
                ),
            ),
        ]
        if needs_decomposition:
            routes.append(RouteCandidate(
                label="decompose",
                description=(
                    "Use for complex problems with multiple sub-questions or "
                    "where breaking into parts before solving reduces errors."
                ),
            ))
        return routes

    if task_family == "reasoning":
        return [
            RouteCandidate(
                label="lookup",
                description=(
                    "Use when the question can be answered by recalling a direct "
                    "fact or single inference step (e.g. 'Is X a Y?')."
                ),
            ),
            RouteCandidate(
                label="solve",
                description=(
                    "Use for multi-hop reasoning where several facts must be "
                    "chained together before arriving at the answer."
                ),
            ),
        ]

    if task_family == "classification":
        return [
            RouteCandidate(
                label="direct",
                description=(
                    "Use for clear-cut examples where the correct label is "
                    "unambiguous from surface-level signals."
                ),
            ),
            RouteCandidate(
                label="solve",
                description=(
                    "Use for ambiguous or nuanced examples that require careful "
                    "reading, comparison of candidates, or contextual reasoning."
                ),
            ),
        ]

    if task_family == "factual_qa":
        return [
            RouteCandidate(
                label="direct",
                description="Use for factual questions with a single well-known answer.",
            ),
            RouteCandidate(
                label="solve",
                description=(
                    "Use for questions requiring comparison, calculation, or "
                    "retrieval from a provided passage."
                ),
            ),
        ]

    if task_family in {"extraction", "instruction_following"}:
        return [
            RouteCandidate(
                label="direct",
                description="Use for structured inputs with well-defined fields to extract.",
            ),
            RouteCandidate(
                label="solve",
                description=(
                    "Use for unstructured or noisy inputs where fields must be "
                    "inferred or the instructions have multiple constraints."
                ),
            ),
        ]

    if task_family == "mixed":
        return [
            RouteCandidate(
                label="direct",
                description="Use for items that can be answered with a straightforward response.",
            ),
            RouteCandidate(
                label="solve",
                description="Use for items that require analysis, reasoning, or multi-step work.",
            ),
        ]

    # Fallback for remaining families (generation, summarization, code, other)
    # only recommend routing when heterogeneity is clearly high.
    if input_heterogeneity == "high":
        return [
            RouteCandidate(
                label="direct",
                description="Use for straightforward items that can be answered directly.",
            ),
            RouteCandidate(
                label="solve",
                description="Use for items that need analysis, reasoning, or careful checking.",
            ),
        ]

    return []


def heuristic_task_analysis_from_prompt(*, user_prompt: str) -> TaskAnalysis:
    description = _extract_prompt_section(user_prompt, "Task description:", "Metric:")
    metric_name = _extract_prompt_section(user_prompt, "Metric:", "Representative examples:")
    examples_blob = _extract_prompt_section(user_prompt, "Representative examples:", "Available seed templates:")
    try:
        raw_examples = json.loads(examples_blob) if examples_blob else []
    except json.JSONDecodeError:
        raw_examples = []
    if isinstance(raw_examples, dict):
        raw_examples = [raw_examples]
    elif not isinstance(raw_examples, list):
        raw_examples = []
    return heuristic_task_analysis_from_payloads(
        task_description=description or "Generic task",
        examples_payloads=raw_examples,
        metric_name=metric_name,
    )


def heuristic_task_analysis_from_payloads(
    *,
    task_description: str,
    examples_payloads: Sequence[Any],
    metric_name: str | None = None,
    output_schema: dict[str, Any] | None = None,
) -> TaskAnalysis:
    task_spec = TaskSpec(
        task_id="heuristic_task",
        description=task_description,
        input_schema={},
        output_schema=output_schema or {},
    )
    examples = _coerce_examples(examples_payloads)
    return heuristic_task_analysis(task_spec=task_spec, examples=examples, metric_name=metric_name)


def _recommend_seed_templates(
    *,
    task_family: str,
    output_format: str,
    needs_reasoning: bool,
    needs_verification: bool,
    needs_decomposition: bool,
    input_heterogeneity: str,
) -> list[str]:
    seeds = ["direct_finalize"]
    if output_format == "json" and task_family in {"extraction", "instruction_following", "mixed"}:
        seeds.append("format_finalize")

    match task_family:
        case "math_reasoning":
            seeds.extend(["plan_solve_finalize", "solve_verify_finalize"])
            if needs_decomposition:
                seeds.append("decompose_solve_finalize")
            if needs_verification:
                seeds.append("plan_solve_verify_finalize")
            if input_heterogeneity != "low":
                seeds.append("route_direct_or_solve_finalize")
        case "reasoning":
            seeds.append("solve_verify_finalize" if needs_verification else "plan_solve_finalize")
            if needs_decomposition:
                seeds.append("decompose_solve_finalize")
            if input_heterogeneity != "low":
                seeds.append("route_direct_or_solve_finalize")
        case "classification":
            if input_heterogeneity != "low":
                seeds.append("route_direct_or_solve_finalize")
            if needs_reasoning:
                seeds.append("solve_verify_finalize")
                seeds.append("direct_self_consistency_x3")
        case "extraction":
            seeds.append("format_finalize")
            if input_heterogeneity == "high" and needs_reasoning:
                seeds.append("route_direct_or_solve_finalize")
        case "generation" | "summarization":
            seeds.append("critique_revise_finalize")
            if needs_decomposition:
                seeds.append("plan_solve_finalize")
        case "code":
            seeds.extend(["critique_revise_finalize", "solve_verify_finalize"])
            if needs_reasoning or needs_decomposition:
                seeds.append("plan_solve_verify_finalize")
        case "instruction_following":
            if output_format == "json":
                seeds.append("format_finalize")
            if needs_verification:
                seeds.append("solve_verify_finalize")
        case "factual_qa":
            if input_heterogeneity != "low":
                seeds.append("route_direct_or_solve_finalize")
            if needs_reasoning:
                seeds.append("solve_verify_finalize")
        case "mixed":
            seeds.extend(["format_finalize", "critique_revise_finalize"])
            if needs_reasoning:
                seeds.append("solve_verify_finalize")
            if input_heterogeneity != "low":
                seeds.append("route_direct_or_solve_finalize")
        case _:
            if output_format == "json":
                seeds.append("format_finalize")
            if needs_reasoning:
                seeds.append("plan_solve_finalize")
            if needs_verification:
                seeds.append("solve_verify_finalize")

    deduped = list(dict.fromkeys(seed for seed in seeds if seed in SEED_LIBRARY))
    return deduped[:5]


def _infer_output_format(
    *,
    task_family: str,
    description: str,
    structured_output: bool,
    choices_present: bool,
) -> str:
    lowered = description.lower()
    if structured_output or any(keyword in lowered for keyword in ("json", "yaml", "xml", "structured")):
        return "json"
    if task_family == "code":
        return "code"
    if task_family == "classification" or choices_present:
        return "label"
    if task_family in {"generation", "summarization"}:
        return "paragraph"
    if "list" in lowered or "bullet" in lowered:
        return "list"
    return "short_answer"


def _looks_structured_output(*, output_schema: dict[str, Any], targets: Sequence[Any], description: str) -> bool:
    schema_type = str(output_schema.get("type", "")).lower()
    if schema_type in {"array", "object"}:
        return True
    if any(_looks_like_json_value(target) for target in targets if target is not None):
        return True
    lowered = description.lower()
    return any(keyword in lowered for keyword in ("json", "structured", "schema", "fields"))


def _coerce_examples(raw_examples: Sequence[Any]) -> list[Example]:
    examples: list[Example] = []
    for index, payload in enumerate(raw_examples, start=1):
        if isinstance(payload, Example):
            examples.append(payload)
            continue
        if not isinstance(payload, dict):
            examples.append(
                Example(
                    example_id=f"example_{index}",
                    input={"prompt": str(payload)},
                    target=None,
                    metadata={},
                )
            )
            continue
        raw_input = payload.get("input")
        if isinstance(raw_input, dict):
            input_payload = dict(raw_input)
        elif raw_input is not None:
            input_payload = {"prompt": raw_input}
        elif "question" in payload:
            input_payload = {"question": payload.get("question")}
        elif "prompt" in payload:
            input_payload = {"prompt": payload.get("prompt")}
        else:
            input_payload = {
                key: value
                for key, value in payload.items()
                if key not in {"answer", "example_id", "id", "label", "metadata", "target"}
            }
        if "choices" in payload and "choices" not in input_payload:
            input_payload["choices"] = payload.get("choices")
        examples.append(
            Example(
                example_id=str(payload.get("example_id") or payload.get("id") or f"example_{index}"),
                input=input_payload,
                target=payload.get("target", payload.get("answer", payload.get("label"))),
                metadata=dict(payload.get("metadata", {}) or {}),
            )
        )
    return examples


def _extract_prompt_section(text: str, start_header: str, end_header: str) -> str:
    pattern = re.compile(rf"{re.escape(start_header)}\n(.*?)(?:\n\n{re.escape(end_header)}|\Z)", flags=re.S)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _average_word_count(rows: Sequence[str]) -> float:
    if not rows:
        return 0.0
    return sum(len(re.findall(r"\w+", row)) for row in rows) / len(rows)


def _normalize_label_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _leading_label_text(value: Any) -> str:
    text = _normalize_label_text(value)
    match = re.match(r"([a-z]+)", text)
    return match.group(1) if match else ""


def _normalize_metric_hint(metric_name: str | None) -> str:
    normalized = (metric_name or "exact_match").strip().lower()
    aliases = {
        "accuracy": "exact_match",
        "gsm8k": "numeric",
        "instruction_following": "ifeval",
        "mmlu": "multiple_choice",
    }
    return aliases.get(normalized, normalized)


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _looks_like_json_value(value: Any) -> bool:
    if isinstance(value, (dict, list)):
        return True
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or text[:1] not in {"{", "["}:
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _looks_like_code_blob(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in ("```", "def ", "class ", "return ", "import ", "function ", "const ", "let ", "select ")
    )
