from __future__ import annotations

import json
import re
from typing import Any, Callable

from langdetect import LangDetectException, detect

from topoprompt.schemas import Example


MetricFn = Callable[[Any, Example], float]


def metric_for_name(name: str | None) -> MetricFn:
    normalized = (name or "exact_match").lower()
    if normalized in {"exact_match", "accuracy"}:
        return exact_match_metric
    if normalized in {"numeric", "gsm8k"}:
        return numeric_metric
    if normalized in {"multiple_choice", "mmlu"}:
        return multiple_choice_metric
    if normalized in {"bbh"}:
        return bbh_metric
    if normalized in {"ifeval", "instruction_following"}:
        return ifeval_metric
    return exact_match_metric


def exact_match_metric(prediction: Any, example: Example) -> float:
    return 1.0 if _normalize_text(prediction) == _normalize_text(example.target) else 0.0


def numeric_metric(prediction: Any, example: Example) -> float:
    pred_value = _extract_reference_number(prediction)
    target_value = _extract_reference_number(example.target)
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


def bbh_metric(prediction: Any, example: Example) -> float:
    if example.input.get("choices"):
        return multiple_choice_metric(prediction, example)
    return exact_match_metric(prediction, example)


def ifeval_metric(prediction: Any, example: Example) -> float:
    text = str(prediction or "")
    instruction_ids = example.metadata.get("instruction_id_list") or []
    instruction_kwargs = example.metadata.get("instruction_kwargs") or []
    if instruction_ids:
        prompt = str(example.input.get("prompt") or example.input.get("question") or "")
        scores = [
            _ifeval_instruction_metric(text, prompt, instruction_id, kwargs or {})
            for instruction_id, kwargs in zip(instruction_ids, instruction_kwargs)
        ]
        return sum(scores) / len(scores) if scores else 0.0

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


def _ifeval_instruction_metric(text: str, prompt: str, instruction_id: str, kwargs: dict[str, Any]) -> float:
    stripped = text.strip()
    lowered = stripped.lower()
    match instruction_id:
        case "punctuation:no_comma":
            return 1.0 if "," not in text else 0.0
        case "length_constraints:number_words":
            return float(_relation_holds(_count_words(text), kwargs.get("relation"), kwargs.get("num_words")))
        case "length_constraints:number_sentences":
            return float(_relation_holds(_count_sentences(text), kwargs.get("relation"), kwargs.get("num_sentences")))
        case "length_constraints:number_paragraphs":
            return float(_count_paragraphs(text) == int(kwargs.get("num_paragraphs", -1)))
        case "length_constraints:nth_paragraph_first_word":
            paragraphs = _paragraphs(text)
            nth = int(kwargs.get("nth_paragraph", 0))
            expected_count = int(kwargs.get("num_paragraphs", 0))
            if expected_count and len(paragraphs) != expected_count:
                return 0.0
            if nth < 1 or nth > len(paragraphs):
                return 0.0
            words = re.findall(r"[A-Za-z']+", paragraphs[nth - 1].lower())
            return 1.0 if words and words[0] == str(kwargs.get("first_word", "")).lower() else 0.0
        case "keywords:forbidden_words":
            forbidden = [str(word).lower() for word in kwargs.get("forbidden_words", [])]
            return 1.0 if all(word not in lowered for word in forbidden) else 0.0
        case "keywords:existence":
            keywords = [str(word).lower() for word in kwargs.get("keywords", [])]
            return 1.0 if all(word in lowered for word in keywords) else 0.0
        case "keywords:frequency":
            keyword = str(kwargs.get("keyword", "")).lower()
            frequency = lowered.count(keyword) if keyword else 0
            return float(_relation_holds(frequency, kwargs.get("relation"), kwargs.get("frequency")))
        case "keywords:letter_frequency":
            letter = str(kwargs.get("letter", ""))
            frequency = text.count(letter)
            return float(_relation_holds(frequency, kwargs.get("let_relation"), kwargs.get("let_frequency")))
        case "detectable_format:number_highlighted_sections":
            highlights = re.findall(r"(?<!\*)\*{1,2}[^*\n][^*\n]*\*{1,2}(?!\*)", text)
            return 1.0 if len(highlights) >= int(kwargs.get("num_highlights", 0)) else 0.0
        case "detectable_format:number_bullet_lists":
            bullets = re.findall(r"(?m)^\s*[-*+]\s+\S", text)
            return 1.0 if len(bullets) >= int(kwargs.get("num_bullets", 0)) else 0.0
        case "detectable_format:title":
            return 1.0 if re.search(r"<<[^<>]+>>", text) else 0.0
        case "detectable_format:json_format":
            return 1.0 if _looks_like_json_block(stripped) else 0.0
        case "detectable_format:multiple_sections":
            splitter = re.escape(str(kwargs.get("section_spliter", "SECTION")))
            pattern = rf"(?m)^\s*{splitter}\s+\d+\b"
            return 1.0 if len(re.findall(pattern, text)) >= int(kwargs.get("num_sections", 0)) else 0.0
        case "detectable_format:constrained_response":
            allowed = _extract_allowed_phrases(prompt)
            if not allowed:
                return 0.0
            normalized_allowed = [_normalize_text(item) for item in allowed]
            normalized_text = _normalize_text(text)
            if "exactly one of" in prompt.lower():
                return 1.0 if normalized_text in normalized_allowed else 0.0
            return 1.0 if any(item in normalized_text for item in normalized_allowed) else 0.0
        case "detectable_content:number_placeholders":
            placeholders = re.findall(r"\[[^\[\]\n]+\]", text)
            return 1.0 if len(placeholders) >= int(kwargs.get("num_placeholders", 0)) else 0.0
        case "detectable_content:postscript":
            marker = str(kwargs.get("postscript_marker", "")).lower()
            return 1.0 if marker and marker in lowered else 0.0
        case "change_case:english_lowercase":
            return 1.0 if not re.search(r"[A-Z]", text) else 0.0
        case "change_case:english_capital":
            return 1.0 if not re.search(r"[a-z]", text) and re.search(r"[A-Z]", text) else 0.0
        case "change_case:capital_word_frequency":
            capital_words = re.findall(r"\b[A-Z]{2,}\b", text)
            return float(_relation_holds(len(capital_words), kwargs.get("capital_relation"), kwargs.get("capital_frequency")))
        case "combination:repeat_prompt":
            return 1.0 if stripped.startswith(str(kwargs.get("prompt_to_repeat", "")).strip()) else 0.0
        case "combination:two_responses":
            parts = [part.strip() for part in text.split("******") if part.strip()]
            return 1.0 if len(parts) == 2 else 0.0
        case "startend:quotation":
            return 1.0 if stripped.startswith('"') and stripped.endswith('"') else 0.0
        case "startend:end_checker":
            return 1.0 if stripped.endswith(str(kwargs.get("end_phrase", "")).strip()) else 0.0
        case "language:response_language":
            expected = str(kwargs.get("language", "")).lower()
            return 1.0 if expected and _detect_language(text) == expected else 0.0
        case _:
            return 0.0


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


def _extract_reference_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    marker_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if marker_match:
        return float(marker_match.group(1))
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return float(matches[-1])


def _relation_holds(value: int, relation: Any, threshold: Any) -> bool:
    if threshold is None:
        return False
    relation_text = _normalize_text(relation)
    target = int(threshold)
    if relation_text in {"at least", ">=", "greater than or equal to"}:
        return value >= target
    if relation_text in {"less than", "<"}:
        return value < target
    if relation_text in {"at most", "<=", "less than or equal to"}:
        return value <= target
    if relation_text in {"more than", "greater than", ">"}:
        return value > target
    if relation_text in {"exactly", "equal to", "equals", "=="}:
        return value == target
    return value == target


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_sentences(text: str) -> int:
    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text.strip()) if chunk.strip()]
    return len(sentences)


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()]


def _count_paragraphs(text: str) -> int:
    return len(_paragraphs(text))


def _extract_allowed_phrases(prompt: str) -> list[str]:
    return [phrase.strip() for phrase in re.findall(r'["“”](.+?)["“”]', prompt) if phrase.strip()]


def _looks_like_json_block(text: str) -> bool:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        json.loads(candidate)
    except Exception:
        return False
    return True


def _detect_language(text: str) -> str | None:
    try:
        return detect(text)
    except LangDetectException:
        return None
