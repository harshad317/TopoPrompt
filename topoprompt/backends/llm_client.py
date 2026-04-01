from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel, Field


class BackendResponse(BaseModel):
    text: str
    structured: dict[str, Any] | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    request_id: str | None = None


class LLMBackend(ABC):
    @abstractmethod
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
    ) -> BackendResponse:
        raise NotImplementedError

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
    ) -> BackendResponse:
        schema_prompt = (
            f"{user_prompt}\n\nReturn strict JSON that matches this schema:\n"
            f"{json.dumps(schema, sort_keys=True)}"
        )
        return self.generate_text(
            system_prompt=system_prompt,
            user_prompt=schema_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def repair_json(self, *, raw_output: str, schema: dict[str, Any], model: str) -> BackendResponse:
        prompt = (
            "Convert the following model output into valid JSON matching this schema.\n"
            "Do not add information that is not present.\n"
            "If a field is missing, set it to null.\n\n"
            f"Schema:\n{json.dumps(schema, sort_keys=True)}\n\n"
            f"Raw output:\n{raw_output}"
        )
        return self.generate_text(system_prompt="You repair malformed JSON.", user_prompt=prompt, model=model, temperature=0.0)

    def embed_text(self, text: str, *, model: str) -> list[float]:
        return [float((sum(ord(ch) for ch in text) % 1000) / 1000.0)]


class FakeBackend(LLMBackend):
    def __init__(
        self,
        *,
        structured_handler: Callable[[str, str, dict[str, Any]], dict[str, Any]] | None = None,
        text_handler: Callable[[str, str], str] | None = None,
    ) -> None:
        self.structured_handler = structured_handler
        self.text_handler = text_handler
        self.request_counter = 0

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
    ) -> BackendResponse:
        start = time.perf_counter()
        self.request_counter += 1
        if self.text_handler is not None:
            text = self.text_handler(system_prompt, user_prompt)
        else:
            text = self._default_text(system_prompt, user_prompt)
        latency_ms = int((time.perf_counter() - start) * 1000)
        token_usage = {"input_tokens": len((system_prompt + "\n" + user_prompt).split()), "output_tokens": len(text.split())}
        return BackendResponse(text=text, latency_ms=latency_ms, token_usage=token_usage, request_id=f"fake-{self.request_counter}")

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
    ) -> BackendResponse:
        start = time.perf_counter()
        self.request_counter += 1
        if self.structured_handler is not None:
            structured = self.structured_handler(system_prompt, user_prompt, schema)
        else:
            structured = self._default_structured(system_prompt, user_prompt, schema)
        text = json.dumps(structured)
        latency_ms = int((time.perf_counter() - start) * 1000)
        token_usage = {"input_tokens": len((system_prompt + "\n" + user_prompt).split()), "output_tokens": len(text.split())}
        return BackendResponse(
            text=text,
            structured=structured,
            latency_ms=latency_ms,
            token_usage=token_usage,
            request_id=f"fake-{self.request_counter}",
        )

    def repair_json(self, *, raw_output: str, schema: dict[str, Any], model: str) -> BackendResponse:
        self.request_counter += 1
        structured = self._coerce_to_schema(raw_output, schema)
        return BackendResponse(
            text=json.dumps(structured),
            structured=structured,
            latency_ms=0,
            token_usage={"input_tokens": len(raw_output.split()), "output_tokens": len(json.dumps(structured).split())},
            request_id=f"fake-{self.request_counter}",
        )

    def embed_text(self, text: str, *, model: str) -> list[float]:
        digest = sum((index + 1) * ord(ch) for index, ch in enumerate(text[:64]))
        return [float((digest + offset) % 997) / 997.0 for offset in range(8)]

    def _default_text(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(self._default_structured(system_prompt, user_prompt, {"type": "object", "properties": {}}))

    def _default_structured(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        context = _extract_context_json(user_prompt)
        properties = schema.get("properties", {})
        field_names = set(properties)
        if "task_family" in field_names and "initial_seed_templates" in field_names:
            return self._analysis_response(context, user_prompt)
        if "branch" in field_names:
            branch = self._choose_branch(context, user_prompt, properties)
            return {"branch": branch, "confidence": 0.9, "reason": "heuristic route choice"}
        if "verification_result" in field_names or "verdict" in field_names:
            verdict = self._verify_answer(context)
            payload = {"verification_result": verdict, "verdict": verdict, "explanation": f"heuristic {verdict}"}
            return {key: payload.get(key) for key in field_names}
        if "plan" in field_names:
            return {"plan": "1. Identify the task.\n2. Solve it carefully.\n3. Return the answer."}
        if "subquestions" in field_names:
            question = _extract_primary_text(context)
            parts = [segment.strip(" ?.") for segment in re.split(r"\band\b", question) if segment.strip()]
            subquestions = [f"What is {part}?" if not part.lower().startswith("what") else part for part in parts[:3]]
            if not subquestions:
                subquestions = [question]
            return {
                "subquestions": subquestions,
                "subquestion_answers": [],
                "decomposition_context": " | ".join(subquestions),
            }
        answer = self._solve(context)
        payload = {
            "candidate_answer": answer,
            "answer": answer,
            "final_answer": answer,
            "rationale": "heuristic solution",
            "reasoning": "heuristic reasoning",
            "format_notes": "plain text",
        }
        if not field_names:
            return payload
        return {key: payload.get(key) for key in field_names}

    def _analysis_response(self, context: dict[str, Any], user_prompt: str) -> dict[str, Any]:
        description = user_prompt.lower()
        examples_text = json.dumps(context, sort_keys=True).lower()
        needs_reasoning = bool(re.search(r"\d+\s*[\+\-\*/]", examples_text)) or "reason" in description or "math" in description
        needs_verification = needs_reasoning or "constraint" in description or "follow" in description
        needs_decomposition = "and" in examples_text and len(examples_text) > 80
        heterogeneity = "high" if "choice" in examples_text and needs_reasoning else "medium" if needs_reasoning else "low"
        seeds = ["direct_finalize", "solve_verify_finalize" if needs_verification else "plan_solve_finalize"]
        if needs_reasoning:
            seeds.append("plan_solve_finalize")
        if heterogeneity != "low":
            seeds.append("route_direct_or_solve_finalize")
        return {
            "task_family": "math_reasoning" if needs_reasoning else "instruction_following" if "instruction" in description else "factual_qa",
            "output_format": "json" if "json" in description else "short_answer",
            "needs_reasoning": needs_reasoning,
            "needs_verification": needs_verification,
            "needs_decomposition": needs_decomposition,
            "input_heterogeneity": heterogeneity,
            "candidate_routes": [
                {"label": "direct", "description": "Use for direct or factual items."},
                {"label": "solve", "description": "Use for multi-step or arithmetic items."},
            ]
            if heterogeneity != "low"
            else [],
            "initial_seed_templates": list(dict.fromkeys(seed for seed in seeds if seed)),
            "analyzer_confidence": 0.72,
            "rationale": "heuristic analyzer output",
        }

    def _choose_branch(self, context: dict[str, Any], user_prompt: str, properties: dict[str, Any]) -> str:
        labels = []
        schema_prompt = user_prompt.lower()
        for label in ["direct", "solve", "reasoning", "plan_solve", "verify"]:
            if label in schema_prompt:
                labels.append(label)
        detected = _extract_primary_text(context).lower()
        wants_reasoning = bool(re.search(r"\d+\s*[\+\-\*/]", detected)) or any(word in detected for word in ["why", "steps", "calculate"])
        preferred = "solve" if wants_reasoning else "direct"
        if preferred in labels:
            return preferred
        if wants_reasoning:
            for label in labels:
                if "solve" in label or "reason" in label or "plan" in label:
                    return label
        for label in labels:
            if "direct" in label or "fact" in label:
                return label
        return labels[0] if labels else "direct"

    def _solve(self, context: dict[str, Any]) -> str:
        if "required_phrase" in context:
            phrase = str(context["required_phrase"])
            base = str(context.get("task_input", context.get("text", ""))).strip()
            return f"{phrase} {base}".strip()
        if "a" in context and "b" in context:
            return str(self._apply_operation(context))
        text = _extract_primary_text(context)
        if "capital of france" in text.lower():
            return "Paris"
        if "capital of japan" in text.lower():
            return "Tokyo"
        arithmetic = _solve_arithmetic_from_text(text)
        if arithmetic is not None:
            return arithmetic
        choices = context.get("choices") or context.get("options")
        if isinstance(choices, list) and choices:
            lowered = text.lower()
            for choice in choices:
                if isinstance(choice, dict):
                    label = str(choice.get("label", ""))
                    value = str(choice.get("text", ""))
                    if value.lower() in lowered:
                        return label or value
            return str(choices[0].get("label", "A")) if isinstance(choices[0], dict) else str(choices[0])
        return text.strip() if text else "unknown"

    def _verify_answer(self, context: dict[str, Any]) -> str:
        candidate = str(context.get("candidate_answer", "")).strip().lower()
        expected = self._solve(context).strip().lower()
        return "PASS" if candidate == expected else "FAIL"

    def _apply_operation(self, context: dict[str, Any]) -> int | float:
        a = float(context["a"])
        b = float(context["b"])
        op = context.get("op", "+")
        if op == "+":
            value = a + b
        elif op == "-":
            value = a - b
        elif op == "*":
            value = a * b
        elif op == "/":
            value = a / b
        else:
            value = a + b
        return int(value) if value.is_integer() else value

    def _coerce_to_schema(self, raw_output: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        properties = schema.get("properties", {})
        if not properties:
            return {"text": raw_output}
        return {key: raw_output for key in properties}


def _extract_context_json(text: str) -> dict[str, Any]:
    match = re.search(r"Context JSON:\n(.*?)\n\nOutput JSON Schema:", text, re.S)
    if not match:
        return {}
    payload = match.group(1).strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {}


def _extract_primary_text(context: dict[str, Any]) -> str:
    for key in ["question", "query", "prompt", "text", "input"]:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value
    task_input = context.get("task_input")
    if isinstance(task_input, dict):
        return _extract_primary_text(task_input)
    if isinstance(task_input, str):
        return task_input
    return json.dumps(context, sort_keys=True)


def _solve_arithmetic_from_text(text: str) -> str | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(numbers) < 2:
        return None
    a = float(numbers[0])
    b = float(numbers[1])
    if "+" in text or "plus" in text.lower():
        result = a + b
    elif "-" in text or "minus" in text.lower():
        result = a - b
    elif "*" in text or "times" in text.lower():
        result = a * b
    elif "/" in text or "divided by" in text.lower():
        result = a / b
    else:
        return None
    return str(int(result) if result.is_integer() else result)

