from __future__ import annotations

import json
import re
from typing import Any

from topoprompt.backends.llm_client import LLMBackend


class ParseFailed(RuntimeError):
    pass


def parse_structured_output(
    *,
    raw_output: str,
    schema: dict[str, Any],
    backend: LLMBackend | None,
    repair_model: str | None,
) -> tuple[dict[str, Any], bool]:
    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            return _coerce_to_schema(parsed, schema), False
    except json.JSONDecodeError:
        pass

    extracted = _extract_minimal_fields(raw_output, schema)
    if extracted is not None:
        return extracted, False

    if backend is not None and repair_model:
        repaired = backend.repair_json(raw_output=raw_output, schema=schema, model=repair_model)
        try:
            parsed = json.loads(repaired.text)
            if isinstance(parsed, dict):
                return _coerce_to_schema(parsed, schema), True
        except json.JSONDecodeError:
            pass
        if repaired.structured is not None:
            return _coerce_to_schema(repaired.structured, schema), True

    raise ParseFailed("Unable to parse output into the expected schema.")


def _coerce_to_schema(parsed: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties", {})
    if not properties:
        return parsed
    result: dict[str, Any] = {}
    for key in properties:
        result[key] = parsed.get(key)
    return result


def _extract_minimal_fields(raw_output: str, schema: dict[str, Any]) -> dict[str, Any] | None:
    properties = schema.get("properties", {})
    if not properties:
        return {"text": raw_output}
    extracted: dict[str, Any] = {}
    for key in properties:
        pattern = rf'"?{re.escape(key)}"?\s*[:=]\s*"?(.*?)"?(?:,|\n|$)'
        match = re.search(pattern, raw_output, re.I)
        if match:
            value = match.group(1).strip().strip('"')
            if key == "confidence":
                try:
                    extracted[key] = float(value)
                except ValueError:
                    extracted[key] = None
            else:
                extracted[key] = value
    if extracted:
        return {key: extracted.get(key) for key in properties}
    if len(properties) == 1:
        key = next(iter(properties))
        return {key: raw_output.strip()}
    return None

