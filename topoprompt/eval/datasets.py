from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import orjson
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download

from topoprompt.config import DataConfig
from topoprompt.schemas import Example


@dataclass
class DatasetPartitions:
    compile_examples: list[Example]
    fewshot_examples: list[Example]
    search_examples: list[Example]
    validation_examples: list[Example]
    test_examples: list[Example]


def load_examples_from_jsonl(path: str | Path) -> list[Example]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        payload = orjson.loads(line)
        rows.append(_example_from_payload(payload, fallback_id=f"example_{line_number}"))
    return rows


def load_benchmark_examples(name: str, *, path: str | Path | None = None, split: str | None = None) -> list[Example]:
    if path is not None:
        return load_examples_from_jsonl(path)
    normalized = name.lower()
    if normalized == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split=split or "train")
        return [_example_from_payload(row, fallback_id=f"{name}_{index}") for index, row in enumerate(dataset)]
    elif normalized == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split=split or "validation")
        return [_example_from_payload(row, fallback_id=f"{name}_{index}") for index, row in enumerate(dataset)]
    elif normalized == "bbh":
        return _load_bbh_examples(split or "test")
    elif normalized == "ifeval":
        return _load_ifeval_examples(split or "train")
    else:
        raise ValueError(f"Unsupported benchmark: {name}")


def partition_examples(
    examples: list[Example],
    *,
    data_config: DataConfig,
    create_test_split: bool = False,
) -> DatasetPartitions:
    total = len(examples)
    if total == 0:
        return DatasetPartitions([], [], [], [], [])

    if create_test_split:
        compile_count = max(1, int(total * data_config.compile_fraction_if_no_official_split))
        validation_count = max(1, int(total * data_config.validation_fraction_if_no_official_split))
        compile_examples = examples[:compile_count]
        validation_examples = examples[compile_count : compile_count + validation_count]
        test_examples = examples[compile_count + validation_count :]
    else:
        validation_count = max(1, int(total * data_config.validation_fraction_if_no_official_split))
        compile_examples = examples[:-validation_count] if total > validation_count else examples[: max(total - 1, 1)]
        validation_examples = examples[len(compile_examples) :]
        test_examples = []

    fewshot_count = min(max(1, int(len(compile_examples) * data_config.fewshot_pool_fraction_of_compile)), data_config.fewshot_pool_max_examples)
    fewshot_examples = compile_examples[:fewshot_count]
    search_examples = compile_examples[fewshot_count:]
    if not search_examples:
        search_examples = fewshot_examples[:]
        fewshot_examples = fewshot_examples[:1]

    return DatasetPartitions(
        compile_examples=compile_examples,
        fewshot_examples=fewshot_examples,
        search_examples=search_examples,
        validation_examples=validation_examples or compile_examples[-1:],
        test_examples=test_examples,
    )


def _example_from_payload(payload: dict[str, Any], *, fallback_id: str) -> Example:
    normalized = dict(payload)
    metadata = dict(normalized.pop("metadata", {}) or {})
    target = normalized.pop("target", None)
    if target is None:
        target = normalized.pop("answer", None) or normalized.pop("label", None)
    if "input" in normalized:
        raw_input = normalized.pop("input")
        if isinstance(raw_input, dict):
            input_payload = dict(raw_input)
        else:
            input_payload = {"prompt": raw_input}
    elif "question" in normalized:
        input_payload = {"question": normalized.pop("question")}
    elif "prompt" in normalized:
        input_payload = {"prompt": normalized.pop("prompt")}
    else:
        input_payload = {key: value for key, value in normalized.items() if key not in {"id", "example_id"}}
    if "choices" in normalized:
        input_payload["choices"] = _normalize_choices(normalized.pop("choices"))
    if "required_phrase" in normalized:
        input_payload["required_phrase"] = normalized.pop("required_phrase")
    if isinstance(target, int) and "choices" in input_payload:
        choices = input_payload["choices"]
        if isinstance(choices, list) and 0 <= target < len(choices):
            choice = choices[target]
            if isinstance(choice, dict):
                target = choice.get("label", chr(ord("A") + target))
            else:
                target = chr(ord("A") + target)
    for key, value in normalized.items():
        if key not in {"id", "example_id"}:
            metadata.setdefault(key, value)
    return Example(
        example_id=str(payload.get("example_id") or payload.get("id") or fallback_id),
        input=input_payload,
        target=target,
        metadata=metadata,
    )


def _load_bbh_examples(split: str) -> list[Example]:
    base_split, selection = _parse_split_spec(split, default_split="test")
    examples: list[Example] = []
    for config_name in get_dataset_config_names("lukaemon/bbh"):
        dataset = load_dataset("lukaemon/bbh", config_name, split=base_split)
        for index, row in enumerate(dataset):
            payload = dict(row)
            metadata = dict(payload.get("metadata", {}) or {})
            metadata["bbh_task"] = config_name
            payload["metadata"] = metadata
            examples.append(_example_from_payload(payload, fallback_id=f"bbh_{config_name}_{index}"))
            if selection.step is None and selection.stop is not None and len(examples) >= selection.stop:
                return _apply_selection(examples, selection)
    return _apply_selection(examples, selection)


def _load_ifeval_examples(split: str) -> list[Example]:
    _, selection = _parse_split_spec(split, default_split="train")
    path = hf_hub_download("google/IFEval", "ifeval_input_data.jsonl", repo_type="dataset")
    examples: list[Example] = []
    with Path(path).open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = orjson.loads(line)
            metadata = {
                "instruction_id_list": payload.get("instruction_id_list", []),
                "instruction_kwargs": payload.get("kwargs", []),
            }
            examples.append(
                Example(
                    example_id=f"ifeval_{payload.get('key', line_number)}",
                    input={"prompt": payload.get("prompt", "")},
                    target=None,
                    metadata=metadata,
                )
            )
            if selection.step is None and selection.stop is not None and len(examples) >= selection.stop:
                return _apply_selection(examples, selection)
    return _apply_selection(examples, selection)


def _parse_split_spec(split: str | None, *, default_split: str) -> tuple[str, slice]:
    value = (split or default_split).strip()
    match = re.match(r"^(?P<name>[^\[]+)(?:\[(?P<body>[^\]]*)\])?$", value)
    if not match:
        raise ValueError(f"Unsupported split syntax: {split}")
    split_name = match.group("name").strip() or default_split
    body = match.group("body")
    if body is None or body == "":
        return split_name, slice(None)
    if ":" not in body:
        raise ValueError(f"Unsupported split slice: {split}")
    start_text, stop_text = body.split(":", 1)
    start = int(start_text) if start_text else None
    stop = int(stop_text) if stop_text else None
    return split_name, slice(start, stop)


def _apply_selection(examples: list[Example], selection: slice) -> list[Example]:
    if selection.start is None and selection.stop is None and selection.step is None:
        return examples
    return examples[selection]


def _normalize_choices(choices: Any) -> Any:
    if not isinstance(choices, list):
        return choices
    if choices and all(isinstance(choice, str) for choice in choices):
        return [
            {"label": chr(ord("A") + index), "text": str(choice)}
            for index, choice in enumerate(choices)
        ]
    return choices
