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
    elif normalized == "sst2":
        return _load_sst2_examples(split or "train")
    elif normalized == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split=split or "validation")
        return [_example_from_payload(row, fallback_id=f"{name}_{index}") for index, row in enumerate(dataset)]
    elif normalized == "bbh":
        return _load_bbh_examples(split or "test")
    elif normalized == "ifeval":
        return _load_ifeval_examples(split or "train")
    else:
        candidate_path = Path(name).expanduser()
        if candidate_path.is_file():
            return load_examples_from_jsonl(candidate_path)
        raise ValueError(
            "Unsupported benchmark: "
            f"{name}. Use one of gsm8k, sst2, mmlu, bbh, or ifeval, or pass a JSONL path via `name` or `path`."
        )


def partition_examples(
    examples: list[Example],
    *,
    data_config: DataConfig,
    create_test_split: bool = False,
) -> DatasetPartitions:
    total = len(examples)
    if total == 0:
        return DatasetPartitions([], [], [], [], [])

    stratify_key = _infer_partition_stratify_key(examples)
    if stratify_key is not None:
        compile_examples, validation_examples, test_examples = _partition_examples_stratified(
            examples,
            stratify_key=stratify_key,
            data_config=data_config,
            create_test_split=create_test_split,
        )
    elif create_test_split:
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
            prompt = payload.get("input") or payload.get("prompt")
            if isinstance(prompt, str):
                choices = _extract_bbh_choices(prompt)
                if choices:
                    payload["choices"] = choices
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


def _load_sst2_examples(split: str) -> list[Example]:
    dataset = load_dataset("stanfordnlp/sst2", split=split)
    examples: list[Example] = []
    for index, row in enumerate(dataset):
        label = row.get("label")
        target = None
        if label is not None and int(label) >= 0:
            target = "positive" if int(label) == 1 else "negative"
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"idx", "sentence", "label"}
        }
        examples.append(
            Example(
                example_id=str(row.get("idx", f"sst2_{index}")),
                input={"sentence": str(row.get("sentence", ""))},
                target=target,
                metadata=metadata,
            )
        )
    return examples


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


def _extract_bbh_choices(prompt: str) -> list[dict[str, str]] | None:
    choices: list[dict[str, str]] = []
    saw_options_header = False
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not saw_options_header:
            if line.lower().startswith("options:"):
                saw_options_header = True
            continue
        match = re.match(r"^\(([A-Z])\)\s*(.+)$", line)
        if match:
            choices.append({"label": match.group(1), "text": match.group(2).strip()})
            continue
        if choices and line:
            choices[-1]["text"] = f"{choices[-1]['text']} {line}".strip()
    return choices or None


def _infer_partition_stratify_key(examples: list[Example]) -> tuple[str, ...] | None:
    if all(example.metadata.get("bbh_task") for example in examples):
        return ("bbh_task",)
    if all(example.metadata.get("instruction_id_list") for example in examples):
        return ("instruction_id_list",)
    return None


def _partition_examples_stratified(
    examples: list[Example],
    *,
    stratify_key: tuple[str, ...],
    data_config: DataConfig,
    create_test_split: bool,
) -> tuple[list[Example], list[Example], list[Example]]:
    grouped = _group_examples_by_stratify_key(examples, stratify_key=stratify_key)
    compile_groups: list[list[Example]] = []
    validation_groups: list[list[Example]] = []
    test_groups: list[list[Example]] = []

    for group_examples in grouped.values():
        compile_count, validation_count, test_count = _group_split_counts(
            len(group_examples),
            data_config=data_config,
            create_test_split=create_test_split,
        )
        compile_groups.append(group_examples[:compile_count])
        validation_groups.append(group_examples[compile_count : compile_count + validation_count])
        test_groups.append(group_examples[compile_count + validation_count : compile_count + validation_count + test_count])

    return (
        _interleave_example_groups(compile_groups),
        _interleave_example_groups(validation_groups),
        _interleave_example_groups(test_groups),
    )


def _group_examples_by_stratify_key(
    examples: list[Example],
    *,
    stratify_key: tuple[str, ...],
) -> dict[str, list[Example]]:
    groups: dict[str, list[Example]] = {}
    for example in examples:
        if stratify_key == ("bbh_task",):
            key = str(example.metadata.get("bbh_task"))
        elif stratify_key == ("instruction_id_list",):
            instruction_ids = example.metadata.get("instruction_id_list") or []
            key = str(instruction_ids[0]) if instruction_ids else "__missing__"
        else:
            key = "__default__"
        groups.setdefault(key, []).append(example)
    return groups


def _group_split_counts(
    total: int,
    *,
    data_config: DataConfig,
    create_test_split: bool,
) -> tuple[int, int, int]:
    if total <= 1:
        return total, 0, 0
    if create_test_split:
        compile_count = max(1, int(total * data_config.compile_fraction_if_no_official_split))
        validation_count = max(1, int(total * data_config.validation_fraction_if_no_official_split)) if total >= 3 else max(0, total - compile_count)
        if compile_count + validation_count >= total:
            if total >= 3:
                while compile_count + validation_count >= total and validation_count > 1:
                    validation_count -= 1
                while compile_count + validation_count >= total and compile_count > 1:
                    compile_count -= 1
            else:
                validation_count = total - compile_count
        test_count = max(0, total - compile_count - validation_count)
        return compile_count, validation_count, test_count

    validation_count = max(1, int(total * data_config.validation_fraction_if_no_official_split))
    if validation_count >= total:
        validation_count = 1
    compile_count = max(1, total - validation_count)
    validation_count = total - compile_count
    return compile_count, validation_count, 0


def _interleave_example_groups(groups: list[list[Example]]) -> list[Example]:
    non_empty_groups = [group for group in groups if group]
    if not non_empty_groups:
        return []
    result: list[Example] = []
    max_group_size = max(len(group) for group in non_empty_groups)
    for index in range(max_group_size):
        for group in non_empty_groups:
            if index < len(group):
                result.append(group[index])
    return result
