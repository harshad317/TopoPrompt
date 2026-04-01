from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
from datasets import load_dataset

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
    elif normalized == "mmlu":
        dataset = load_dataset("cais/mmlu", "all", split=split or "validation")
    elif normalized == "bbh":
        dataset = load_dataset("lukaemon/bbh", split=split or "test")
    elif normalized == "ifeval":
        dataset = load_dataset("google/IFEval", split=split or "train")
    else:
        raise ValueError(f"Unsupported benchmark: {name}")
    return [_example_from_payload(row, fallback_id=f"{name}_{index}") for index, row in enumerate(dataset)]


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
    if "input" in payload:
        return Example.model_validate(
            {
                "example_id": payload.get("example_id", fallback_id),
                "input": payload["input"],
                "target": payload.get("target"),
                "metadata": payload.get("metadata", {}),
            }
        )
    normalized = dict(payload)
    target = normalized.pop("target", None)
    if target is None:
        target = normalized.pop("answer", None) or normalized.pop("label", None)
    metadata = normalized.pop("metadata", {})
    if "question" in normalized:
        input_payload = {"question": normalized.pop("question")}
    elif "prompt" in normalized:
        input_payload = {"prompt": normalized.pop("prompt")}
    else:
        input_payload = {key: value for key, value in normalized.items() if key not in {"id", "example_id"}}
    if "choices" in normalized:
        input_payload["choices"] = normalized["choices"]
    if "required_phrase" in normalized:
        input_payload["required_phrase"] = normalized["required_phrase"]
    return Example(
        example_id=str(payload.get("example_id") or payload.get("id") or fallback_id),
        input=input_payload,
        target=target,
        metadata=metadata,
    )

