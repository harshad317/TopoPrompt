from __future__ import annotations

from pathlib import Path

import pytest

from topoprompt.config import DataConfig
from topoprompt.eval.datasets import _example_from_payload, load_benchmark_examples, partition_examples
from topoprompt.schemas import Example


def test_example_from_payload_normalizes_scalar_input_and_choice_targets():
    example = _example_from_payload(
        {
            "input": "What is 2 + 2?",
            "choices": ["3", "4", "5"],
            "answer": 1,
            "subject": "arithmetic",
        },
        fallback_id="example_1",
    )

    assert example.input["prompt"] == "What is 2 + 2?"
    assert example.input["choices"][1]["label"] == "B"
    assert example.target == "B"
    assert example.metadata["subject"] == "arithmetic"


def test_load_benchmark_examples_aggregates_bbh_configs(monkeypatch):
    def fake_get_configs(_: str) -> list[str]:
        return ["task_a", "task_b"]

    def fake_load_dataset(name: str, config: str | None = None, *, split: str):
        assert name == "lukaemon/bbh"
        assert split == "test"
        return [
            {
                "input": f"{config} prompt 1\nOptions:\n(A) Alpha\n(B) Beta",
                "target": "(B)",
            },
            {"input": f"{config} prompt 2", "target": f"{config} answer 2"},
        ]

    monkeypatch.setattr("topoprompt.eval.datasets.get_dataset_config_names", fake_get_configs)
    monkeypatch.setattr("topoprompt.eval.datasets.load_dataset", fake_load_dataset)

    examples = load_benchmark_examples("bbh", split="test[:3]")

    assert [example.example_id for example in examples] == [
        "bbh_task_a_0",
        "bbh_task_a_1",
        "bbh_task_b_0",
    ]
    assert examples[0].input["prompt"].startswith("task_a prompt 1")
    assert examples[0].input["choices"][1]["label"] == "B"
    assert examples[0].input["choices"][1]["text"] == "Beta"
    assert examples[2].metadata["bbh_task"] == "task_b"


def test_load_benchmark_examples_reads_ifeval_jsonl(monkeypatch, tmp_path: Path):
    source = tmp_path / "ifeval_input_data.jsonl"
    source.write_text(
        "\n".join(
            [
                '{"key": 1, "prompt": "First prompt", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}',
                '{"key": 2, "prompt": "Second prompt", "instruction_id_list": ["detectable_format:title"], "kwargs": [{}]}',
            ]
        )
    )

    monkeypatch.setattr("topoprompt.eval.datasets.hf_hub_download", lambda *_, **__: str(source))

    examples = load_benchmark_examples("ifeval", split="train[:1]")

    assert len(examples) == 1
    assert examples[0].example_id == "ifeval_1"
    assert examples[0].input["prompt"] == "First prompt"
    assert examples[0].metadata["instruction_id_list"] == ["punctuation:no_comma"]


def test_load_benchmark_examples_uses_jsonl_path_when_name_is_not_builtin(tmp_path: Path):
    source = tmp_path / "custom_examples.jsonl"
    source.write_text('{"example_id":"custom_1","question":"What is 2 + 2?","target":"4"}\n')

    examples = load_benchmark_examples(str(source))

    assert len(examples) == 1
    assert examples[0].example_id == "custom_1"
    assert examples[0].input["question"] == "What is 2 + 2?"


def test_load_benchmark_examples_unknown_name_mentions_jsonl_path():
    with pytest.raises(ValueError, match="JSONL path"):
        load_benchmark_examples("custom_benchmark_name")


def test_partition_examples_stratifies_bbh_groups():
    examples = [
        Example(example_id=f"a_{index}", input={"prompt": f"A {index}"}, target="True", metadata={"bbh_task": "task_a"})
        for index in range(10)
    ] + [
        Example(example_id=f"b_{index}", input={"prompt": f"B {index}"}, target="False", metadata={"bbh_task": "task_b"})
        for index in range(10)
    ]

    partitions = partition_examples(
        examples,
        data_config=DataConfig(),
        create_test_split=True,
    )

    assert {example.metadata["bbh_task"] for example in partitions.validation_examples} == {"task_a", "task_b"}
    assert {example.metadata["bbh_task"] for example in partitions.test_examples} == {"task_a", "task_b"}


def test_partition_examples_stratifies_ifeval_by_primary_instruction():
    examples = [
        Example(
            example_id=f"comma_{index}",
            input={"prompt": f"Prompt {index}"},
            metadata={"instruction_id_list": ["punctuation:no_comma"], "instruction_kwargs": [{}]},
        )
        for index in range(10)
    ] + [
        Example(
            example_id=f"title_{index}",
            input={"prompt": f"Prompt {index}"},
            metadata={"instruction_id_list": ["detectable_format:title"], "instruction_kwargs": [{}]},
        )
        for index in range(10)
    ]

    partitions = partition_examples(
        examples,
        data_config=DataConfig(),
        create_test_split=True,
    )

    validation_primary = {example.metadata["instruction_id_list"][0] for example in partitions.validation_examples}
    test_primary = {example.metadata["instruction_id_list"][0] for example in partitions.test_examples}
    assert validation_primary == {"punctuation:no_comma", "detectable_format:title"}
    assert test_primary == {"punctuation:no_comma", "detectable_format:title"}
