from __future__ import annotations

from pathlib import Path

from topoprompt.eval.datasets import _example_from_payload, load_benchmark_examples


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
            {"input": f"{config} prompt 1", "target": f"{config} answer 1"},
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
    assert examples[0].input["prompt"] == "task_a prompt 1"
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
