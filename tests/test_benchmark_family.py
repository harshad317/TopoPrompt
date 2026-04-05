from __future__ import annotations

from pathlib import Path

import orjson

from topoprompt.eval.benchmark_runner import BenchmarkRunner, bbh_family_for_task, group_benchmark_examples
from topoprompt.schemas import Example


def test_group_benchmark_examples_maps_bbh_tasks_to_families():
    examples = [
        Example(example_id="arith_1", input={"question": "2 + 2?"}, target="A", metadata={"bbh_task": "multistep_arithmetic_two"}),
        Example(example_id="logic_1", input={"question": "Is the statement valid?"}, target="A", metadata={"bbh_task": "formal_fallacies"}),
        Example(example_id="logic_2", input={"question": "Who is lying?"}, target="B", metadata={"bbh_task": "web_of_lies"}),
    ]

    grouped = group_benchmark_examples(benchmark_name="bbh", examples=examples, grouping="family")

    assert set(grouped) == {"arithmetic_counting", "logical_deduction"}
    assert bbh_family_for_task("tracking_shuffled_objects_three_objects") == "state_tracking"


def test_benchmark_runner_compile_and_compare_by_family_smoke(fake_backend, small_config, tmp_path: Path):
    runner = BenchmarkRunner(config=small_config, backend=fake_backend)
    examples_path = tmp_path / "bbh_family_examples.jsonl"
    rows = [
        {
            "example_id": "arith_1",
            "question": "Which option gives the result of 2 + 3?",
            "choices": [
                {"label": "A", "text": "5"},
                {"label": "B", "text": "6"},
                {"label": "C", "text": "7"},
            ],
            "target": "A",
            "metadata": {"bbh_task": "multistep_arithmetic_two"},
        },
        {
            "example_id": "arith_2",
            "question": "Which option gives the result of 6 - 1?",
            "choices": [
                {"label": "A", "text": "5"},
                {"label": "B", "text": "4"},
                {"label": "C", "text": "6"},
            ],
            "target": "A",
            "metadata": {"bbh_task": "multistep_arithmetic_two"},
        },
        {
            "example_id": "arith_3",
            "question": "Which option gives the result of 3 * 2?",
            "choices": [
                {"label": "A", "text": "6"},
                {"label": "B", "text": "5"},
                {"label": "C", "text": "7"},
            ],
            "target": "A",
            "metadata": {"bbh_task": "multistep_arithmetic_two"},
        },
        {
            "example_id": "arith_4",
            "question": "Which option gives the result of 8 - 3?",
            "choices": [
                {"label": "A", "text": "5"},
                {"label": "B", "text": "6"},
                {"label": "C", "text": "4"},
            ],
            "target": "A",
            "metadata": {"bbh_task": "multistep_arithmetic_two"},
        },
        {
            "example_id": "logic_1",
            "question": "Choose the option that states the conclusion follows logically.",
            "choices": [
                {"label": "A", "text": "valid"},
                {"label": "B", "text": "invalid"},
            ],
            "target": "A",
            "metadata": {"bbh_task": "formal_fallacies"},
        },
        {
            "example_id": "logic_2",
            "question": "Choose the option that states the conclusion follows logically.",
            "choices": [
                {"label": "A", "text": "valid"},
                {"label": "B", "text": "invalid"},
            ],
            "target": "A",
            "metadata": {"bbh_task": "formal_fallacies"},
        },
        {
            "example_id": "logic_3",
            "question": "Choose the option that states the conclusion follows logically.",
            "choices": [
                {"label": "A", "text": "valid"},
                {"label": "B", "text": "invalid"},
            ],
            "target": "A",
            "metadata": {"bbh_task": "formal_fallacies"},
        },
        {
            "example_id": "logic_4",
            "question": "Choose the option that states the conclusion follows logically.",
            "choices": [
                {"label": "A", "text": "valid"},
                {"label": "B", "text": "invalid"},
            ],
            "target": "A",
            "metadata": {"bbh_task": "formal_fallacies"},
        },
    ]
    examples_path.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")

    summary = runner.compile_and_compare_by_family(
        benchmark_name="bbh",
        examples_path=examples_path,
        task_description="Solve diverse reasoning tasks with the correct final answer.",
        output_dir=tmp_path / "bbh_family_run",
        grouping="family",
    )

    assert summary["group_count"] == 2
    assert set(summary["families"]) == {"arithmetic_counting", "logical_deduction"}
    assert summary["total_eval_examples"] > 0
    assert (tmp_path / "bbh_family_run" / "family_summary.json").exists()
    assert (tmp_path / "bbh_family_run" / "arithmetic_counting" / "compare_compiled_vs_direct" / "compare_summary.json").exists()
