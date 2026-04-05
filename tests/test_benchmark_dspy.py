from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from topoprompt.eval.benchmark_runner import BenchmarkRunner
from topoprompt.schemas import CompileArtifact, CompileMetrics, PromptProgram, TaskSpec


def test_benchmark_runner_compile_and_compare_with_dspy_smoke(monkeypatch, fake_backend, small_config, tmp_path: Path):
    rows = [
        {"example_id": "sst2_1", "input": {"sentence": "A joyful watch."}, "target": "positive"},
        {"example_id": "sst2_2", "input": {"sentence": "A dull mess."}, "target": "negative"},
        {"example_id": "sst2_3", "input": {"sentence": "Smart and engaging."}, "target": "positive"},
        {"example_id": "sst2_4", "input": {"sentence": "Painfully boring."}, "target": "negative"},
        {"example_id": "sst2_5", "input": {"sentence": "A winning performance."}, "target": "positive"},
        {"example_id": "sst2_6", "input": {"sentence": "An empty exercise."}, "target": "negative"},
    ]
    examples_path = tmp_path / "sst2_examples.jsonl"
    examples_path.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")

    compiled_optimizers: list[str] = []

    def fake_compile_task(**kwargs):
        assert kwargs["metric"] == "sst2"
        program = PromptProgram(
            program_id="compiled_prog",
            task_id="sst2_compiled",
            nodes=[],
            edges=[],
            entry_node_id="entry",
            finalize_node_id="finalize",
            metadata={"seed_template": "direct_finalize"},
        )
        task_spec = TaskSpec(
            task_id="sst2_compiled",
            description=kwargs["task_description"],
            input_schema={"sentence": "str"},
            output_schema={"type": "string"},
            task_family="classification",
        )
        metrics = CompileMetrics(
            best_program_id="compiled_prog",
            best_validation_score=0.75,
            smallest_effective_program_id="compiled_prog",
            smallest_effective_score=0.75,
            final_program_id="compiled_prog",
            final_program_score=0.75,
            final_program_policy="best_candidate",
            epsilon=0.01,
            planned_budget_calls=120,
            spent_budget_calls=64,
            winning_topology_family="direct-finalize",
        )
        return CompileArtifact(
            task_spec=task_spec,
            best_program_ir=program,
            smallest_effective_program_ir=program,
            program_ir=program,
            python_program=program,
            metrics=metrics,
            config={},
            output_dir=str(kwargs["output_dir"]) if kwargs.get("output_dir") is not None else None,
        )

    def fake_compile_dspy_baseline(**kwargs):
        optimizer_name = kwargs["optimizer_name"]
        compiled_optimizers.append(optimizer_name)
        program_id = f"dspy_{optimizer_name}_predict"
        return {
            "program": SimpleNamespace(_topoprompt_program_id=program_id),
            "summary": {
                "program_id": program_id,
                "student_strategy": "predict",
                "compile_seconds": 1.25 if optimizer_name == "mipro" else 1.5,
            },
        }

    def fake_compare_topoprompt_vs_dspy(**kwargs):
        dspy_program_id = getattr(kwargs["dspy_program"], "_topoprompt_program_id")
        dspy_score = 0.66 if "mipro" in dspy_program_id else 0.62
        return {
            "label_a": "topoprompt",
            "label_b": dspy_program_id,
            "program_a_id": "compiled_prog",
            "program_b_id": dspy_program_id,
            "sample_count": len(kwargs["examples"]),
            "repeats": 1,
            "score_a_mean": 0.78,
            "score_b_mean": dspy_score,
            "score_delta_a_minus_b_mean": 0.78 - dspy_score,
            "significance": {
                "repeat_results": [
                    {"mcnemar_exact_p_value": 0.5},
                ]
            },
        }

    def fake_compare_dspy_programs(**kwargs):
        return {
            "label_a": kwargs["label_a"],
            "label_b": kwargs["label_b"],
            "program_a_id": getattr(kwargs["program_a"], "_topoprompt_program_id"),
            "program_b_id": getattr(kwargs["program_b"], "_topoprompt_program_id"),
            "sample_count": len(kwargs["examples"]),
            "repeats": 1,
            "score_a_mean": 0.66,
            "score_b_mean": 0.62,
            "score_delta_a_minus_b_mean": 0.04,
            "significance": {
                "repeat_results": [
                    {"mcnemar_exact_p_value": 0.75},
                ]
            },
        }

    monkeypatch.setattr("topoprompt.eval.benchmark_runner.compile_task", fake_compile_task)
    monkeypatch.setattr("topoprompt.eval.benchmark_runner.compile_dspy_baseline", fake_compile_dspy_baseline)
    monkeypatch.setattr("topoprompt.eval.benchmark_runner.compare_topoprompt_vs_dspy", fake_compare_topoprompt_vs_dspy)
    monkeypatch.setattr("topoprompt.eval.benchmark_runner.compare_dspy_programs", fake_compare_dspy_programs)

    runner = BenchmarkRunner(config=small_config, backend=fake_backend)
    summary = runner.compile_and_compare_with_dspy(
        benchmark_name="sst2",
        examples_path=examples_path,
        output_dir=tmp_path / "sst2_dspy_run",
        optimizers="topoprompt,mipro,gepa",
    )

    assert summary["benchmark_name"] == "sst2"
    assert summary["task_family"] == "classification"
    assert summary["topoprompt"]["program_id"] == "compiled_prog"
    assert compiled_optimizers == ["mipro", "gepa"]
    assert set(summary["comparisons"]) == {"mipro", "gepa"}
    assert set(summary["pairwise_comparisons"]) == {
        "topoprompt_vs_mipro",
        "topoprompt_vs_gepa",
        "mipro_vs_gepa",
    }
    assert summary["comparisons"]["mipro"]["topoprompt_score"] == 0.78
    assert summary["comparisons"]["gepa"]["dspy_score"] == 0.62
    assert summary["pairwise_comparisons"]["mipro_vs_gepa"]["delta_a_minus_b"] == 0.04
    assert (tmp_path / "sst2_dspy_run" / "benchmark_dspy_summary.json").exists()
    assert (tmp_path / "sst2_dspy_run" / "benchmark_dspy_summary.md").exists()


def test_benchmark_runner_compile_and_compare_with_dspy_fails_fast_without_dspy(
    monkeypatch,
    fake_backend,
    small_config,
):
    compile_called = False

    def fail_without_dspy():
        raise RuntimeError("DSPy baselines require the optional `dspy` extra. Run `uv sync --extra dspy`.")

    def fake_compile_task(**kwargs):
        nonlocal compile_called
        compile_called = True
        raise AssertionError("compile_task should not run before the DSPy dependency check")

    monkeypatch.setattr("topoprompt.eval.benchmark_runner._require_dspy", fail_without_dspy)
    monkeypatch.setattr("topoprompt.eval.benchmark_runner.compile_task", fake_compile_task)

    runner = BenchmarkRunner(config=small_config, backend=fake_backend)
    with pytest.raises(RuntimeError, match="optional `dspy` extra"):
        runner.compile_and_compare_with_dspy(
            benchmark_name="gsm8k",
            optimizers="topoprompt,mipro,gepa",
            output_dir=None,
        )

    assert compile_called is False
