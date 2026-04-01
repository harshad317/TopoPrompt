from __future__ import annotations

from pathlib import Path

from topoprompt.eval.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_smoke_all_four(fake_backend, small_config, tmp_path):
    runner = BenchmarkRunner(config=small_config, backend=fake_backend)
    fixtures = Path(__file__).resolve().parent / "fixtures" / "smoke"
    for benchmark in ["gsm8k", "mmlu", "bbh", "ifeval"]:
        artifact = runner.compile_benchmark(
            benchmark_name=benchmark,
            examples_path=fixtures / f"{benchmark}_examples.jsonl",
            task_description=(fixtures / f"{benchmark}_task.md").read_text(),
            output_dir=tmp_path / benchmark,
        )
        assert artifact.program_ir.program_id

