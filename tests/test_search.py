from __future__ import annotations

from pathlib import Path

from topoprompt.compiler.search import compile_task


def test_compile_task_runs_end_to_end(fake_backend, small_config, gsm8k_examples, tmp_path):
    artifact = compile_task(
        task_description="Answer simple arithmetic questions accurately.",
        examples=gsm8k_examples,
        metric="gsm8k",
        backend=fake_backend,
        config=small_config,
        output_dir=tmp_path / "run",
        task_id="gsm8k_smoke",
    )
    assert artifact.program_ir.program_id
    assert artifact.metrics.best_validation_score >= 0.0
    assert (tmp_path / "run" / "final_program.json").exists()
    assert artifact.candidate_archive

