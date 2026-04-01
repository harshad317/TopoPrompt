from __future__ import annotations

from pathlib import Path

from topoprompt.compiler.search import compile_task, evaluate_program_on_examples
from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.eval.metrics import numeric_metric
from topoprompt.schemas import Example, TaskAnalysis


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


def test_evaluate_program_runs_full_dataset_without_compile_budget_cap(fake_backend, small_config, simple_task_spec):
    analysis = TaskAnalysis(needs_reasoning=False, initial_seed_templates=["direct_finalize"])
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    examples = [
        Example(
            example_id=f"eval_{index}",
            input={"question": f"What is {index} + 1?"},
            target=str(index + 1),
        )
        for index in range(25)
    ]

    result = evaluate_program_on_examples(
        program=program,
        task_spec=simple_task_spec,
        examples=examples,
        metric_fn=numeric_metric,
        backend=fake_backend,
        config=small_config,
        phase="confirmation",
    )

    assert len(result["traces"]) == len(examples)
