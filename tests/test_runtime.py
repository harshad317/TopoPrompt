from __future__ import annotations

from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.eval.metrics import numeric_metric
from topoprompt.runtime.executor import ProgramExecutor
from topoprompt.runtime.parser import parse_structured_output
from topoprompt.schemas import Example, TaskAnalysis


def test_runtime_executes_solve_verify_program(fake_backend, small_config, simple_task_spec, gsm8k_examples):
    analysis = TaskAnalysis(
        needs_reasoning=True,
        needs_verification=True,
        initial_seed_templates=["solve_verify_finalize"],
    )
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="solve_verify_finalize")
    assert program is not None
    executor = ProgramExecutor(backend=fake_backend, config=small_config)
    result = executor.run_program(
        program=program,
        task_spec=simple_task_spec,
        example_id=gsm8k_examples[0].example_id,
        task_input=gsm8k_examples[0].input,
        phase="confirmation",
    )
    assert result.trace.final_output == "4"
    assert any(node.node_id == "verify_1" for node in result.trace.node_traces)


def test_parser_falls_back_to_repair(fake_backend):
    schema = {
        "type": "object",
        "properties": {"candidate_answer": {"type": "string"}},
        "required": ["candidate_answer"],
    }
    parsed, repair_used = parse_structured_output(
        raw_output='candidate_answer: "42"',
        schema=schema,
        backend=fake_backend,
        repair_model="fake-model",
    )
    assert parsed["candidate_answer"] == "42"
    assert repair_used is False


def test_gsm8k_metric_uses_final_answer_marker():
    example = Example(
        example_id="gsm8k_metric",
        input={"question": "dummy"},
        target="Some reasoning here.\n#### 72",
    )
    assert numeric_metric("72", example) == 1.0
    assert numeric_metric("48", example) == 0.0
