from __future__ import annotations

from topoprompt.backends.llm_client import FakeBackend
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


def test_runtime_executes_direct_self_consistency_majority_vote(small_config, simple_task_spec):
    responses = iter(["11", "12", "12"])

    def structured_handler(_system_prompt: str, _user_prompt: str, _schema: dict[str, object]) -> dict[str, str]:
        return {"candidate_answer": next(responses)}

    backend = FakeBackend(structured_handler=structured_handler)
    analysis = TaskAnalysis(
        needs_reasoning=False,
        initial_seed_templates=["direct_self_consistency_x3"],
    )
    program = instantiate_seed_program(
        task_spec=simple_task_spec,
        analysis=analysis,
        template_name="direct_self_consistency_x3",
    )
    assert program is not None

    executor = ProgramExecutor(backend=backend, config=small_config)
    result = executor.run_program(
        program=program,
        task_spec=simple_task_spec,
        example_id="consensus_case",
        task_input={"question": "What is 6 + 6?"},
        phase="confirmation",
    )

    assert result.trace.final_output == "12"
    assert result.trace.total_invocations == 3
    assert [trace.node_id for trace in result.trace.node_traces] == [
        "direct_1",
        "direct_2",
        "direct_3",
        "finalize_1",
    ]
    assert result.state["candidate_answer_1"] == "11"
    assert result.state["candidate_answer_2"] == "12"
    assert result.state["candidate_answer_3"] == "12"


def test_runtime_executes_format_finalize_program(fake_backend, small_config, simple_task_spec, gsm8k_examples):
    analysis = TaskAnalysis(
        output_format="json",
        initial_seed_templates=["format_finalize"],
    )
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="format_finalize")
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
    assert [trace.node_id for trace in result.trace.node_traces] == ["direct_1", "format_1", "finalize_1"]


def test_runtime_executes_critique_revise_program(fake_backend, small_config, simple_task_spec, gsm8k_examples):
    analysis = TaskAnalysis(
        task_family="generation",
        initial_seed_templates=["critique_revise_finalize"],
    )
    program = instantiate_seed_program(
        task_spec=simple_task_spec,
        analysis=analysis,
        template_name="critique_revise_finalize",
    )
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
    assert [trace.node_id for trace in result.trace.node_traces] == [
        "direct_1",
        "critique_1",
        "revise_1",
        "finalize_1",
    ]


def test_gsm8k_metric_uses_final_answer_marker():
    example = Example(
        example_id="gsm8k_metric",
        input={"question": "dummy"},
        target="Some reasoning here.\n#### 72",
    )
    assert numeric_metric("72", example) == 1.0
    assert numeric_metric("48", example) == 0.0
    assert numeric_metric("We compute 8 * 9 = 72, so the final answer is 72.", example) == 1.0
    assert numeric_metric("First we saw 8 and 9, but the final answer is 48.", example) == 0.0
