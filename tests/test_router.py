from __future__ import annotations

from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.runtime.executor import ProgramExecutor
from topoprompt.schemas import Example, TaskAnalysis, TaskSpec


def test_route_program_chooses_reasoning_branch_for_arithmetic(fake_backend, small_config):
    task_spec = TaskSpec(task_id="task", description="Answer questions.")
    analysis = TaskAnalysis(input_heterogeneity="medium", initial_seed_templates=["route_direct_or_solve_finalize"])
    program = instantiate_seed_program(task_spec=task_spec, analysis=analysis, template_name="route_direct_or_solve_finalize")
    assert program is not None
    executor = ProgramExecutor(backend=fake_backend, config=small_config)
    example = Example(example_id="arith", input={"question": "What is 3 + 5?"}, target="8")
    result = executor.run_program(
        program=program,
        task_spec=task_spec,
        example_id=example.example_id,
        task_input=example.input,
        phase="confirmation",
    )
    route_trace = next(trace for trace in result.trace.node_traces if trace.node_id == "route_1")
    assert route_trace.route_choice == "solve"

