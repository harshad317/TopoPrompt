from __future__ import annotations

from topoprompt.compiler.objective import description_length
from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.ir import family_signature, topology_fingerprint
from topoprompt.schemas import TaskAnalysis, TaskSpec


def test_topology_fingerprint_stable(small_config):
    task_spec = TaskSpec(task_id="task", description="Answer questions.")
    analysis = TaskAnalysis(initial_seed_templates=["direct_finalize"])
    program_a = instantiate_seed_program(task_spec=task_spec, analysis=analysis, template_name="direct_finalize")
    program_b = instantiate_seed_program(task_spec=task_spec, analysis=analysis, template_name="direct_finalize")
    assert program_a is not None and program_b is not None
    assert topology_fingerprint(program_a) == topology_fingerprint(program_b)
    assert family_signature(program_a) == family_signature(program_b)


def test_description_length_prefers_smaller_program(small_config):
    task_spec = TaskSpec(task_id="task", description="Answer questions.")
    analysis = TaskAnalysis(initial_seed_templates=["direct_finalize", "plan_solve_verify_finalize"])
    small = instantiate_seed_program(task_spec=task_spec, analysis=analysis, template_name="direct_finalize")
    large = instantiate_seed_program(task_spec=task_spec, analysis=analysis, template_name="plan_solve_verify_finalize")
    assert small is not None and large is not None
    assert description_length(small, small_config.program) < description_length(large, small_config.program)

