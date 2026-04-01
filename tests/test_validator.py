from __future__ import annotations

import pytest

from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.compiler.validator import ProgramValidationError, validate_program
from topoprompt.schemas import ProgramEdge, TaskAnalysis, TaskSpec


def test_validator_accepts_valid_seed(small_config):
    task_spec = TaskSpec(task_id="task", description="Answer questions.")
    analysis = TaskAnalysis(initial_seed_templates=["route_direct_or_solve_finalize"])
    program = instantiate_seed_program(task_spec=task_spec, analysis=analysis, template_name="route_direct_or_solve_finalize")
    assert program is not None
    validate_program(program, small_config.program)


def test_validator_rejects_route_label_mismatch(small_config):
    task_spec = TaskSpec(task_id="task", description="Answer questions.")
    analysis = TaskAnalysis(initial_seed_templates=["route_direct_or_solve_finalize"])
    program = instantiate_seed_program(task_spec=task_spec, analysis=analysis, template_name="route_direct_or_solve_finalize")
    assert program is not None
    program.edges = [ProgramEdge(source="route_1", target="direct_1", label="wrong"), *program.edges[1:]]
    with pytest.raises(ProgramValidationError):
        validate_program(program, small_config.program)

