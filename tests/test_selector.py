from __future__ import annotations

from topoprompt.config import ObjectiveConfig
from topoprompt.compiler.selector import choose_smallest_effective
from topoprompt.schemas import CandidateEvaluation, PromptProgram


def test_selector_picks_smallest_effective_candidate():
    small_program = PromptProgram(
        program_id="small",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    large_program = PromptProgram(
        program_id="large",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    best = CandidateEvaluation(
        program=large_program,
        topology_fingerprint="b",
        family_signature="large",
        stage="confirmation",
        example_scores=[1, 1, 1, 0],
        score=0.75,
        search_score=0.7,
        mean_invocations=2,
        mean_tokens=10,
        complexity=0.7,
    )
    smaller = CandidateEvaluation(
        program=small_program,
        topology_fingerprint="a",
        family_signature="small",
        stage="confirmation",
        example_scores=[1, 1, 0, 1],
        score=0.75,
        search_score=0.69,
        mean_invocations=1,
        mean_tokens=8,
        complexity=0.2,
    )
    _, smallest, epsilon, effective = choose_smallest_effective(
        [best, smaller],
        objective_config=ObjectiveConfig(epsilon_floor=0.01, epsilon_z=1.0),
    )
    assert smallest.program.program_id == "small"
    assert epsilon >= 0.01
    assert len(effective) == 2
