from __future__ import annotations

from topoprompt.compiler.objective import compute_variance_adaptive_epsilon
from topoprompt.config import ObjectiveConfig
from topoprompt.schemas import CandidateEvaluation


def choose_smallest_effective(
    candidates: list[CandidateEvaluation],
    *,
    objective_config: ObjectiveConfig,
) -> tuple[CandidateEvaluation, CandidateEvaluation, float, list[CandidateEvaluation]]:
    if not candidates:
        raise ValueError("No candidates available for selection.")
    best = max(candidates, key=lambda item: item.score)
    epsilon = compute_variance_adaptive_epsilon(candidates, objective_config)
    effective = [candidate for candidate in candidates if candidate.score >= best.score - epsilon]
    smallest = min(effective, key=lambda item: (item.complexity, item.mean_invocations, item.program.program_id))
    return best, smallest, epsilon, effective

