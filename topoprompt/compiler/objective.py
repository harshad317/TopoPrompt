from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev, stdev

from topoprompt.compiler.task_priors import normalize_task_family
from topoprompt.config import ObjectiveConfig, ProgramConfig
from topoprompt.ir import branch_count, prompt_token_count
from topoprompt.schemas import CandidateEvaluation, PromptProgram


def description_length(program: PromptProgram, program_config: ProgramConfig) -> float:
    node_term = min(len(program.nodes) / max(program_config.max_nodes, 1), 1.0)
    max_edges = max(program_config.max_nodes * max(program_config.max_branch_fanout, 1), 1)
    edge_term = min(len(program.edges) / max_edges, 1.0)
    branch_term = min(branch_count(program) / max(program_config.max_route_nodes * program_config.max_branch_fanout, 1), 1.0)
    prompt_term = min(prompt_token_count(program) / max(program_config.prompt_token_cap, 1), 1.0)
    return 0.25 * (node_term + edge_term + branch_term + prompt_term)


def search_score(
    *,
    perf: float,
    mean_invocations: float,
    complexity: float,
    parse_failure_rate: float,
    coverage_ratio: float,
    objective_config: ObjectiveConfig,
    program_config: ProgramConfig,
    task_family: str | None = None,
    budget_examples: int | None = None,
) -> float:
    weights = _resolve_objective_weights(objective_config=objective_config, task_family=task_family)
    cost_norm = min(mean_invocations / max(program_config.max_nodes, 1), 1.0)
    # Normalize coverage against the stage budget cap rather than the raw
    # example-set size.  Without this, a candidate evaluated on 100 of 1000
    # examples receives a massive penalty compared to the same candidate
    # evaluated on 40 of 50 examples at screening — purely due to set sizes,
    # not actual coverage quality.
    if budget_examples is not None and budget_examples > 0:
        effective_coverage = min(coverage_ratio, 1.0)
    else:
        effective_coverage = coverage_ratio
    coverage_penalty = objective_config.delta_partial_coverage * max(1.0 - effective_coverage, 0.0)
    return (
        perf
        - weights["alpha_cost"] * cost_norm
        - weights["beta_complexity"] * complexity
        - objective_config.gamma_parse_failure * parse_failure_rate
        - coverage_penalty
    )


def _resolve_objective_weights(*, objective_config: ObjectiveConfig, task_family: str | None) -> dict[str, float]:
    normalized_family = normalize_task_family(task_family)
    overrides = objective_config.family_overrides.get(normalized_family, {})
    return {
        "alpha_cost": float(overrides.get("alpha_cost", objective_config.alpha_cost)),
        "beta_complexity": float(overrides.get("beta_complexity", objective_config.beta_complexity)),
    }


def compute_variance_adaptive_epsilon(candidates: list[CandidateEvaluation], objective_config: ObjectiveConfig) -> float:
    if not candidates:
        return objective_config.epsilon_floor
    best = max(candidates, key=lambda item: item.score)
    scores = best.example_scores
    if not scores:
        return objective_config.epsilon_floor
    if all(score in (0.0, 1.0) for score in scores):
        p = mean(scores)
        n = len(scores)
        se = sqrt(max(p * (1 - p), 0.0) / max(n, 1))
    else:
        # Use the unbiased sample standard deviation (n-1 denominator) to avoid
        # underestimating variance on small evaluation sets.
        se = (stdev(scores) if len(scores) > 1 else 0.0) / sqrt(max(len(scores), 1))
    return max(objective_config.epsilon_floor, objective_config.epsilon_z * se)
