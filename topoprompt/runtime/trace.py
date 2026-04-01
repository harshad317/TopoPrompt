from __future__ import annotations

from statistics import mean

from topoprompt.schemas import ProgramExecutionTrace


def aggregate_route_metrics(traces: list[ProgramExecutionTrace]) -> tuple[float | None, float | None]:
    diagnostics = [diag for trace in traces for diag in trace.route_diagnostics if diag.oracle_branch]
    if not diagnostics:
        return None, None
    accuracy = mean(1.0 if diag.chosen_branch == diag.oracle_branch else 0.0 for diag in diagnostics)
    regrets = [diag.regret for diag in diagnostics if diag.regret is not None]
    regret = mean(regrets) if regrets else None
    return accuracy, regret

