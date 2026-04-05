from __future__ import annotations

import math
import random
from typing import Any

from topoprompt.schemas import CandidateEvaluation


class NoOpAcquisition:
    def choose(self, candidates: list[CandidateEvaluation], *, limit: int | None = None) -> list[CandidateEvaluation]:
        return candidates if limit is None else candidates[:limit]


class DiversityAcquisition:
    """Selects candidates balancing exploitation (high score) and exploration
    (diverse topology families not yet well-represented in the beam).

    Strategy:
        1. Always include the highest-scoring candidate (exploit the best).
        2. Fill remaining slots using an Upper Confidence Bound (UCB) score:

               ucb = mean_family_score + beta * sqrt(log(total_evals + 1) / (family_evals + 1))

           Families with few evaluations get a large exploration bonus, which
           encourages the search to try under-explored topological structures
           before committing to the locally dominant family.

    This mirrors the multi-armed bandit approach used in Bayesian optimizers
    like MIPRO but applied at the topology-family level rather than instruction
    token level.
    """

    def __init__(
        self,
        historical_records: list[dict[str, Any]],
        beta: float = 0.3,
        seed: int = 0,
    ) -> None:
        self._beta = beta
        self._rng = random.Random(seed)
        # Accumulate family evaluation counts and scores from history.
        self._family_evals: dict[str, int] = {}
        self._family_score_sum: dict[str, float] = {}
        for record in historical_records:
            sig = str(record.get("family_signature") or "")
            score = record.get("winning_score")
            if sig and score is not None:
                self._family_evals[sig] = self._family_evals.get(sig, 0) + 1
                self._family_score_sum[sig] = self._family_score_sum.get(sig, 0.0) + float(score)

    def _ucb(self, candidate: CandidateEvaluation, total_evals: int) -> float:
        sig = candidate.family_signature
        n = self._family_evals.get(sig, 0)
        mean_score = (
            self._family_score_sum.get(sig, 0.0) / n if n > 0 else candidate.score
        )
        exploration_bonus = self._beta * math.sqrt(
            math.log(total_evals + 1) / (n + 1)
        )
        return mean_score + exploration_bonus

    def choose(
        self,
        candidates: list[CandidateEvaluation],
        *,
        limit: int | None = None,
    ) -> list[CandidateEvaluation]:
        if not candidates:
            return candidates
        effective_limit = limit if limit is not None else len(candidates)

        total_evals = max(sum(self._family_evals.values()), 1)

        # Always keep the outright best candidate.
        best = max(candidates, key=lambda c: c.score)
        remaining = [c for c in candidates if c is not best]

        selected = [best]
        seen_families = {best.family_signature}

        # Score remaining by UCB and pick greedily with diversity preference.
        scored = sorted(remaining, key=lambda c: self._ucb(c, total_evals), reverse=True)
        for candidate in scored:
            if len(selected) >= effective_limit:
                break
            selected.append(candidate)
            seen_families.add(candidate.family_signature)

        return selected
