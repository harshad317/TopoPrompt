from __future__ import annotations

from statistics import mean, stdev
from typing import Any

from topoprompt.schemas import CandidateEvaluation


class NoOpPosterior:
    def rank(self, candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
        return candidates


class HistoricalPosterior:
    """Ranks candidates by combining their observed score with a prior derived
    from historical compile records that share the same topology family.

    For each candidate the posterior estimate is:

        posterior = w_obs * observed_score + w_prior * prior_mean

    where ``w_prior`` scales with how many historical records exist for this
    topology family (more history → stronger prior) and ``w_obs = 1 - w_prior``.

    This makes warm-started programs that previously performed well on similar
    tasks rank higher even before expensive evaluation, which reduces wasted
    budget on topologies with a poor track record.
    """

    # Maximum weight given to the prior regardless of how much history exists.
    _MAX_PRIOR_WEIGHT: float = 0.35
    # Number of historical records after which the prior reaches max weight.
    _SATURATION_COUNT: int = 10

    def __init__(self, historical_records: list[dict[str, Any]]) -> None:
        # Build a lookup: family_signature → list[winning_score]
        self._family_scores: dict[str, list[float]] = {}
        for record in historical_records:
            if record.get("record_type") != "compile_winner":
                continue
            sig = str(record.get("family_signature") or "")
            score = record.get("winning_score")
            if sig and score is not None:
                self._family_scores.setdefault(sig, []).append(float(score))

    def _prior_for_family(self, family_signature: str) -> tuple[float, float]:
        """Return (prior_mean, prior_weight) for a topology family."""
        scores = self._family_scores.get(family_signature, [])
        if not scores:
            return 0.5, 0.0
        prior_mean = mean(scores)
        # Weight grows with evidence count, saturates at _MAX_PRIOR_WEIGHT.
        prior_weight = self._MAX_PRIOR_WEIGHT * min(
            len(scores) / self._SATURATION_COUNT, 1.0
        )
        return prior_mean, prior_weight

    def rank(self, candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
        if not candidates:
            return candidates

        def posterior_score(candidate: CandidateEvaluation) -> float:
            prior_mean, prior_weight = self._prior_for_family(candidate.family_signature)
            obs_weight = 1.0 - prior_weight
            return obs_weight * candidate.score + prior_weight * prior_mean

        return sorted(candidates, key=posterior_score, reverse=True)
