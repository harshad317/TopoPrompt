from __future__ import annotations

from topoprompt.schemas import CandidateEvaluation


class NoOpPosterior:
    def rank(self, candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
        return candidates

