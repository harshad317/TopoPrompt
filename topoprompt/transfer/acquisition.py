from __future__ import annotations

from topoprompt.schemas import CandidateEvaluation


class NoOpAcquisition:
    def choose(self, candidates: list[CandidateEvaluation], *, limit: int | None = None) -> list[CandidateEvaluation]:
        return candidates if limit is None else candidates[:limit]

