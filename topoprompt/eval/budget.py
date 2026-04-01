from __future__ import annotations

from dataclasses import dataclass

from topoprompt.config import CompileConfig
from topoprompt.schemas import BudgetPhaseSpend


@dataclass
class BudgetLedger:
    analyzer_budget_calls: int
    seed_budget_calls: int
    screening_budget_calls: int
    narrowing_budget_calls: int
    confirmation_budget_calls: int
    reserve_budget_calls: int
    spent: dict[str, int]

    @classmethod
    def from_compile_config(cls, config: CompileConfig) -> "BudgetLedger":
        return cls(
            analyzer_budget_calls=config.analyzer_budget_calls,
            seed_budget_calls=config.seed_budget_calls,
            screening_budget_calls=config.screening_budget_calls,
            narrowing_budget_calls=config.narrowing_budget_calls,
            confirmation_budget_calls=config.confirmation_budget_calls,
            reserve_budget_calls=config.reserve_budget_calls,
            spent={
                "analyzer": 0,
                "seed": 0,
                "screening": 0,
                "narrowing": 0,
                "confirmation": 0,
                "reserve": 0,
            },
        )

    def planned_total(self) -> int:
        return (
            self.analyzer_budget_calls
            + self.seed_budget_calls
            + self.screening_budget_calls
            + self.narrowing_budget_calls
            + self.confirmation_budget_calls
            + self.reserve_budget_calls
        )

    def spent_total(self) -> int:
        return sum(self.spent.values())

    def remaining(self, phase: str) -> int:
        phase_budget = getattr(self, f"{phase}_budget_calls")
        return max(phase_budget - self.spent.get(phase, 0), 0)

    def can_spend(self, phase: str, calls: int = 1, *, allow_reserve: bool = False) -> bool:
        if self.remaining(phase) >= calls:
            return True
        if allow_reserve and self.remaining("reserve") >= calls:
            return True
        return False

    def spend(self, phase: str, calls: int = 1, *, allow_reserve: bool = False) -> bool:
        if self.remaining(phase) >= calls:
            self.spent[phase] += calls
            return True
        if allow_reserve and self.remaining("reserve") >= calls:
            self.spent["reserve"] += calls
            return True
        return False

    def snapshot(self) -> list[BudgetPhaseSpend]:
        return [
            BudgetPhaseSpend(phase="analyzer", planned_calls=self.analyzer_budget_calls, spent_calls=self.spent["analyzer"]),
            BudgetPhaseSpend(phase="seed", planned_calls=self.seed_budget_calls, spent_calls=self.spent["seed"]),
            BudgetPhaseSpend(phase="screening", planned_calls=self.screening_budget_calls, spent_calls=self.spent["screening"]),
            BudgetPhaseSpend(phase="narrowing", planned_calls=self.narrowing_budget_calls, spent_calls=self.spent["narrowing"]),
            BudgetPhaseSpend(
                phase="confirmation",
                planned_calls=self.confirmation_budget_calls,
                spent_calls=self.spent["confirmation"],
            ),
            BudgetPhaseSpend(phase="reserve", planned_calls=self.reserve_budget_calls, spent_calls=self.spent["reserve"]),
        ]

