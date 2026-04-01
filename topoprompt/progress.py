from __future__ import annotations

from typing import Any, Iterable, TypeVar

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

T = TypeVar("T")


class CompileProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool = False,
        verbosity: int = 1,
        console: Console | None = None,
    ) -> None:
        self.enabled = enabled
        self.verbosity = verbosity
        self.console = console or Console(stderr=True, soft_wrap=True)

    def rule(self, title: str, *, level: int = 1, style: str = "bold cyan") -> None:
        if self.enabled and self.verbosity >= level:
            self.console.rule(title, style=style)

    def log(self, message: str, *, level: int = 1, style: str | None = None) -> None:
        if self.enabled and self.verbosity >= level:
            if style:
                self.console.log(f"[{style}]{message}[/{style}]")
            else:
                self.console.log(message)

    def track(
        self,
        iterable: Iterable[T],
        *,
        desc: str,
        total: int | None = None,
        leave: bool = False,
        level: int = 1,
    ) -> Iterable[T]:
        if not self.enabled or self.verbosity < level:
            return iterable
        return tqdm(
            iterable,
            desc=desc,
            total=total,
            leave=leave,
            dynamic_ncols=True,
            file=self.console.file,
        )

    def print_analysis(self, analysis: Any) -> None:
        if not self.enabled:
            return
        table = Table(title="Task Analysis", show_header=True, header_style="bold magenta")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")
        for field in [
            "task_family",
            "output_format",
            "needs_reasoning",
            "needs_verification",
            "needs_decomposition",
            "input_heterogeneity",
            "analyzer_confidence",
        ]:
            value = getattr(analysis, field, None)
            table.add_row(field, str(value))
        routes = ", ".join(getattr(route, "label", str(route)) for route in getattr(analysis, "candidate_routes", [])) or "-"
        seeds = ", ".join(getattr(analysis, "initial_seed_templates", []) or []) or "-"
        table.add_row("candidate_routes", routes)
        table.add_row("initial_seed_templates", seeds)
        table.add_row("rationale", getattr(analysis, "rationale", ""))
        self.console.print(table)

    def log_candidate(self, candidate: Any, *, prefix: str = "", level: int = 1) -> None:
        if not self.enabled or self.verbosity < level:
            return
        self.log(
            (
                f"{prefix}{candidate.program.program_id} "
                f"stage={candidate.stage} "
                f"score={candidate.score:.4f} "
                f"search={candidate.search_score:.4f} "
                f"calls={candidate.mean_invocations:.2f} "
                f"tokens={candidate.mean_tokens:.2f} "
                f"complexity={candidate.complexity:.3f} "
                f"parse_fail={candidate.parse_failure_rate:.3f}"
            ),
            level=level,
            style="green",
        )

    def log_example_result(self, *, program_id: str, example_id: str, score: float, invocations: int, parse_failures: int) -> None:
        self.log(
            (
                f"example {example_id} via {program_id}: "
                f"score={score:.2f} calls={invocations} parse_failures={parse_failures}"
            ),
            level=2,
            style="blue",
        )

    def log_node_event(
        self,
        *,
        program_id: str,
        example_id: str,
        node_id: str,
        node_type: str,
        route_choice: str | None = None,
        parse_error: str | None = None,
    ) -> None:
        detail = f"{program_id}:{example_id} -> {node_id} ({node_type})"
        if route_choice:
            detail += f" branch={route_choice}"
        if parse_error:
            detail += f" parse_error={parse_error}"
        self.log(detail, level=3, style="yellow")

    def log_budget(self, *, spent: int, planned: int, phase: str | None = None, level: int = 1) -> None:
        label = f"{phase}: " if phase else ""
        self.log(f"{label}budget {spent}/{planned} calls spent", level=level, style="cyan")

