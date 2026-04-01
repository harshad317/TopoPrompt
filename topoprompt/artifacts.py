from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import yaml

from topoprompt.schemas import CompileArtifact, ProgramExecutionTrace, PromptProgram, TaskSpec


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = b"\n".join(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) for row in rows)
    path.write_bytes(content + (b"\n" if rows else b""))


def save_program_json(program: PromptProgram, path: str | Path) -> None:
    target = Path(path)
    _write_json(target, program.model_dump(mode="json"))


def save_program_yaml(program: PromptProgram, path: str | Path) -> None:
    target = Path(path)
    target.write_text(yaml.safe_dump(program.model_dump(mode="json"), sort_keys=False))


def save_task_spec_json(task_spec: TaskSpec, path: str | Path) -> None:
    target = Path(path)
    _write_json(target, task_spec.model_dump(mode="json"))


def save_compile_traces_jsonl(traces: list[ProgramExecutionTrace], path: str | Path) -> None:
    rows = [trace.model_dump(mode="json") for trace in traces]
    _write_jsonl(Path(path), rows)


def write_compile_artifact(artifact: CompileArtifact, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_task_spec_json(artifact.task_spec, out_dir / "task_spec.json")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(artifact.config, sort_keys=False))
    _write_json(out_dir / "seed_programs.json", [seed.model_dump(mode="json") for seed in artifact.seed_programs])
    _write_json(out_dir / "final_program.json", artifact.program_ir.model_dump(mode="json"))
    save_program_yaml(artifact.program_ir, out_dir / "final_program.yaml")
    save_compile_traces_jsonl(artifact.compile_trace, out_dir / "compile_trace.jsonl")
    _write_jsonl(out_dir / "candidate_archive.jsonl", [record.model_dump(mode="json") for record in artifact.candidate_archive])
    _write_json(out_dir / "metrics.json", artifact.metrics.model_dump(mode="json"))
    summary = _render_summary(artifact)
    (out_dir / "summary.md").write_text(summary)
    artifact.output_dir = str(out_dir)
    return out_dir


def _render_summary(artifact: CompileArtifact) -> str:
    metrics = artifact.metrics
    lines = [
        "# TopoPrompt Compile Summary",
        "",
        f"- Task: `{artifact.task_spec.task_id}`",
        f"- Winning program: `{metrics.smallest_effective_program_id}`",
        f"- Best validation score: `{metrics.best_validation_score:.4f}`",
        f"- Smallest effective score: `{metrics.smallest_effective_score:.4f}`",
        f"- Epsilon: `{metrics.epsilon:.4f}`",
        f"- Winning topology family: `{metrics.winning_topology_family}`",
        f"- Planned budget calls: `{metrics.planned_budget_calls}`",
        f"- Spent budget calls: `{metrics.spent_budget_calls}`",
        "",
        "## Budget by phase",
        "",
    ]
    for phase in metrics.spent_budget_by_phase:
        lines.append(f"- `{phase.phase}`: spent {phase.spent_calls}/{phase.planned_calls}")
    return "\n".join(lines) + "\n"
