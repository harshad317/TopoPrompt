from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import orjson

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.compiler.search import evaluate_program_on_examples
from topoprompt.config import TopoPromptConfig
from topoprompt.eval.metrics import MetricFn
from topoprompt.eval.significance import build_significance_summary, render_significance_summary
from topoprompt.progress import CompileProgressReporter
from topoprompt.schemas import Example, PromptProgram, TaskSpec


def compare_programs(
    *,
    program_a: PromptProgram,
    program_b: PromptProgram,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_fn: MetricFn,
    backend: LLMBackend,
    config: TopoPromptConfig,
    label_a: str = "program_a",
    label_b: str = "program_b",
    phase: str = "confirmation",
    repeats: int = 1,
    output_dir: str | Path | None = None,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 0,
    show_progress: bool = False,
    progress_verbosity: int = 1,
    progress_reporter: CompileProgressReporter | None = None,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    reporter = progress_reporter or CompileProgressReporter(enabled=show_progress, verbosity=progress_verbosity)
    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    repeat_results: list[dict[str, Any]] = []
    disagreement_rows_by_repeat: list[list[dict[str, Any]]] = []

    for repeat_index in range(repeats):
        if repeats > 1:
            reporter.rule(f"Compare Repeat {repeat_index + 1}", level=1, style="bold blue")
        config_a = _comparison_config(config=config, repeat_index=repeat_index + 1, side="a", output_dir=out_dir)
        config_b = _comparison_config(config=config, repeat_index=repeat_index + 1, side="b", output_dir=out_dir)
        reporter.log(f"Evaluating {label_a}: {program_a.program_id}", level=1)
        result_a = evaluate_program_on_examples(
            program=program_a,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            backend=backend,
            config=config_a,
            phase=phase,
            show_progress=show_progress,
            progress_verbosity=progress_verbosity,
            progress_reporter=reporter,
        )
        reporter.log(f"Evaluating {label_b}: {program_b.program_id}", level=1)
        result_b = evaluate_program_on_examples(
            program=program_b,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            backend=backend,
            config=config_b,
            phase=phase,
            show_progress=show_progress,
            progress_verbosity=progress_verbosity,
            progress_reporter=reporter,
        )
        repeat_summary, disagreement_rows = _compare_repeat_results(
            repeat_index=repeat_index + 1,
            label_a=label_a,
            label_b=label_b,
            program_a=program_a,
            program_b=program_b,
            examples=examples,
            result_a=result_a,
            result_b=result_b,
        )
        repeat_results.append(repeat_summary)
        disagreement_rows_by_repeat.append(disagreement_rows)

    summary = _build_compare_summary(
        label_a=label_a,
        label_b=label_b,
        program_a=program_a,
        program_b=program_b,
        sample_count=len(examples),
        repeat_results=repeat_results,
    )
    significance = build_significance_summary(
        label_a=label_a,
        label_b=label_b,
        program_a_id=program_a.program_id,
        program_b_id=program_b.program_id,
        sample_count=len(examples),
        repeat_results=repeat_results,
        confidence_level=confidence_level,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    summary["significance"] = significance

    if out_dir is not None:
        _write_json(out_dir / "compare_summary.json", summary)
        _write_jsonl(out_dir / "repeat_metrics.jsonl", repeat_results)
        for repeat_result, disagreement_rows in zip(repeat_results, disagreement_rows_by_repeat, strict=False):
            _write_jsonl(out_dir / f"disagreements_repeat_{repeat_result['repeat_index']}.jsonl", disagreement_rows)
        (out_dir / "compare_summary.md").write_text(_render_compare_summary(summary))
        _write_json(out_dir / "significance_summary.json", significance)
        (out_dir / "significance_summary.md").write_text(render_significance_summary(significance))

    return summary


def _compare_repeat_results(
    *,
    repeat_index: int,
    label_a: str,
    label_b: str,
    program_a: PromptProgram,
    program_b: PromptProgram,
    examples: list[Example],
    result_a: dict[str, Any],
    result_b: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traces_a = {trace["example_id"]: trace for trace in result_a["traces"]}
    traces_b = {trace["example_id"]: trace for trace in result_b["traces"]}

    a_better_count = 0
    b_better_count = 0
    tied_count = 0
    both_positive_count = 0
    a_only_positive_count = 0
    b_only_positive_count = 0
    both_zero_count = 0
    disagreement_rows: list[dict[str, Any]] = []

    for example in examples:
        trace_a = traces_a.get(example.example_id, {})
        trace_b = traces_b.get(example.example_id, {})
        score_a = float(trace_a.get("correctness") or 0.0)
        score_b = float(trace_b.get("correctness") or 0.0)
        final_output_a = trace_a.get("final_output")
        final_output_b = trace_b.get("final_output")

        if score_a > score_b:
            a_better_count += 1
        elif score_b > score_a:
            b_better_count += 1
        else:
            tied_count += 1

        positive_a = score_a > 0.0
        positive_b = score_b > 0.0
        if positive_a and positive_b:
            both_positive_count += 1
        elif positive_a:
            a_only_positive_count += 1
        elif positive_b:
            b_only_positive_count += 1
        else:
            both_zero_count += 1

        if score_a != score_b or final_output_a != final_output_b:
            disagreement_rows.append(
                {
                    "repeat_index": repeat_index,
                    "example_id": example.example_id,
                    "input": example.input,
                    "target": example.target,
                    "program_a_label": label_a,
                    "program_a_id": program_a.program_id,
                    "program_a_output": final_output_a,
                    "program_a_score": score_a,
                    "program_b_label": label_b,
                    "program_b_id": program_b.program_id,
                    "program_b_output": final_output_b,
                    "program_b_score": score_b,
                }
            )

    return (
        {
            "repeat_index": repeat_index,
            "program_a_id": program_a.program_id,
            "program_b_id": program_b.program_id,
            "score_a": result_a["score"],
            "score_b": result_b["score"],
            "score_delta_a_minus_b": result_a["score"] - result_b["score"],
            "score_delta_b_minus_a": result_b["score"] - result_a["score"],
            "mean_invocations_a": result_a["mean_invocations"],
            "mean_invocations_b": result_b["mean_invocations"],
            "a_better_count": a_better_count,
            "b_better_count": b_better_count,
            "tied_count": tied_count,
            "both_positive_count": both_positive_count,
            "a_only_positive_count": a_only_positive_count,
            "b_only_positive_count": b_only_positive_count,
            "both_zero_count": both_zero_count,
            "disagreement_count": len(disagreement_rows),
        },
        disagreement_rows,
    )


def _build_compare_summary(
    *,
    label_a: str,
    label_b: str,
    program_a: PromptProgram,
    program_b: PromptProgram,
    sample_count: int,
    repeat_results: list[dict[str, Any]],
) -> dict[str, Any]:
    score_a_values = [float(row["score_a"]) for row in repeat_results]
    score_b_values = [float(row["score_b"]) for row in repeat_results]
    invocations_a_values = [float(row["mean_invocations_a"]) for row in repeat_results]
    invocations_b_values = [float(row["mean_invocations_b"]) for row in repeat_results]
    delta_values = [float(row["score_delta_b_minus_a"]) for row in repeat_results]

    return {
        "label_a": label_a,
        "label_b": label_b,
        "program_a_id": program_a.program_id,
        "program_b_id": program_b.program_id,
        "sample_count": sample_count,
        "repeats": len(repeat_results),
        "score_a_mean": mean(score_a_values),
        "score_a_std": _std(score_a_values),
        "score_b_mean": mean(score_b_values),
        "score_b_std": _std(score_b_values),
        "score_delta_a_minus_b_mean": mean(float(row["score_delta_a_minus_b"]) for row in repeat_results),
        "score_delta_a_minus_b_std": _std([float(row["score_delta_a_minus_b"]) for row in repeat_results]),
        "score_delta_b_minus_a_mean": mean(delta_values),
        "score_delta_b_minus_a_std": _std(delta_values),
        "mean_invocations_a_mean": mean(invocations_a_values),
        "mean_invocations_a_std": _std(invocations_a_values),
        "mean_invocations_b_mean": mean(invocations_b_values),
        "mean_invocations_b_std": _std(invocations_b_values),
        "a_better_count_mean": mean(float(row["a_better_count"]) for row in repeat_results),
        "b_better_count_mean": mean(float(row["b_better_count"]) for row in repeat_results),
        "tied_count_mean": mean(float(row["tied_count"]) for row in repeat_results),
        "both_positive_count_mean": mean(float(row["both_positive_count"]) for row in repeat_results),
        "a_only_positive_count_mean": mean(float(row["a_only_positive_count"]) for row in repeat_results),
        "b_only_positive_count_mean": mean(float(row["b_only_positive_count"]) for row in repeat_results),
        "both_zero_count_mean": mean(float(row["both_zero_count"]) for row in repeat_results),
        "disagreement_count_mean": mean(float(row["disagreement_count"]) for row in repeat_results),
        "repeat_results": repeat_results,
    }


def _render_compare_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# TopoPrompt Compare Summary",
        "",
        f"- Program A: `{summary['label_a']}` / `{summary['program_a_id']}`",
        f"- Program B: `{summary['label_b']}` / `{summary['program_b_id']}`",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Repeats: `{summary['repeats']}`",
        f"- Score A mean/std: `{summary['score_a_mean']:.4f}` / `{summary['score_a_std']:.4f}`",
        f"- Score B mean/std: `{summary['score_b_mean']:.4f}` / `{summary['score_b_std']:.4f}`",
        f"- Delta (A - B) mean/std: `{summary['score_delta_a_minus_b_mean']:.4f}` / `{summary['score_delta_a_minus_b_std']:.4f}`",
        f"- Delta (B - A) mean/std: `{summary['score_delta_b_minus_a_mean']:.4f}` / `{summary['score_delta_b_minus_a_std']:.4f}`",
        f"- Mean invocations A: `{summary['mean_invocations_a_mean']:.2f}`",
        f"- Mean invocations B: `{summary['mean_invocations_b_mean']:.2f}`",
        "",
        "## Per Repeat",
        "",
    ]
    for row in summary["repeat_results"]:
        lines.append(
            (
                f"- Repeat {row['repeat_index']}: "
                f"score_a={row['score_a']:.4f}, "
                f"score_b={row['score_b']:.4f}, "
                f"delta={row['score_delta_b_minus_a']:.4f}, "
                f"a_better={row['a_better_count']}, "
                f"b_better={row['b_better_count']}, "
                f"ties={row['tied_count']}, "
                f"disagreements={row['disagreement_count']}"
            )
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = b"\n".join(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) for row in rows)
    path.write_bytes(content + (b"\n" if rows else b""))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values)


def _comparison_config(
    *,
    config: TopoPromptConfig,
    repeat_index: int,
    side: str,
    output_dir: Path | None,
) -> TopoPromptConfig:
    cloned = TopoPromptConfig.model_validate(config.model_dump())
    if not cloned.runtime.cache_enabled:
        return cloned
    cache_root = output_dir / ".compare_cache" if output_dir is not None else Path(cloned.runtime.cache_dir)
    cloned.runtime.cache_dir = str(cache_root / f"repeat_{repeat_index}_{side}")
    return cloned
