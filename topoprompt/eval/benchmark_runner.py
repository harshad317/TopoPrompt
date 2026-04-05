from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import orjson

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.compiler.search import compile_task, evaluate_program_on_examples
from topoprompt.config import TopoPromptConfig
from topoprompt.eval.compare import compare_programs
from topoprompt.eval.dspy_baselines import compare_topoprompt_vs_dspy, compile_dspy_baseline
from topoprompt.eval.datasets import load_benchmark_examples, partition_examples
from topoprompt.eval.metrics import metric_for_name
from topoprompt.eval.significance import build_significance_summary
from topoprompt.schemas import CompileArtifact, Example, PromptProgram, TaskAnalysis, TaskSpec


DEFAULT_BENCHMARK_TASKS = {
    "gsm8k": "Solve grade-school math word problems accurately.",
    "sst2": "Classify the sentiment of each sentence as positive or negative.",
    "mmlu": "Answer multiple-choice knowledge questions correctly.",
    "bbh": "Solve diverse reasoning tasks with the correct final answer.",
    "ifeval": "Follow instructions precisely and satisfy all stated constraints.",
}

BBH_TASK_FAMILY_MAP = {
    "boolean_expressions": "logical_deduction",
    "causal_judgement": "commonsense_qa",
    "date_understanding": "commonsense_qa",
    "disambiguation_qa": "commonsense_qa",
    "dyck_languages": "symbolic_structures",
    "formal_fallacies": "logical_deduction",
    "geometric_shapes": "symbolic_structures",
    "hyperbaton": "language_understanding",
    "logical_deduction_three_objects": "logical_deduction",
    "logical_deduction_five_objects": "logical_deduction",
    "logical_deduction_seven_objects": "logical_deduction",
    "movie_recommendation": "commonsense_qa",
    "multistep_arithmetic_two": "arithmetic_counting",
    "navigate": "state_tracking",
    "object_counting": "arithmetic_counting",
    "penguins_in_a_table": "commonsense_qa",
    "reasoning_about_colored_objects": "state_tracking",
    "ruin_names": "language_understanding",
    "salient_translation_error_detection": "language_understanding",
    "snarks": "language_understanding",
    "sports_understanding": "commonsense_qa",
    "temporal_sequences": "state_tracking",
    "tracking_shuffled_objects_three_objects": "state_tracking",
    "tracking_shuffled_objects_five_objects": "state_tracking",
    "tracking_shuffled_objects_seven_objects": "state_tracking",
    "web_of_lies": "logical_deduction",
    "word_sorting": "language_understanding",
}


class BenchmarkRunner:
    def __init__(self, *, config: TopoPromptConfig, backend: LLMBackend) -> None:
        self.config = config
        self.backend = backend

    def compile_benchmark(
        self,
        *,
        benchmark_name: str,
        examples_path: str | Path | None = None,
        task_description: str | None = None,
        output_dir: str | Path | None = None,
    ) -> CompileArtifact:
        examples = load_benchmark_examples(benchmark_name, path=examples_path)
        task_description = task_description or DEFAULT_BENCHMARK_TASKS[benchmark_name.lower()]
        metric_name = benchmark_name.lower()
        return compile_task(
            task_description=task_description,
            examples=examples,
            metric=metric_name,
            backend=self.backend,
            config=self.config,
            output_dir=output_dir,
        )

    def evaluate_program(
        self,
        *,
        program: PromptProgram,
        task_spec: TaskSpec,
        benchmark_name: str,
        examples_path: str | Path | None = None,
        split: str | None = None,
    ) -> dict[str, Any]:
        examples = load_benchmark_examples(benchmark_name, path=examples_path, split=split)
        metric_name = benchmark_name.lower()
        metric_fn = metric_for_name(metric_name)
        return evaluate_program_on_examples(
            program=program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            backend=self.backend,
            config=self.config,
            phase="confirmation",
        )

    def compile_and_evaluate(
        self,
        *,
        benchmark_name: str,
        examples_path: str | Path | None = None,
        task_description: str | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        examples = load_benchmark_examples(benchmark_name, path=examples_path)
        partitions = partition_examples(examples, data_config=self.config.data, create_test_split=True)
        artifact = compile_task(
            task_description=task_description or DEFAULT_BENCHMARK_TASKS[benchmark_name.lower()],
            examples=partitions.compile_examples + partitions.validation_examples,
            metric=benchmark_name.lower(),
            backend=self.backend,
            config=self.config,
            output_dir=output_dir,
        )
        evaluation = evaluate_program_on_examples(
            program=artifact.program_ir,
            task_spec=artifact.task_spec,
            examples=partitions.test_examples or partitions.validation_examples,
            metric_fn=metric_for_name(benchmark_name.lower()),
            backend=self.backend,
            config=self.config,
            phase="confirmation",
        )
        return {"artifact": artifact, "evaluation": evaluation}

    def compile_and_compare_by_family(
        self,
        *,
        benchmark_name: str,
        examples_path: str | Path | None = None,
        split: str | None = None,
        task_description: str | None = None,
        output_dir: str | Path | None = None,
        grouping: str = "family",
        include_groups: list[str] | None = None,
        compare_repeats: int = 1,
        compile_budget: int | None = None,
        show_progress: bool = False,
        progress_verbosity: int = 1,
    ) -> dict[str, Any]:
        normalized_benchmark = benchmark_name.lower()
        if normalized_benchmark != "bbh":
            raise ValueError("Family-aware benchmark compilation is currently supported only for BBH.")

        examples = load_benchmark_examples(benchmark_name, path=examples_path, split=split)
        grouped_examples = group_benchmark_examples(
            benchmark_name=normalized_benchmark,
            examples=examples,
            grouping=grouping,
        )
        if include_groups is not None:
            allowed = set(include_groups)
            grouped_examples = {
                group_name: group_examples
                for group_name, group_examples in grouped_examples.items()
                if group_name in allowed
            }
        if not grouped_examples:
            raise ValueError("No benchmark families matched the requested filters.")

        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        family_summaries: dict[str, dict[str, Any]] = {}
        compiled_vs_direct_results: list[dict[str, Any]] = []
        compiled_vs_direct_x3_results: list[dict[str, Any]] = []

        for group_name, group_examples in grouped_examples.items():
            partitions = partition_examples(group_examples, data_config=self.config.data, create_test_split=True)
            eval_examples = partitions.test_examples or partitions.validation_examples
            group_dir = out_dir / _slugify_group_name(group_name) if out_dir is not None else None
            artifact = compile_task(
                task_description=_group_task_description(
                    benchmark_name=normalized_benchmark,
                    group_name=group_name,
                    group_examples=group_examples,
                    task_description=task_description or DEFAULT_BENCHMARK_TASKS[normalized_benchmark],
                    grouping=grouping,
                ),
                examples=group_examples,
                search_examples=partitions.search_examples,
                validation_examples=partitions.validation_examples,
                fewshot_examples=partitions.fewshot_examples,
                metric=normalized_benchmark,
                backend=self.backend,
                config=self.config,
                compile_budget=compile_budget,
                output_dir=group_dir,
                task_id=f"{normalized_benchmark}_{_slugify_group_name(group_name)}",
                show_progress=show_progress,
                progress_verbosity=progress_verbosity,
            )

            direct = instantiate_seed_program(
                task_spec=artifact.task_spec,
                analysis=TaskAnalysis(initial_seed_templates=["direct_finalize"]),
                template_name="direct_finalize",
            )
            direct_x3 = instantiate_seed_program(
                task_spec=artifact.task_spec,
                analysis=TaskAnalysis(initial_seed_templates=["direct_self_consistency_x3"]),
                template_name="direct_self_consistency_x3",
            )
            assert direct is not None
            assert direct_x3 is not None

            compiled_vs_direct = compare_programs(
                program_a=artifact.program_ir,
                program_b=direct,
                task_spec=artifact.task_spec,
                examples=eval_examples,
                metric_fn=metric_for_name(normalized_benchmark),
                backend=self.backend,
                config=self.config,
                label_a="compiled",
                label_b="direct",
                repeats=compare_repeats,
                output_dir=(group_dir / "compare_compiled_vs_direct") if group_dir is not None else None,
                show_progress=show_progress,
                progress_verbosity=progress_verbosity,
            )
            compiled_vs_direct_x3 = compare_programs(
                program_a=artifact.program_ir,
                program_b=direct_x3,
                task_spec=artifact.task_spec,
                examples=eval_examples,
                metric_fn=metric_for_name(normalized_benchmark),
                backend=self.backend,
                config=self.config,
                label_a="compiled",
                label_b="direct_x3",
                repeats=compare_repeats,
                output_dir=(group_dir / "compare_compiled_vs_direct_x3") if group_dir is not None else None,
                show_progress=show_progress,
                progress_verbosity=progress_verbosity,
            )

            compiled_vs_direct_results.append(compiled_vs_direct)
            compiled_vs_direct_x3_results.append(compiled_vs_direct_x3)

            family_summaries[group_name] = {
                "run_dir": str(group_dir) if group_dir is not None else None,
                "task_names": _group_task_names(group_examples, group_name=group_name),
                "source_examples": len(group_examples),
                "eval_examples": len(eval_examples),
                "final_program_id": artifact.metrics.final_program_id,
                "best_validation_score": artifact.metrics.best_validation_score,
                "compiled_score": compiled_vs_direct["score_a_mean"],
                "direct_score": compiled_vs_direct["score_b_mean"],
                "compiled_minus_direct": compiled_vs_direct["score_a_mean"] - compiled_vs_direct["score_b_mean"],
                "compiled_vs_direct_mcnemar_p": _compare_p_value(compiled_vs_direct),
                "compiled_score_vs_direct_x3": compiled_vs_direct_x3["score_a_mean"],
                "direct_x3_score": compiled_vs_direct_x3["score_b_mean"],
                "compiled_minus_direct_x3": compiled_vs_direct_x3["score_a_mean"] - compiled_vs_direct_x3["score_b_mean"],
                "compiled_vs_direct_x3_mcnemar_p": _compare_p_value(compiled_vs_direct_x3),
            }

        aggregate_vs_direct = _aggregate_compare_summaries(compiled_vs_direct_results)
        aggregate_vs_direct_x3 = _aggregate_compare_summaries(compiled_vs_direct_x3_results)
        summary = {
            "benchmark_name": normalized_benchmark,
            "grouping": grouping,
            "group_count": len(family_summaries),
            "total_source_examples": len(examples),
            "total_eval_examples": sum(payload["eval_examples"] for payload in family_summaries.values()),
            "aggregate": {
                "compiled_score": aggregate_vs_direct["score_a_mean"],
                "direct_score": aggregate_vs_direct["score_b_mean"],
                "compiled_minus_direct": aggregate_vs_direct["score_delta_a_minus_b_mean"],
                "compiled_vs_direct_mcnemar_p": _compare_p_value(aggregate_vs_direct),
                "compiled_score_vs_direct_x3": aggregate_vs_direct_x3["score_a_mean"],
                "direct_x3_score": aggregate_vs_direct_x3["score_b_mean"],
                "compiled_minus_direct_x3": aggregate_vs_direct_x3["score_delta_a_minus_b_mean"],
                "compiled_vs_direct_x3_mcnemar_p": _compare_p_value(aggregate_vs_direct_x3),
                "positive_groups_vs_direct": sum(
                    1 for payload in family_summaries.values() if payload["compiled_minus_direct"] > 0.0
                ),
                "positive_groups_vs_direct_x3": sum(
                    1 for payload in family_summaries.values() if payload["compiled_minus_direct_x3"] > 0.0
                ),
            },
            "families": family_summaries,
            "aggregate_compare_compiled_vs_direct": aggregate_vs_direct,
            "aggregate_compare_compiled_vs_direct_x3": aggregate_vs_direct_x3,
        }
        if out_dir is not None:
            _write_json(out_dir / "family_summary.json", summary)
            (out_dir / "family_summary.md").write_text(_render_family_summary(summary))
        return summary

    def compile_and_compare_with_dspy(
        self,
        *,
        benchmark_name: str,
        optimizers: str | list[str] | tuple[str, ...] = ("mipro", "gepa"),
        examples_path: str | Path | None = None,
        split: str | None = None,
        task_description: str | None = None,
        output_dir: str | Path | None = None,
        compare_repeats: int = 1,
        compile_budget: int | None = None,
        student_strategy: str = "auto",
        model_name: str | None = None,
        reflection_model_name: str | None = None,
        optimizer_auto: str = "light",
        show_progress: bool = False,
        progress_verbosity: int = 1,
    ) -> dict[str, Any]:
        normalized_benchmark = benchmark_name.lower()
        examples = load_benchmark_examples(benchmark_name, path=examples_path, split=split)
        if any(example.target is None for example in examples):
            raise ValueError(
                "Benchmark comparison requires labeled examples. Choose a split with targets."
            )

        task_description = task_description or DEFAULT_BENCHMARK_TASKS.get(
            normalized_benchmark,
            f"Solve the {normalized_benchmark} task accurately.",
        )
        partitions = partition_examples(examples, data_config=self.config.data, create_test_split=True)
        eval_examples = partitions.test_examples or partitions.validation_examples
        metric_fn = metric_for_name(normalized_benchmark)
        normalized_optimizers = _normalize_optimizer_names(optimizers)

        out_dir = Path(output_dir) if output_dir is not None else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        topoprompt_dir = out_dir / "topoprompt" if out_dir is not None else None
        artifact = compile_task(
            task_description=task_description,
            examples=examples,
            search_examples=partitions.search_examples,
            validation_examples=partitions.validation_examples,
            fewshot_examples=partitions.fewshot_examples,
            metric=normalized_benchmark,
            backend=self.backend,
            config=self.config,
            compile_budget=compile_budget,
            output_dir=topoprompt_dir,
            task_id=f"{normalized_benchmark}_compiled",
            show_progress=show_progress,
            progress_verbosity=progress_verbosity,
        )

        comparison_summaries: dict[str, dict[str, Any]] = {}
        for optimizer_name in normalized_optimizers:
            dspy_dir = out_dir / f"dspy_{optimizer_name}" if out_dir is not None else None
            dspy_result = compile_dspy_baseline(
                optimizer_name=optimizer_name,
                task_spec=artifact.task_spec,
                train_examples=partitions.compile_examples,
                val_examples=partitions.validation_examples,
                metric_fn=metric_fn,
                config=self.config,
                student_strategy=student_strategy,
                model_name=model_name or self.config.model.name,
                reflection_model_name=reflection_model_name,
                optimizer_auto=optimizer_auto,
                output_dir=dspy_dir,
                show_progress=show_progress,
                progress_verbosity=progress_verbosity,
            )
            compare_dir = out_dir / f"compare_topoprompt_vs_{optimizer_name}" if out_dir is not None else None
            compare_summary = compare_topoprompt_vs_dspy(
                topoprompt_program=artifact.program_ir,
                dspy_program=dspy_result["program"],
                task_spec=artifact.task_spec,
                examples=eval_examples,
                metric_fn=metric_fn,
                backend=self.backend,
                config=self.config,
                dspy_model_name=model_name or self.config.model.name,
                label_topoprompt="topoprompt",
                label_dspy=f"dspy_{optimizer_name}",
                repeats=compare_repeats,
                output_dir=compare_dir,
                show_progress=show_progress,
                progress_verbosity=progress_verbosity,
            )
            comparison_summaries[optimizer_name] = {
                "optimizer_name": optimizer_name,
                "dspy_program_id": dspy_result["summary"]["program_id"],
                "student_strategy": dspy_result["summary"]["student_strategy"],
                "compile_seconds": dspy_result["summary"]["compile_seconds"],
                "topoprompt_score": compare_summary["score_a_mean"],
                "dspy_score": compare_summary["score_b_mean"],
                "topoprompt_minus_dspy": compare_summary["score_delta_a_minus_b_mean"],
                "mcnemar_exact_p_value": _compare_p_value(compare_summary),
                "dspy_output_dir": str(dspy_dir) if dspy_dir is not None else None,
                "compare_output_dir": str(compare_dir) if compare_dir is not None else None,
            }

        summary = {
            "benchmark_name": normalized_benchmark,
            "task_description": task_description,
            "metric_name": normalized_benchmark,
            "task_family": artifact.task_spec.task_family,
            "source_examples": len(examples),
            "train_examples": len(partitions.compile_examples),
            "validation_examples": len(partitions.validation_examples),
            "eval_examples": len(eval_examples),
            "topoprompt": {
                "program_id": artifact.program_ir.program_id,
                "validation_score": artifact.metrics.final_program_score,
                "output_dir": str(topoprompt_dir) if topoprompt_dir is not None else artifact.output_dir,
            },
            "comparisons": comparison_summaries,
        }
        if out_dir is not None:
            _write_json(out_dir / "benchmark_dspy_summary.json", summary)
            (out_dir / "benchmark_dspy_summary.md").write_text(_render_benchmark_dspy_summary(summary))
        return summary


def group_benchmark_examples(
    *,
    benchmark_name: str,
    examples: list[Example],
    grouping: str = "family",
) -> dict[str, list[Example]]:
    normalized_benchmark = benchmark_name.lower()
    if normalized_benchmark != "bbh":
        raise ValueError("Family-aware grouping is currently supported only for BBH.")
    if grouping not in {"family", "task"}:
        raise ValueError(f"Unsupported grouping mode: {grouping}")

    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        task_name = str(example.metadata.get("bbh_task") or "unknown")
        group_name = task_name if grouping == "task" else bbh_family_for_task(task_name)
        grouped[group_name].append(example)
    return dict(sorted(grouped.items()))


def bbh_family_for_task(task_name: str) -> str:
    return BBH_TASK_FAMILY_MAP.get(task_name, "other")


def _group_task_description(
    *,
    benchmark_name: str,
    group_name: str,
    group_examples: list[Example],
    task_description: str,
    grouping: str,
) -> str:
    if benchmark_name != "bbh":
        return task_description
    task_names = _group_task_names(group_examples, group_name=group_name)
    if grouping == "task":
        return f"{task_description} Focus on the BBH task `{group_name}`."
    tasks_fragment = ", ".join(task_names)
    return f"{task_description} Focus on the BBH family `{group_name}` covering tasks: {tasks_fragment}."


def _group_task_names(group_examples: list[Example], *, group_name: str) -> list[str]:
    task_names = sorted({str(example.metadata.get("bbh_task") or group_name) for example in group_examples})
    return task_names or [group_name]


def _aggregate_compare_summaries(compare_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not compare_summaries:
        raise ValueError("compare_summaries must not be empty")
    label_a = str(compare_summaries[0]["label_a"])
    label_b = str(compare_summaries[0]["label_b"])
    aggregate_program_a_id = f"{label_a}_family_bank"
    aggregate_program_b_id = _shared_program_id(compare_summaries) or f"{label_b}_family_bank"
    repeats = int(compare_summaries[0]["repeats"])
    sample_count = sum(int(summary["sample_count"]) for summary in compare_summaries)
    aggregate_repeat_results: list[dict[str, Any]] = []

    for repeat_index in range(repeats):
        repeat_rows = [summary["repeat_results"][repeat_index] for summary in compare_summaries]
        aggregate_repeat_results.append(
            {
                "repeat_index": repeat_index + 1,
                "program_a_id": aggregate_program_a_id,
                "program_b_id": aggregate_program_b_id,
                "score_a": _weighted_repeat_metric(compare_summaries, repeat_rows, key="score_a"),
                "score_b": _weighted_repeat_metric(compare_summaries, repeat_rows, key="score_b"),
                "score_delta_a_minus_b": _weighted_repeat_metric(compare_summaries, repeat_rows, key="score_delta_a_minus_b"),
                "score_delta_b_minus_a": _weighted_repeat_metric(compare_summaries, repeat_rows, key="score_delta_b_minus_a"),
                "mean_invocations_a": _weighted_repeat_metric(compare_summaries, repeat_rows, key="mean_invocations_a"),
                "mean_invocations_b": _weighted_repeat_metric(compare_summaries, repeat_rows, key="mean_invocations_b"),
                "a_better_count": sum(int(row["a_better_count"]) for row in repeat_rows),
                "b_better_count": sum(int(row["b_better_count"]) for row in repeat_rows),
                "tied_count": sum(int(row["tied_count"]) for row in repeat_rows),
                "both_positive_count": sum(int(row["both_positive_count"]) for row in repeat_rows),
                "a_only_positive_count": sum(int(row["a_only_positive_count"]) for row in repeat_rows),
                "b_only_positive_count": sum(int(row["b_only_positive_count"]) for row in repeat_rows),
                "both_zero_count": sum(int(row["both_zero_count"]) for row in repeat_rows),
                "disagreement_count": sum(int(row["disagreement_count"]) for row in repeat_rows),
            }
        )

    significance = build_significance_summary(
        label_a=label_a,
        label_b=label_b,
        program_a_id=aggregate_program_a_id,
        program_b_id=aggregate_program_b_id,
        sample_count=sample_count,
        repeat_results=aggregate_repeat_results,
    )

    score_a_values = [float(row["score_a"]) for row in aggregate_repeat_results]
    score_b_values = [float(row["score_b"]) for row in aggregate_repeat_results]
    invocations_a_values = [float(row["mean_invocations_a"]) for row in aggregate_repeat_results]
    invocations_b_values = [float(row["mean_invocations_b"]) for row in aggregate_repeat_results]
    delta_values = [float(row["score_delta_a_minus_b"]) for row in aggregate_repeat_results]

    return {
        "label_a": label_a,
        "label_b": label_b,
        "program_a_id": aggregate_program_a_id,
        "program_b_id": aggregate_program_b_id,
        "sample_count": sample_count,
        "repeats": repeats,
        "score_a_mean": mean(score_a_values),
        "score_a_std": _std(score_a_values),
        "score_b_mean": mean(score_b_values),
        "score_b_std": _std(score_b_values),
        "score_delta_a_minus_b_mean": mean(delta_values),
        "score_delta_a_minus_b_std": _std(delta_values),
        "score_delta_b_minus_a_mean": mean(float(row["score_delta_b_minus_a"]) for row in aggregate_repeat_results),
        "score_delta_b_minus_a_std": _std([float(row["score_delta_b_minus_a"]) for row in aggregate_repeat_results]),
        "mean_invocations_a_mean": mean(invocations_a_values),
        "mean_invocations_a_std": _std(invocations_a_values),
        "mean_invocations_b_mean": mean(invocations_b_values),
        "mean_invocations_b_std": _std(invocations_b_values),
        "a_better_count_mean": mean(float(row["a_better_count"]) for row in aggregate_repeat_results),
        "b_better_count_mean": mean(float(row["b_better_count"]) for row in aggregate_repeat_results),
        "tied_count_mean": mean(float(row["tied_count"]) for row in aggregate_repeat_results),
        "both_positive_count_mean": mean(float(row["both_positive_count"]) for row in aggregate_repeat_results),
        "a_only_positive_count_mean": mean(float(row["a_only_positive_count"]) for row in aggregate_repeat_results),
        "b_only_positive_count_mean": mean(float(row["b_only_positive_count"]) for row in aggregate_repeat_results),
        "both_zero_count_mean": mean(float(row["both_zero_count"]) for row in aggregate_repeat_results),
        "disagreement_count_mean": mean(float(row["disagreement_count"]) for row in aggregate_repeat_results),
        "repeat_results": aggregate_repeat_results,
        "significance": significance,
    }


def _weighted_repeat_metric(compare_summaries: list[dict[str, Any]], repeat_rows: list[dict[str, Any]], *, key: str) -> float:
    sample_count = sum(int(summary["sample_count"]) for summary in compare_summaries)
    weighted_total = 0.0
    for summary, repeat_row in zip(compare_summaries, repeat_rows, strict=False):
        weighted_total += float(repeat_row[key]) * int(summary["sample_count"])
    return weighted_total / sample_count


def _shared_program_id(compare_summaries: list[dict[str, Any]]) -> str | None:
    program_ids = {str(summary["program_b_id"]) for summary in compare_summaries}
    if len(program_ids) == 1:
        return next(iter(program_ids))
    return None


def _compare_p_value(summary: dict[str, Any]) -> float | None:
    significance = summary.get("significance") or {}
    repeat_results = significance.get("repeat_results") or []
    if not repeat_results:
        return None
    return repeat_results[0].get("mcnemar_exact_p_value")


def _normalize_optimizer_names(optimizers: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(optimizers, str):
        raw_values = [value.strip() for value in optimizers.split(",")]
    else:
        raw_values = [str(value).strip() for value in optimizers]
    normalized_values: list[str] = []
    for value in raw_values:
        if not value:
            continue
        normalized = "mipro" if value.lower() == "miprov2" else value.lower()
        if normalized not in {"mipro", "gepa"}:
            raise ValueError(f"Unsupported DSPy optimizer: {value}")
        if normalized not in normalized_values:
            normalized_values.append(normalized)
    if not normalized_values:
        raise ValueError("At least one DSPy optimizer is required.")
    return normalized_values


def _slugify_group_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_") or "group"


def _render_family_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# TopoPrompt Family Benchmark Summary",
        "",
        f"- Benchmark: `{summary['benchmark_name']}`",
        f"- Grouping: `{summary['grouping']}`",
        f"- Groups: `{summary['group_count']}`",
        f"- Total source examples: `{summary['total_source_examples']}`",
        f"- Total eval examples: `{summary['total_eval_examples']}`",
        "",
        "## Aggregate",
        "",
        f"- Compiled vs direct: `{summary['aggregate']['compiled_score']:.4f}` vs `{summary['aggregate']['direct_score']:.4f}` (delta `{summary['aggregate']['compiled_minus_direct']:+.4f}`, p `{summary['aggregate']['compiled_vs_direct_mcnemar_p']}`)",
        f"- Compiled vs direct_x3: `{summary['aggregate']['compiled_score_vs_direct_x3']:.4f}` vs `{summary['aggregate']['direct_x3_score']:.4f}` (delta `{summary['aggregate']['compiled_minus_direct_x3']:+.4f}`, p `{summary['aggregate']['compiled_vs_direct_x3_mcnemar_p']}`)",
        "",
        "## Per Group",
        "",
        "| Group | Tasks | Eval | Compiled | Direct | Delta | p | Direct_x3 | Delta | p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group_name, payload in sorted(summary["families"].items()):
        task_names = ", ".join(payload["task_names"])
        lines.append(
            "| "
            f"{group_name} | {task_names} | {payload['eval_examples']} | "
            f"{payload['compiled_score']:.4f} | {payload['direct_score']:.4f} | {payload['compiled_minus_direct']:+.4f} | {payload['compiled_vs_direct_mcnemar_p']} | "
            f"{payload['direct_x3_score']:.4f} | {payload['compiled_minus_direct_x3']:+.4f} | {payload['compiled_vs_direct_x3_mcnemar_p']} |"
        )
    return "\n".join(lines) + "\n"


def _render_benchmark_dspy_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# TopoPrompt vs DSPy Benchmark Summary",
        "",
        f"- Benchmark: `{summary['benchmark_name']}`",
        f"- Task family: `{summary['task_family']}`",
        f"- Metric: `{summary['metric_name']}`",
        f"- Source examples: `{summary['source_examples']}`",
        f"- Train examples: `{summary['train_examples']}`",
        f"- Validation examples: `{summary['validation_examples']}`",
        f"- Eval examples: `{summary['eval_examples']}`",
        f"- TopoPrompt program: `{summary['topoprompt']['program_id']}`",
        f"- TopoPrompt validation score: `{summary['topoprompt']['validation_score']:.4f}`",
        "",
        "## Comparisons",
        "",
        "| Optimizer | DSPy Program | Student | TopoPrompt | DSPy | Delta | p |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for optimizer_name, payload in summary["comparisons"].items():
        lines.append(
            "| "
            f"{optimizer_name} | {payload['dspy_program_id']} | {payload['student_strategy']} | "
            f"{payload['topoprompt_score']:.4f} | {payload['dspy_score']:.4f} | "
            f"{payload['topoprompt_minus_dspy']:+.4f} | {payload['mcnemar_exact_p_value']} |"
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values)
