from __future__ import annotations

import argparse
import json
from pathlib import Path

from topoprompt.backends.llm_client import FakeBackend
from topoprompt.backends.openai_backend import OpenAIBackend
from topoprompt.compiler.search import compile_task, evaluate_program_on_examples
from topoprompt.config import load_config
from topoprompt.eval.benchmark_runner import BenchmarkRunner
from topoprompt.eval.compare import compare_programs
from topoprompt.eval.dspy_baselines import (
    compare_topoprompt_vs_dspy,
    compile_dspy_baseline,
    evaluate_dspy_program_on_examples,
    load_dspy_program,
)
from topoprompt.eval.datasets import load_examples_from_jsonl
from topoprompt.eval.metrics import metric_for_name
from topoprompt.eval.significance import summarize_significance_from_compare_dir
from topoprompt.schemas import PromptProgram, TaskSpec


def main() -> None:
    parser = argparse.ArgumentParser(prog="topoprompt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--task-file", required=True)
    compile_parser.add_argument("--examples-file", required=True)
    compile_parser.add_argument("--config", default=None)
    compile_parser.add_argument("--output-dir", required=True)
    compile_parser.add_argument("--metric", default="exact_match")
    compile_parser.add_argument("--fake-backend", action="store_true")
    compile_parser.add_argument("--quiet", action="store_true")
    compile_parser.add_argument("-v", "--verbose", action="count", default=0)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--program", required=True)
    evaluate_parser.add_argument("--dataset", required=True)
    evaluate_parser.add_argument("--task-spec", default=None)
    evaluate_parser.add_argument("--config", default=None)
    evaluate_parser.add_argument("--metric", default="exact_match")
    evaluate_parser.add_argument("--fake-backend", action="store_true")
    evaluate_parser.add_argument("--quiet", action="store_true")
    evaluate_parser.add_argument("-v", "--verbose", action="count", default=0)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--program-a", required=True)
    compare_parser.add_argument("--program-b", required=True)
    compare_parser.add_argument("--dataset", required=True)
    compare_parser.add_argument("--task-spec", default=None)
    compare_parser.add_argument("--config", default=None)
    compare_parser.add_argument("--metric", default="exact_match")
    compare_parser.add_argument("--label-a", default="program_a")
    compare_parser.add_argument("--label-b", default="program_b")
    compare_parser.add_argument("--repeats", type=int, default=1)
    compare_parser.add_argument("--output-dir", default=None)
    compare_parser.add_argument("--fake-backend", action="store_true")
    compare_parser.add_argument("--quiet", action="store_true")
    compare_parser.add_argument("-v", "--verbose", action="count", default=0)

    significance_parser = subparsers.add_parser("significance")
    significance_parser.add_argument("--compare-dir", required=True)
    significance_parser.add_argument("--output-dir", default=None)
    significance_parser.add_argument("--confidence-level", type=float, default=0.95)
    significance_parser.add_argument("--bootstrap-samples", type=int, default=10000)
    significance_parser.add_argument("--bootstrap-seed", type=int, default=0)

    dspy_baseline_parser = subparsers.add_parser("dspy-baseline")
    dspy_baseline_parser.add_argument("--optimizer", required=True, choices=["mipro", "miprov2", "gepa"])
    dspy_baseline_parser.add_argument("--examples-file", required=True)
    dspy_baseline_parser.add_argument("--validation-file", default=None)
    dspy_baseline_parser.add_argument("--task-spec", default=None)
    dspy_baseline_parser.add_argument("--task-file", default=None)
    dspy_baseline_parser.add_argument("--config", default=None)
    dspy_baseline_parser.add_argument("--metric", default="exact_match")
    dspy_baseline_parser.add_argument("--output-dir", required=True)
    dspy_baseline_parser.add_argument("--student-strategy", choices=["auto", "predict", "chain_of_thought"], default="auto")
    dspy_baseline_parser.add_argument("--model", default=None)
    dspy_baseline_parser.add_argument("--reflection-model", default=None)
    dspy_baseline_parser.add_argument("--optimizer-auto", choices=["light", "medium", "heavy"], default="light")
    dspy_baseline_parser.add_argument("--quiet", action="store_true")
    dspy_baseline_parser.add_argument("-v", "--verbose", action="count", default=0)

    dspy_evaluate_parser = subparsers.add_parser("dspy-evaluate")
    dspy_evaluate_parser.add_argument("--state-file", required=True)
    dspy_evaluate_parser.add_argument("--dataset", required=True)
    dspy_evaluate_parser.add_argument("--task-spec", default=None)
    dspy_evaluate_parser.add_argument("--task-file", default=None)
    dspy_evaluate_parser.add_argument("--config", default=None)
    dspy_evaluate_parser.add_argument("--metric", default="exact_match")
    dspy_evaluate_parser.add_argument("--output-dir", default=None)
    dspy_evaluate_parser.add_argument("--model", default=None)
    dspy_evaluate_parser.add_argument("--quiet", action="store_true")
    dspy_evaluate_parser.add_argument("-v", "--verbose", action="count", default=0)

    dspy_compare_parser = subparsers.add_parser("dspy-compare")
    dspy_compare_parser.add_argument("--topoprompt-program", required=True)
    dspy_compare_parser.add_argument("--dspy-state-file", required=True)
    dspy_compare_parser.add_argument("--dataset", required=True)
    dspy_compare_parser.add_argument("--task-spec", default=None)
    dspy_compare_parser.add_argument("--task-file", default=None)
    dspy_compare_parser.add_argument("--config", default=None)
    dspy_compare_parser.add_argument("--metric", default="exact_match")
    dspy_compare_parser.add_argument("--label-topoprompt", default="topoprompt")
    dspy_compare_parser.add_argument("--label-dspy", default="dspy")
    dspy_compare_parser.add_argument("--repeats", type=int, default=1)
    dspy_compare_parser.add_argument("--output-dir", default=None)
    dspy_compare_parser.add_argument("--model", default=None)
    dspy_compare_parser.add_argument("--quiet", action="store_true")
    dspy_compare_parser.add_argument("-v", "--verbose", action="count", default=0)

    benchmark_family_parser = subparsers.add_parser("benchmark-family")
    benchmark_family_parser.add_argument("--benchmark", required=True, choices=["bbh"])
    benchmark_family_parser.add_argument("--examples-file", default=None)
    benchmark_family_parser.add_argument("--split", default="test")
    benchmark_family_parser.add_argument("--task-file", default=None)
    benchmark_family_parser.add_argument("--config", default=None)
    benchmark_family_parser.add_argument("--output-dir", required=True)
    benchmark_family_parser.add_argument("--grouping", choices=["family", "task"], default="family")
    benchmark_family_parser.add_argument("--groups", default=None)
    benchmark_family_parser.add_argument("--compile-budget", type=int, default=None)
    benchmark_family_parser.add_argument("--compare-repeats", type=int, default=1)
    benchmark_family_parser.add_argument("--fake-backend", action="store_true")
    benchmark_family_parser.add_argument("--quiet", action="store_true")
    benchmark_family_parser.add_argument("-v", "--verbose", action="count", default=0)

    benchmark_dspy_parser = subparsers.add_parser("benchmark-dspy")
    benchmark_dspy_parser.add_argument("--benchmark", required=True, choices=["gsm8k", "sst2", "mmlu", "bbh", "ifeval"])
    benchmark_dspy_parser.add_argument("--examples-file", default=None)
    benchmark_dspy_parser.add_argument("--split", default=None)
    benchmark_dspy_parser.add_argument("--task-file", default=None)
    benchmark_dspy_parser.add_argument("--config", default=None)
    benchmark_dspy_parser.add_argument("--output-dir", required=True)
    benchmark_dspy_parser.add_argument(
        "--optimizers",
        default="mipro,gepa",
        help="Comma-separated DSPy optimizers to compare against TopoPrompt. "
        "TopoPrompt is always included, and `topoprompt` is accepted as an optional explicit token.",
    )
    benchmark_dspy_parser.add_argument("--compile-budget", type=int, default=None)
    benchmark_dspy_parser.add_argument("--compare-repeats", type=int, default=1)
    benchmark_dspy_parser.add_argument("--student-strategy", choices=["auto", "predict", "chain_of_thought"], default="auto")
    benchmark_dspy_parser.add_argument("--model", default=None)
    benchmark_dspy_parser.add_argument("--reflection-model", default=None)
    benchmark_dspy_parser.add_argument("--optimizer-auto", choices=["light", "medium", "heavy"], default="light")
    benchmark_dspy_parser.add_argument("--quiet", action="store_true")
    benchmark_dspy_parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args()
    config = load_config(getattr(args, "config", None))

    if args.command == "compile":
        backend = FakeBackend() if getattr(args, "fake_backend", False) else OpenAIBackend()
        task_description = Path(args.task_file).read_text().strip()
        examples = load_examples_from_jsonl(args.examples_file)
        artifact = compile_task(
            task_description=task_description,
            examples=examples,
            metric=args.metric,
            backend=backend,
            config=config,
            output_dir=args.output_dir,
            show_progress=not args.quiet,
            progress_verbosity=1 + int(args.verbose or 0),
        )
        print(json.dumps(artifact.metrics.model_dump(mode="json"), indent=2))
        return

    if args.command == "evaluate":
        backend = FakeBackend() if getattr(args, "fake_backend", False) else OpenAIBackend()
        program = PromptProgram.model_validate_json(Path(args.program).read_text())
        examples = load_examples_from_jsonl(args.dataset)
        task_spec = _load_task_spec(args.task_spec, examples, task_file=None)
        result = evaluate_program_on_examples(
            program=program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_for_name(args.metric),
            backend=backend,
            config=config,
            phase="confirmation",
            show_progress=not args.quiet,
            progress_verbosity=1 + int(args.verbose or 0),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "compare":
        backend = FakeBackend() if getattr(args, "fake_backend", False) else OpenAIBackend()
        program_a = PromptProgram.model_validate_json(Path(args.program_a).read_text())
        program_b = PromptProgram.model_validate_json(Path(args.program_b).read_text())
        examples = load_examples_from_jsonl(args.dataset)
        task_spec = _load_task_spec(args.task_spec, examples, task_file=None)
        result = compare_programs(
            program_a=program_a,
            program_b=program_b,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_for_name(args.metric),
            backend=backend,
            config=config,
            label_a=args.label_a,
            label_b=args.label_b,
            repeats=args.repeats,
            output_dir=args.output_dir,
            show_progress=not args.quiet,
            progress_verbosity=1 + int(args.verbose or 0),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "dspy-baseline":
        if args.model:
            config.model.name = args.model
        examples = load_examples_from_jsonl(args.examples_file)
        validation_examples = load_examples_from_jsonl(args.validation_file) if args.validation_file else None
        task_spec = _load_task_spec(args.task_spec, examples, task_file=args.task_file)
        result = compile_dspy_baseline(
            optimizer_name=args.optimizer,
            task_spec=task_spec,
            train_examples=examples,
            val_examples=validation_examples,
            metric_fn=metric_for_name(args.metric),
            config=config,
            student_strategy=args.student_strategy,
            model_name=args.model or config.model.name,
            reflection_model_name=args.reflection_model,
            optimizer_auto=args.optimizer_auto,
            output_dir=args.output_dir,
            show_progress=not args.quiet,
            progress_verbosity=1 + int(args.verbose or 0),
        )
        print(json.dumps(result["summary"], indent=2))
        return

    if args.command == "dspy-evaluate":
        if args.model:
            config.model.name = args.model
        examples = load_examples_from_jsonl(args.dataset)
        task_spec = _load_task_spec(args.task_spec, examples, task_file=args.task_file)
        program = load_dspy_program(
            state_path=args.state_file,
            config=config,
            model_name=args.model or config.model.name,
        )
        result = evaluate_dspy_program_on_examples(
            program=program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_for_name(args.metric),
            config=config,
            model_name=args.model or config.model.name,
            output_dir=args.output_dir,
            show_progress=not args.quiet,
            progress_verbosity=1 + int(args.verbose or 0),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "dspy-compare":
        if args.model:
            config.model.name = args.model
        backend = OpenAIBackend()
        examples = load_examples_from_jsonl(args.dataset)
        task_spec = _load_task_spec(args.task_spec, examples, task_file=args.task_file)
        topoprompt_program = PromptProgram.model_validate_json(Path(args.topoprompt_program).read_text())
        dspy_program = load_dspy_program(
            state_path=args.dspy_state_file,
            config=config,
            model_name=args.model or config.model.name,
        )
        result = compare_topoprompt_vs_dspy(
            topoprompt_program=topoprompt_program,
            dspy_program=dspy_program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_for_name(args.metric),
            backend=backend,
            config=config,
            dspy_model_name=args.model or config.model.name,
            label_topoprompt=args.label_topoprompt,
            label_dspy=args.label_dspy,
            repeats=args.repeats,
            output_dir=args.output_dir,
            show_progress=not args.quiet,
            progress_verbosity=1 + int(args.verbose or 0),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "significance":
        result = summarize_significance_from_compare_dir(
            args.compare_dir,
            output_dir=args.output_dir,
            confidence_level=args.confidence_level,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "benchmark-family":
        backend = FakeBackend() if getattr(args, "fake_backend", False) else OpenAIBackend()
        runner = BenchmarkRunner(config=config, backend=backend)
        task_description = Path(args.task_file).read_text().strip() if args.task_file else None
        group_names = [value.strip() for value in (args.groups or "").split(",") if value.strip()] or None
        result = runner.compile_and_compare_by_family(
            benchmark_name=args.benchmark,
            examples_path=args.examples_file,
            split=args.split,
            task_description=task_description,
            output_dir=args.output_dir,
            grouping=args.grouping,
            include_groups=group_names,
            compare_repeats=args.compare_repeats,
            compile_budget=args.compile_budget,
            show_progress=not args.quiet,
            progress_verbosity=1 + int(args.verbose or 0),
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "benchmark-dspy":
        if args.model:
            config.model.name = args.model
        backend = OpenAIBackend()
        runner = BenchmarkRunner(config=config, backend=backend)
        task_description = Path(args.task_file).read_text().strip() if args.task_file else None
        result = runner.compile_and_compare_with_dspy(
            benchmark_name=args.benchmark,
            optimizers=args.optimizers,
            examples_path=args.examples_file,
            split=args.split,
            task_description=task_description,
            output_dir=args.output_dir,
            compare_repeats=args.compare_repeats,
            compile_budget=args.compile_budget,
            student_strategy=args.student_strategy,
            model_name=args.model or config.model.name,
            reflection_model_name=args.reflection_model,
            optimizer_auto=args.optimizer_auto,
            show_progress=not args.quiet,
            progress_verbosity=1 + int(args.verbose or 0),
        )
        print(json.dumps(result, indent=2))
        return


def _load_task_spec(task_spec_path: str | None, examples: list, *, task_file: str | None) -> TaskSpec:
    if task_spec_path:
        return TaskSpec.model_validate_json(Path(task_spec_path).read_text())
    description = Path(task_file).read_text().strip() if task_file else "Evaluate a compiled TopoPrompt program."
    return TaskSpec(
        task_id="evaluate_task",
        description=description,
        input_schema={key: type(value).__name__ for key, value in examples[0].input.items()},
        output_schema={"type": "string"},
    )


if __name__ == "__main__":
    main()
