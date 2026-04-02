from __future__ import annotations

import argparse
import json
from pathlib import Path

from topoprompt.backends.llm_client import FakeBackend
from topoprompt.backends.openai_backend import OpenAIBackend
from topoprompt.compiler.search import compile_task, evaluate_program_on_examples
from topoprompt.config import load_config
from topoprompt.eval.compare import compare_programs
from topoprompt.eval.datasets import load_examples_from_jsonl
from topoprompt.eval.metrics import metric_for_name
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

    args = parser.parse_args()
    config = load_config(args.config)
    backend = FakeBackend() if getattr(args, "fake_backend", False) else OpenAIBackend()

    if args.command == "compile":
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
        program = PromptProgram.model_validate_json(Path(args.program).read_text())
        examples = load_examples_from_jsonl(args.dataset)
        task_spec = _load_task_spec(args.task_spec, examples)
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
        program_a = PromptProgram.model_validate_json(Path(args.program_a).read_text())
        program_b = PromptProgram.model_validate_json(Path(args.program_b).read_text())
        examples = load_examples_from_jsonl(args.dataset)
        task_spec = _load_task_spec(args.task_spec, examples)
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


def _load_task_spec(task_spec_path: str | None, examples: list) -> TaskSpec:
    if task_spec_path:
        return TaskSpec.model_validate_json(Path(task_spec_path).read_text())
    return TaskSpec(
        task_id="evaluate_task",
        description="Evaluate a compiled TopoPrompt program.",
        input_schema={key: type(value).__name__ for key, value in examples[0].input.items()},
        output_schema={"type": "string"},
    )


if __name__ == "__main__":
    main()
