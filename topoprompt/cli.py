from __future__ import annotations

import argparse
import json
from pathlib import Path

from topoprompt.backends.llm_client import FakeBackend
from topoprompt.backends.openai_backend import OpenAIBackend
from topoprompt.compiler.search import compile_task, evaluate_program_on_examples
from topoprompt.config import load_config
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

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--program", required=True)
    evaluate_parser.add_argument("--dataset", required=True)
    evaluate_parser.add_argument("--task-spec", default=None)
    evaluate_parser.add_argument("--config", default=None)
    evaluate_parser.add_argument("--metric", default="exact_match")
    evaluate_parser.add_argument("--fake-backend", action="store_true")

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
        )
        print(json.dumps(artifact.metrics.model_dump(mode="json"), indent=2))
        return

    if args.command == "evaluate":
        program = PromptProgram.model_validate_json(Path(args.program).read_text())
        examples = load_examples_from_jsonl(args.dataset)
        if args.task_spec:
            task_spec = TaskSpec.model_validate_json(Path(args.task_spec).read_text())
        else:
            task_spec = TaskSpec(
                task_id="evaluate_task",
                description="Evaluate a compiled TopoPrompt program.",
                input_schema={key: type(value).__name__ for key, value in examples[0].input.items()},
                output_schema={"type": "string"},
            )
        result = evaluate_program_on_examples(
            program=program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_for_name(args.metric),
            backend=backend,
            config=config,
            phase="confirmation",
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

