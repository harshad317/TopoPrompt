from __future__ import annotations

from pathlib import Path
from typing import Any

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.compiler.search import compile_task, evaluate_program_on_examples
from topoprompt.config import TopoPromptConfig
from topoprompt.eval.datasets import load_benchmark_examples, partition_examples
from topoprompt.eval.metrics import metric_for_name
from topoprompt.schemas import CompileArtifact, PromptProgram, TaskSpec


DEFAULT_BENCHMARK_TASKS = {
    "gsm8k": "Solve grade-school math word problems accurately.",
    "mmlu": "Answer multiple-choice knowledge questions correctly.",
    "bbh": "Solve diverse reasoning tasks with the correct final answer.",
    "ifeval": "Follow instructions precisely and satisfy all stated constraints.",
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

