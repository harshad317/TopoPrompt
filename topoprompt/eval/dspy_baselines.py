from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Literal

import orjson

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.compiler.search import evaluate_program_on_examples
from topoprompt.config import TopoPromptConfig
from topoprompt.eval.compare import (
    _build_compare_summary,
    _compare_repeat_results,
    _comparison_config,
    _render_compare_summary,
    _write_json,
    _write_jsonl,
)
from topoprompt.eval.metrics import MetricFn
from topoprompt.eval.significance import build_significance_summary, render_significance_summary
from topoprompt.progress import CompileProgressReporter
from topoprompt.schemas import Example, PromptProgram, TaskSpec


StudentStrategy = Literal["auto", "predict", "chain_of_thought"]
OptimizerName = Literal["mipro", "miprov2", "gepa"]


class _ProgramRef:
    def __init__(self, program_id: str) -> None:
        self.program_id = program_id


def compile_dspy_baseline(
    *,
    optimizer_name: OptimizerName,
    task_spec: TaskSpec,
    train_examples: list[Example],
    metric_fn: MetricFn,
    config: TopoPromptConfig,
    val_examples: list[Example] | None = None,
    student_strategy: StudentStrategy = "auto",
    model_name: str | None = None,
    reflection_model_name: str | None = None,
    optimizer_auto: Literal["light", "medium", "heavy"] | None = "light",
    num_threads: int | None = None,
    output_dir: str | Path | None = None,
    show_progress: bool = False,
    progress_verbosity: int = 1,
    progress_reporter: CompileProgressReporter | None = None,
) -> dict[str, Any]:
    dspy = _require_dspy()
    reporter = progress_reporter or CompileProgressReporter(enabled=show_progress, verbosity=progress_verbosity)
    resolved_strategy = _resolve_student_strategy(student_strategy, examples=train_examples, task_spec=task_spec)
    input_keys = _infer_input_keys(train_examples)
    signature = _build_signature(input_keys=input_keys, output_field="answer")
    dspy_trainset = _to_dspy_examples(dspy, train_examples, input_keys=input_keys)
    dspy_valset = _to_dspy_examples(dspy, val_examples or [], input_keys=input_keys) if val_examples is not None else None
    lm = _build_dspy_lm(dspy=dspy, model_name=model_name or config.model.name, config=config)
    reflection_lm = None
    if reflection_model_name or _normalize_optimizer_name(optimizer_name) == "gepa":
        reflection_lm = _build_dspy_lm(
            dspy=dspy,
            model_name=reflection_model_name or model_name or config.model.name,
            config=config,
        )

    reporter.rule(f"DSPy {optimizer_name.upper()} Compile", level=1, style="bold cyan")
    reporter.log(
        (
            f"optimizer={_normalize_optimizer_name(optimizer_name)} "
            f"student={resolved_strategy} "
            f"train={len(dspy_trainset)} "
            f"val={len(dspy_valset or [])}"
        ),
        level=1,
    )

    student = _build_student_program(
        dspy=dspy,
        signature=signature,
        strategy=resolved_strategy,
        input_keys=input_keys,
        output_field="answer",
        config=config,
    )
    optimizer = _build_optimizer(
        dspy=dspy,
        optimizer_name=optimizer_name,
        metric_fn=metric_fn,
        config=config,
        output_field="answer",
        optimizer_auto=optimizer_auto,
        task_lm=lm,
        reflection_lm=reflection_lm,
        num_threads=num_threads,
        log_dir=str(output_dir) if output_dir is not None else None,
    )

    started_at = perf_counter()
    with dspy.context(lm=lm):
        optimized = optimizer.compile(student, trainset=dspy_trainset, valset=dspy_valset)
    compile_seconds = perf_counter() - started_at

    program_id = f"dspy_{_normalize_optimizer_name(optimizer_name)}_{resolved_strategy}"
    optimized._topoprompt_program_id = program_id
    optimized._topoprompt_input_keys = input_keys
    optimized._topoprompt_output_field = "answer"
    optimized._topoprompt_student_strategy = resolved_strategy
    optimized._topoprompt_estimated_invocations = 1
    optimized._topoprompt_model_name = _normalize_dspy_model_name(model_name or config.model.name)

    program_state = extract_dspy_program_state(optimized)
    summary = {
        "optimizer_name": _normalize_optimizer_name(optimizer_name),
        "program_id": program_id,
        "student_strategy": resolved_strategy,
        "model_name": _normalize_dspy_model_name(model_name or config.model.name),
        "reflection_model_name": _normalize_dspy_model_name(reflection_model_name or model_name or config.model.name)
        if reflection_lm is not None
        else None,
        "input_keys": input_keys,
        "output_field": "answer",
        "signature": signature,
        "trainset_size": len(dspy_trainset),
        "valset_size": len(dspy_valset or []),
        "compile_seconds": compile_seconds,
        "optimizer_auto": optimizer_auto,
        "predictor_names": sorted(program_state["predictors"].keys()),
        "predictor_instructions": {
            name: state["instructions"] for name, state in program_state["predictors"].items()
        },
    }

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "program_state.json", program_state)
        _write_json(out_dir / "baseline_summary.json", summary)
        (out_dir / "baseline_summary.md").write_text(_render_dspy_baseline_summary(summary))
    reporter.log(
        (
            f"compiled {summary['program_id']} in {compile_seconds:.2f}s "
            f"with predictors={', '.join(summary['predictor_names']) or '-'}"
        ),
        level=1,
    )
    return {
        "program": optimized,
        "program_state": program_state,
        "summary": summary,
    }


def evaluate_dspy_program_on_examples(
    *,
    program: Any,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_fn: MetricFn,
    config: TopoPromptConfig,
    model_name: str | None = None,
    output_dir: str | Path | None = None,
    show_progress: bool = False,
    progress_verbosity: int = 1,
    progress_reporter: CompileProgressReporter | None = None,
) -> dict[str, Any]:
    dspy = _require_dspy()
    reporter = progress_reporter or CompileProgressReporter(enabled=show_progress, verbosity=progress_verbosity)
    input_keys = getattr(program, "_topoprompt_input_keys", None) or _infer_input_keys(examples)
    output_field = getattr(program, "_topoprompt_output_field", "answer")
    estimated_invocations = int(getattr(program, "_topoprompt_estimated_invocations", 1))
    dspy_examples = _to_dspy_examples(dspy, examples, input_keys=input_keys)
    resolved_model_name = model_name or getattr(program, "_topoprompt_model_name", None) or config.model.name
    lm = _build_dspy_lm(dspy=dspy, model_name=resolved_model_name, config=config)
    program_id = str(getattr(program, "_topoprompt_program_id", program.__class__.__name__))

    reporter.rule(f"Evaluate DSPy Program: {program_id}", level=1, style="bold blue")
    traces: list[dict[str, Any]] = []
    scores: list[float] = []
    iterable = reporter.track(
        dspy_examples,
        desc="DSPy evaluation",
        total=len(dspy_examples),
        leave=False,
        level=1,
    ) if reporter is not None else dspy_examples
    with dspy.context(lm=lm):
        for dspy_example in iterable:
            prediction = program(**dspy_example.inputs())
            final_output = _extract_prediction_value(prediction, output_field=output_field)
            source_example = _restore_topoprompt_example(dspy_example)
            score = float(metric_fn(final_output, source_example))
            traces.append(
                {
                    "example_id": source_example.example_id,
                    "final_output": final_output,
                    "correctness": score,
                    "total_invocations": estimated_invocations,
                    "parse_failures": 0,
                }
            )
            scores.append(score)

    mean_score = mean(scores) if scores else 0.0
    result = {
        "score": mean_score,
        "mean_invocations": float(estimated_invocations),
        "mean_tokens": 0.0,
        "parse_failure_rate": 0.0,
        "traces": traces,
    }
    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "evaluation_summary.json", {
            "program_id": program_id,
            "sample_count": len(examples),
            "score": mean_score,
            "mean_invocations": float(estimated_invocations),
        })
        _write_jsonl(out_dir / "evaluation_traces.jsonl", traces)
    reporter.log(
        (
            f"Evaluation {program_id} "
            f"score={mean_score:.4f} "
            f"calls={float(estimated_invocations):.2f} "
            f"coverage={len(traces)}/{len(examples)}"
        ),
        level=1,
    )
    return result


def compare_topoprompt_vs_dspy(
    *,
    topoprompt_program: PromptProgram,
    dspy_program: Any,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_fn: MetricFn,
    backend: LLMBackend,
    config: TopoPromptConfig,
    dspy_model_name: str | None = None,
    label_topoprompt: str = "topoprompt",
    label_dspy: str = "dspy",
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
    dspy_ref = _ProgramRef(str(getattr(dspy_program, "_topoprompt_program_id", dspy_program.__class__.__name__)))

    for repeat_index in range(repeats):
        if repeats > 1:
            reporter.rule(f"Compare Repeat {repeat_index + 1}", level=1, style="bold blue")
        topoprompt_config = _comparison_config(config=config, repeat_index=repeat_index + 1, side="a", output_dir=out_dir)
        dspy_config = _comparison_config(config=config, repeat_index=repeat_index + 1, side="b", output_dir=out_dir)
        reporter.log(f"Evaluating {label_topoprompt}: {topoprompt_program.program_id}", level=1)
        topoprompt_result = evaluate_program_on_examples(
            program=topoprompt_program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            backend=backend,
            config=topoprompt_config,
            phase=phase,
            show_progress=show_progress,
            progress_verbosity=progress_verbosity,
            progress_reporter=reporter,
        )
        reporter.log(f"Evaluating {label_dspy}: {dspy_ref.program_id}", level=1)
        dspy_result = evaluate_dspy_program_on_examples(
            program=dspy_program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            config=dspy_config,
            model_name=dspy_model_name or config.model.name,
            show_progress=show_progress,
            progress_verbosity=progress_verbosity,
            progress_reporter=reporter,
        )
        repeat_summary, disagreement_rows = _compare_repeat_results(
            repeat_index=repeat_index + 1,
            label_a=label_topoprompt,
            label_b=label_dspy,
            program_a=topoprompt_program,
            program_b=dspy_ref,
            examples=examples,
            result_a=topoprompt_result,
            result_b=dspy_result,
        )
        repeat_results.append(repeat_summary)
        disagreement_rows_by_repeat.append(disagreement_rows)

    summary = _build_compare_summary(
        label_a=label_topoprompt,
        label_b=label_dspy,
        program_a=topoprompt_program,
        program_b=dspy_ref,
        sample_count=len(examples),
        repeat_results=repeat_results,
    )
    significance = build_significance_summary(
        label_a=label_topoprompt,
        label_b=label_dspy,
        program_a_id=topoprompt_program.program_id,
        program_b_id=dspy_ref.program_id,
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


def compare_dspy_programs(
    *,
    program_a: Any,
    program_b: Any,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_fn: MetricFn,
    config: TopoPromptConfig,
    model_name_a: str | None = None,
    model_name_b: str | None = None,
    label_a: str = "dspy_a",
    label_b: str = "dspy_b",
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
    program_a_ref = _ProgramRef(str(getattr(program_a, "_topoprompt_program_id", program_a.__class__.__name__)))
    program_b_ref = _ProgramRef(str(getattr(program_b, "_topoprompt_program_id", program_b.__class__.__name__)))

    for repeat_index in range(repeats):
        if repeats > 1:
            reporter.rule(f"Compare Repeat {repeat_index + 1}", level=1, style="bold blue")
        config_a = _comparison_config(config=config, repeat_index=repeat_index + 1, side="a", output_dir=out_dir)
        config_b = _comparison_config(config=config, repeat_index=repeat_index + 1, side="b", output_dir=out_dir)
        reporter.log(f"Evaluating {label_a}: {program_a_ref.program_id}", level=1)
        result_a = evaluate_dspy_program_on_examples(
            program=program_a,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            config=config_a,
            model_name=model_name_a or config.model.name,
            show_progress=show_progress,
            progress_verbosity=progress_verbosity,
            progress_reporter=reporter,
        )
        reporter.log(f"Evaluating {label_b}: {program_b_ref.program_id}", level=1)
        result_b = evaluate_dspy_program_on_examples(
            program=program_b,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            config=config_b,
            model_name=model_name_b or config.model.name,
            show_progress=show_progress,
            progress_verbosity=progress_verbosity,
            progress_reporter=reporter,
        )
        repeat_summary, disagreement_rows = _compare_repeat_results(
            repeat_index=repeat_index + 1,
            label_a=label_a,
            label_b=label_b,
            program_a=program_a_ref,
            program_b=program_b_ref,
            examples=examples,
            result_a=result_a,
            result_b=result_b,
        )
        repeat_results.append(repeat_summary)
        disagreement_rows_by_repeat.append(disagreement_rows)

    summary = _build_compare_summary(
        label_a=label_a,
        label_b=label_b,
        program_a=program_a_ref,
        program_b=program_b_ref,
        sample_count=len(examples),
        repeat_results=repeat_results,
    )
    significance = build_significance_summary(
        label_a=label_a,
        label_b=label_b,
        program_a_id=program_a_ref.program_id,
        program_b_id=program_b_ref.program_id,
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


def load_dspy_program(
    *,
    state_path: str | Path,
    config: TopoPromptConfig,
    model_name: str | None = None,
) -> Any:
    dspy = _require_dspy()
    state = json.loads(Path(state_path).read_text())
    program = _build_student_program(
        dspy=dspy,
        signature=state["signature"],
        strategy=state["student_strategy"],
        input_keys=list(state["input_keys"]),
        output_field=state["output_field"],
        config=config,
    )
    apply_dspy_program_state(dspy=dspy, program=program, state=state)
    program._topoprompt_program_id = state["program_id"]
    program._topoprompt_input_keys = list(state["input_keys"])
    program._topoprompt_output_field = state["output_field"]
    program._topoprompt_student_strategy = state["student_strategy"]
    program._topoprompt_estimated_invocations = 1
    program._topoprompt_model_name = _normalize_dspy_model_name(model_name or state.get("model_name") or config.model.name)
    return program


def extract_dspy_program_state(program: Any) -> dict[str, Any]:
    state = {
        "program_id": str(getattr(program, "_topoprompt_program_id", program.__class__.__name__)),
        "student_strategy": getattr(program, "_topoprompt_student_strategy", "predict"),
        "input_keys": list(getattr(program, "_topoprompt_input_keys", [])),
        "output_field": str(getattr(program, "_topoprompt_output_field", "answer")),
        "signature": str(getattr(program, "_topoprompt_signature", "")),
        "model_name": getattr(program, "_topoprompt_model_name", None),
        "predictors": {},
    }
    for name, predictor in program.named_predictors():
        state["predictors"][name] = {
            "instructions": predictor.signature.instructions,
            "demos": [demo.toDict() for demo in getattr(predictor, "demos", [])],
        }
    return state


def apply_dspy_program_state(*, dspy: Any, program: Any, state: dict[str, Any]) -> None:
    predictors = {name: predictor for name, predictor in program.named_predictors()}
    input_keys = list(state.get("input_keys", []))
    for name, predictor_state in state.get("predictors", {}).items():
        predictor = predictors.get(name)
        if predictor is None:
            continue
        predictor.signature.instructions = predictor_state.get("instructions", predictor.signature.instructions)
        demos = [
            dspy.Example(**demo_payload).with_inputs(*input_keys)
            for demo_payload in predictor_state.get("demos", [])
        ]
        predictor.demos = demos


def _build_optimizer(
    *,
    dspy: Any,
    optimizer_name: OptimizerName,
    metric_fn: MetricFn,
    config: TopoPromptConfig,
    output_field: str,
    optimizer_auto: Literal["light", "medium", "heavy"] | None,
    task_lm: Any,
    reflection_lm: Any | None,
    num_threads: int | None,
    log_dir: str | None,
) -> Any:
    normalized = _normalize_optimizer_name(optimizer_name)
    if normalized == "mipro":
        return dspy.MIPROv2(
            metric=_make_dspy_metric(metric_fn=metric_fn, output_field=output_field),
            prompt_model=reflection_lm or task_lm,
            task_model=task_lm,
            auto=optimizer_auto,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
            num_threads=num_threads,
            seed=9,
            verbose=False,
            log_dir=log_dir,
        )
    return dspy.GEPA(
        metric=_make_gepa_metric(metric_fn=metric_fn, output_field=output_field),
        auto=optimizer_auto,
        reflection_lm=reflection_lm or task_lm,
        num_threads=num_threads,
        log_dir=log_dir,
        track_stats=True,
    )


def _build_student_program(
    *,
    dspy: Any,
    signature: str,
    strategy: Literal["predict", "chain_of_thought"],
    input_keys: list[str],
    output_field: str,
    config: TopoPromptConfig,
) -> Any:
    predictor_factory = dspy.Predict if strategy == "predict" else dspy.ChainOfThought

    class DSPyBaselineProgram(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.pred = predictor_factory(
                signature,
                temperature=config.model.temperature,
                max_tokens=config.model.max_output_tokens,
            )

        def forward(self, **kwargs):
            return self.pred(**kwargs)

    program = DSPyBaselineProgram()
    program._topoprompt_signature = signature
    program._topoprompt_input_keys = input_keys
    program._topoprompt_output_field = output_field
    program._topoprompt_student_strategy = strategy
    program._topoprompt_estimated_invocations = 1
    return program


def _make_dspy_metric(*, metric_fn: MetricFn, output_field: str):
    def metric(example, pred, trace=None):
        source_example = _restore_topoprompt_example(example)
        return float(metric_fn(_extract_prediction_value(pred, output_field=output_field), source_example))

    return metric


def _make_gepa_metric(*, metric_fn: MetricFn, output_field: str):
    dspy = _require_dspy()
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        source_example = _restore_topoprompt_example(gold)
        prediction_value = _extract_prediction_value(pred, output_field=output_field)
        score = float(metric_fn(prediction_value, source_example))
        feedback = _build_gepa_feedback(example=source_example, prediction_value=prediction_value, score=score)
        return ScoreWithFeedback(score=score, feedback=feedback)

    return metric


def _build_gepa_feedback(*, example: Example, prediction_value: Any, score: float) -> str:
    if score >= 1.0:
        return "Correct output. Keep the successful instruction behavior."
    expected = str(example.target) if example.target is not None else "all instruction constraints"
    got = str(prediction_value)
    if example.target is None:
        return f"Did not satisfy all required instruction constraints. Output was: {got}"
    return f"Incorrect output. Expected {expected} but got {got}. Improve the final answer."


def _build_dspy_lm(*, dspy: Any, model_name: str, config: TopoPromptConfig) -> Any:
    return dspy.LM(
        _normalize_dspy_model_name(model_name),
        temperature=config.model.temperature,
        max_tokens=config.model.max_output_tokens,
        cache=config.runtime.cache_enabled,
    )


def _normalize_dspy_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def _normalize_optimizer_name(name: OptimizerName | str) -> Literal["mipro", "gepa"]:
    normalized = str(name).strip().lower()
    if normalized in {"mipro", "miprov2"}:
        return "mipro"
    if normalized == "gepa":
        return "gepa"
    raise ValueError(f"Unsupported DSPy optimizer: {name}")


def _resolve_student_strategy(
    strategy: StudentStrategy,
    *,
    examples: list[Example],
    task_spec: TaskSpec,
) -> Literal["predict", "chain_of_thought"]:
    if strategy in {"predict", "chain_of_thought"}:
        return strategy
    description = f"{task_spec.description} {task_spec.task_family or ''}".lower()
    if any(keyword in description for keyword in ("math", "reason", "solve", "proof", "explain")):
        return "chain_of_thought"
    if any("choices" in example.input for example in examples[:3]):
        return "predict"
    return "predict"


def _infer_input_keys(examples: list[Example]) -> list[str]:
    if not examples:
        return ["question"]
    return list(examples[0].input.keys())


def _build_signature(*, input_keys: list[str], output_field: str) -> str:
    return ", ".join(input_keys) + f" -> {output_field}"


def _to_dspy_examples(dspy: Any, examples: list[Example], *, input_keys: list[str]) -> list[Any]:
    dspy_examples = []
    for example in examples:
        payload = {
            key: _serialize_input_value(key, example.input.get(key))
            for key in input_keys
        }
        payload["example_id"] = example.example_id
        payload["target"] = example.target
        payload["source_input_json"] = json.dumps(example.input, sort_keys=True)
        payload["source_metadata_json"] = json.dumps(example.metadata, sort_keys=True)
        dspy_examples.append(dspy.Example(**payload).with_inputs(*input_keys))
    return dspy_examples


def _restore_topoprompt_example(dspy_example: Any) -> Example:
    source_input = json.loads(getattr(dspy_example, "source_input_json", "{}"))
    source_metadata = json.loads(getattr(dspy_example, "source_metadata_json", "{}"))
    return Example(
        example_id=str(getattr(dspy_example, "example_id")),
        input=source_input,
        target=getattr(dspy_example, "target", None),
        metadata=source_metadata,
    )


def _serialize_input_value(key: str, value: Any) -> str:
    if key == "choices" and isinstance(value, list):
        normalized_lines = []
        for index, choice in enumerate(value):
            if isinstance(choice, dict):
                label = choice.get("label", chr(ord("A") + index))
                text = choice.get("text", "")
            else:
                label = chr(ord("A") + index)
                text = str(choice)
            normalized_lines.append(f"{label}. {text}")
        return "\n".join(normalized_lines)
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, sort_keys=True)


def _extract_prediction_value(prediction: Any, *, output_field: str) -> Any:
    if prediction is None:
        return None
    if hasattr(prediction, output_field):
        return getattr(prediction, output_field)
    if hasattr(prediction, "toDict"):
        payload = prediction.toDict()
        if output_field in payload:
            return payload[output_field]
        for key, value in payload.items():
            if key != "reasoning":
                return value
    if isinstance(prediction, dict):
        if output_field in prediction:
            return prediction[output_field]
        for key, value in prediction.items():
            if key != "reasoning":
                return value
    return str(prediction)


def _render_dspy_baseline_summary(summary: dict[str, Any]) -> str:
    return (
        "# DSPy Baseline Summary\n\n"
        f"- Program: `{summary['program_id']}`\n"
        f"- Optimizer: `{summary['optimizer_name']}`\n"
        f"- Student strategy: `{summary['student_strategy']}`\n"
        f"- Model: `{summary['model_name']}`\n"
        f"- Reflection model: `{summary['reflection_model_name']}`\n"
        f"- Signature: `{summary['signature']}`\n"
        f"- Train examples: `{summary['trainset_size']}`\n"
        f"- Validation examples: `{summary['valset_size']}`\n"
        f"- Compile time (s): `{summary['compile_seconds']:.2f}`\n"
    )


def _require_dspy():
    try:
        import dspy  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised in environments without the extra
        raise RuntimeError("DSPy baselines require the optional `dspy` extra. Run `uv sync --extra dspy`.") from exc
    return dspy
