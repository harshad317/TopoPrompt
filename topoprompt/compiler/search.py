from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from topoprompt.artifacts import write_compile_artifact
from topoprompt.backends.dspy_backend import compile_to_dspy
from topoprompt.backends.llm_client import FakeBackend, LLMBackend
from topoprompt.backends.openai_backend import OpenAIBackend
from topoprompt.compiler.analyzer import analyze_task
from topoprompt.compiler.edits import apply_edit, generate_heuristic_edits
from topoprompt.compiler.objective import description_length, search_score
from topoprompt.compiler.seeds import SEED_LIBRARY, instantiate_seed_programs
from topoprompt.compiler.selector import choose_smallest_effective
from topoprompt.compiler.validator import ProgramValidationError, validate_program
from topoprompt.config import TopoPromptConfig, load_config
from topoprompt.eval.budget import BudgetLedger
from topoprompt.eval.datasets import DatasetPartitions, partition_examples
from topoprompt.eval.metrics import MetricFn, canonical_metric_name, metric_for_name
from topoprompt.ir import family_signature, topology_fingerprint
from topoprompt.progress import CompileProgressReporter
from topoprompt.runtime.executor import BudgetExhausted, ProgramExecutor
from topoprompt.runtime.trace import aggregate_route_metrics
from topoprompt.schemas import (
    CandidateArchiveRecord,
    CandidateEdit,
    CandidateEvaluation,
    CompileArtifact,
    CompileMetrics,
    Example,
    PromptProgram,
    RouteDiagnostic,
    TaskAnalysis,
    TaskSpec,
)
from topoprompt.transfer.features import extract_transfer_features
from topoprompt.transfer.store import TraceStore


def compile_task(
    *,
    task_description: str,
    examples: list[Example],
    model: str | None = None,
    compile_budget: int | None = None,
    metric: str | MetricFn | None = None,
    config: TopoPromptConfig | None = None,
    config_path: str | Path | None = None,
    backend: LLMBackend | None = None,
    output_dir: str | Path | None = None,
    task_id: str | None = None,
    search_examples: list[Example] | None = None,
    validation_examples: list[Example] | None = None,
    fewshot_examples: list[Example] | None = None,
    show_progress: bool = False,
    progress_verbosity: int = 1,
    progress_reporter: CompileProgressReporter | None = None,
) -> CompileArtifact:
    if config is None:
        overrides: dict[str, Any] = {}
        if model is not None:
            overrides.setdefault("model", {})["name"] = model
        if compile_budget is not None:
            overrides.setdefault("compile", {})["total_budget_calls"] = compile_budget
        config = load_config(config_path, overrides=overrides)
    else:
        config = TopoPromptConfig.model_validate(config.model_dump())
        if model is not None:
            config.model.name = model
        if compile_budget is not None:
            config.compile.rebalance_phase_budgets(compile_budget)
    if compile_budget is not None and config.compile.phase_budget_total() != compile_budget:
        config.compile.rebalance_phase_budgets(compile_budget)

    backend = backend or OpenAIBackend()
    reporter = progress_reporter or CompileProgressReporter(enabled=show_progress, verbosity=progress_verbosity)
    metric_name, metric_fn = _resolve_metric(metric)
    partitions = _resolve_partitions(
        examples=examples,
        config=config,
        search_examples=search_examples,
        validation_examples=validation_examples,
        fewshot_examples=fewshot_examples,
    )
    task_spec = _infer_task_spec(task_description=task_description, examples=examples, task_id=task_id)
    budget = BudgetLedger.from_compile_config(config.compile)
    trace_store = TraceStore(Path(output_dir) / "transfer_trace_store.jsonl" if output_dir else None)
    reporter.rule(f"TopoPrompt Compile: {task_spec.task_id}")
    reporter.log(
        (
            f"metric={metric_name} examples={len(examples)} "
            f"fewshot={len(partitions.fewshot_examples)} "
            f"search={len(partitions.search_examples)} "
            f"validation={len(partitions.validation_examples)}"
        ),
        style="cyan",
    )
    reporter.log_budget(spent=budget.spent_total(), planned=budget.planned_total(), phase="compile")

    analysis = _run_analysis(
        task_spec=task_spec,
        examples=partitions.search_examples,
        metric_name=metric_name,
        backend=backend,
        config=config,
        budget=budget,
        reporter=reporter,
    )
    reporter.print_analysis(analysis)
    seed_names = (analysis.initial_seed_templates or SEED_LIBRARY)[:5]
    reporter.log(f"Selected seed templates: {', '.join(seed_names)}")
    seed_programs = instantiate_seed_programs(
        task_spec=task_spec,
        analysis=analysis,
        include_direct_baseline=config.compile.always_include_direct_seed,
        seed_names=seed_names,
    )
    seed_programs = [program for program in seed_programs if _validate_candidate(program, config)]
    reporter.log(f"Valid instantiated seeds: {len(seed_programs)}")

    seed_evals = []
    for program in reporter.track(
        seed_programs,
        desc="Seed evaluation",
        total=len(seed_programs),
        leave=False,
        level=1,
    ):
        evaluation = _evaluate_candidate(
            program=program,
            task_spec=task_spec,
            examples=partitions.search_examples,
            metric_fn=metric_fn,
            backend=backend,
            config=config,
            budget=budget,
            phase="seed",
            stage="screening",
            max_examples=config.compile.screening_examples,
            fewshot_examples=partitions.fewshot_examples,
            reporter=reporter,
        )
        if evaluation is not None:
            seed_evals.append(evaluation)
    seed_evals = [evaluation for evaluation in seed_evals if evaluation is not None]

    direct_baseline = next((evaluation for evaluation in seed_evals if evaluation.program.program_id == "direct_finalize"), None)
    best_non_direct = max((evaluation.score for evaluation in seed_evals if evaluation.program.program_id != "direct_finalize"), default=0.0)
    if direct_baseline is not None and best_non_direct < direct_baseline.score - config.compile.reseed_margin:
        reporter.log(
            "Analyzer-ranked seeds underperformed the direct baseline. Re-seeding from the full library.",
        )
        seed_programs = instantiate_seed_programs(
            task_spec=task_spec,
            analysis=analysis,
            include_direct_baseline=config.compile.always_include_direct_seed,
            seed_names=SEED_LIBRARY,
        )
        seed_programs = [program for program in seed_programs if _validate_candidate(program, config)]
        seed_evals = []
        for program in reporter.track(
            seed_programs,
            desc="Re-seed evaluation",
            total=len(seed_programs),
            leave=False,
            level=1,
        ):
            evaluation = _evaluate_candidate(
                program=program,
                task_spec=task_spec,
                examples=partitions.search_examples,
                metric_fn=metric_fn,
                backend=backend,
                config=config,
                budget=budget,
                phase="seed",
                stage="screening",
                max_examples=config.compile.screening_examples,
                fewshot_examples=partitions.fewshot_examples,
                reporter=reporter,
            )
            if evaluation is not None:
                seed_evals.append(evaluation)
        seed_evals = [evaluation for evaluation in seed_evals if evaluation is not None]

    archive_records = [_archive_record(candidate, round_index=0) for candidate in seed_evals]
    compile_traces = [trace for candidate in seed_evals for trace in candidate.traces]
    beam = _select_diverse_beam(seed_evals, width=config.compile.beam_width, min_families=config.compile.min_structural_families)
    beam_family_count_by_round = [len({candidate.family_signature for candidate in beam})] if beam else [0]
    best_beam_score = max((candidate.search_score for candidate in beam), default=0.0)
    stale_rounds = 0

    for round_index in range(1, config.compile.max_rounds + 1):
        reporter.rule(f"Search Round {round_index}", level=1, style="bold blue")
        reporter.log(
            f"beam_size={len(beam)} beam_families={len({candidate.family_signature for candidate in beam})}",
        )
        reporter.log_budget(spent=budget.spent_total(), planned=budget.planned_total(), phase="compile")
        proposals: list[tuple[PromptProgram, str, str]] = []
        for parent in reporter.track(
            beam,
            desc=f"Round {round_index}: propose edits",
            total=len(beam),
            leave=False,
            level=1,
        ):
            edits = generate_heuristic_edits(program=parent.program, analysis=analysis, config=config)
            if config.compile.llm_edit_proposals_enabled:
                edits.extend(
                    _llm_guided_edit_proposals(
                        parent=parent,
                        analysis=analysis,
                        backend=backend,
                        config=config,
                        budget=budget,
                    )
                )
            for edit in _dedupe_edits(edits):
                try:
                    candidate = apply_edit(
                        program=parent.program,
                        edit=edit,
                        analysis=analysis,
                        fewshot_pool=partitions.fewshot_examples,
                    )
                    validate_program(candidate, config.program)
                    proposals.append((candidate, parent.program.program_id, edit.model_dump_json()))
                except (ProgramValidationError, ValueError, KeyError):
                    continue

        proposals = _dedupe_proposals(proposals)
        reporter.log(f"Round {round_index}: {len(proposals)} unique valid proposals")
        if not proposals:
            reporter.log(f"Round {round_index}: no proposals survived validation; stopping.")
            break

        scored = _evaluate_candidates_multi_fidelity(
            proposals=proposals,
            task_spec=task_spec,
            examples=partitions.search_examples,
            metric_fn=metric_fn,
            backend=backend,
            config=config,
            budget=budget,
            fewshot_examples=partitions.fewshot_examples,
            reporter=reporter,
        )
        if not scored:
            reporter.log(f"Round {round_index}: no candidates survived evaluation; stopping.")
            break

        for candidate in scored:
            compile_traces.extend(candidate.traces)
            trace_store.append(extract_transfer_features(task_spec=task_spec, program=candidate.program, candidate=candidate))
            archive_records.append(_archive_record(candidate, round_index=round_index))

        beam = _select_diverse_beam(scored, width=config.compile.beam_width, min_families=config.compile.min_structural_families)
        beam_family_count_by_round.append(len({candidate.family_signature for candidate in beam}))
        if not beam:
            reporter.log(f"Round {round_index}: beam collapsed to empty; stopping.")
            break

        leader = beam[0]
        reporter.log_candidate(leader, prefix=f"Round {round_index} leader ", level=1)

        current_best = max(candidate.search_score for candidate in beam)
        if current_best < best_beam_score + config.compile.early_stop_min_improvement:
            stale_rounds += 1
        else:
            best_beam_score = current_best
            stale_rounds = 0
        if stale_rounds >= config.compile.early_stop_patience_rounds:
            reporter.log(
                f"Round {round_index}: early stop patience reached with no meaningful improvement.",
            )
            break
        if budget.remaining("confirmation") <= 0 and budget.remaining("screening") <= 0 and budget.remaining("narrowing") <= 0:
            reporter.log("Search budgets exhausted; stopping.")
            break

    final_confirmation_candidates = _select_affordable_confirmation_candidates(
        candidates=beam[: config.compile.confirm_top_k],
        validation_examples=partitions.validation_examples,
        budget=budget,
        config=config,
        reporter=reporter,
    )
    finalists = _confirm_candidates(
        candidates=final_confirmation_candidates,
        task_spec=task_spec,
        validation_examples=partitions.validation_examples,
        metric_fn=metric_fn,
        backend=backend,
        config=config,
        budget=budget,
        reporter=reporter,
    )
    compile_traces.extend(trace for candidate in finalists for trace in candidate.traces)
    selection_candidates, used_search_fallback = _select_final_selection_pool(
        finalists=finalists,
        beam=beam,
        seed_evals=seed_evals,
        config=config,
    )
    best_candidate, smallest_effective, epsilon, _effective = choose_smallest_effective(
        selection_candidates,
        objective_config=config.objective,
    )
    final_candidate = _select_final_candidate(
        best_candidate=best_candidate,
        smallest_effective=smallest_effective,
        config=config,
    )
    route_accuracy, route_regret = aggregate_route_metrics(final_candidate.traces)
    reporter.rule("Final Selection", level=1, style="bold green")
    if used_search_fallback:
        reporter.log(
            "No candidates completed final confirmation. Falling back to the strongest search-stage candidates.",
        )
    reporter.log_candidate(best_candidate, prefix="Best fallback " if used_search_fallback else "Best confirmed ", level=1)
    reporter.log_candidate(
        smallest_effective,
        prefix="Smallest fallback " if used_search_fallback else "Smallest effective ",
        level=1,
    )
    reporter.log_candidate(final_candidate, prefix="Final exported ", level=1)
    reporter.log(f"epsilon={epsilon:.4f} route_accuracy={route_accuracy} route_regret={route_regret}")

    metrics = CompileMetrics(
        best_program_id=best_candidate.program.program_id,
        best_validation_score=best_candidate.score,
        smallest_effective_program_id=smallest_effective.program.program_id,
        smallest_effective_score=smallest_effective.score,
        final_program_id=final_candidate.program.program_id,
        final_program_score=final_candidate.score,
        final_program_policy=config.compile.final_program_policy,
        epsilon=epsilon,
        planned_budget_calls=budget.planned_total(),
        spent_budget_calls=budget.spent_total(),
        planned_budget_by_phase=[
            phase.model_copy(update={"spent_calls": 0})
            for phase in budget.snapshot()
        ],
        spent_budget_by_phase=budget.snapshot(),
        winning_topology_family=final_candidate.family_signature,
        beam_family_count_by_round=beam_family_count_by_round,
        parser_failure_rate=final_candidate.parse_failure_rate,
        route_accuracy=route_accuracy,
        route_regret=route_regret,
    )
    dspy_program = compile_to_dspy(program=final_candidate.program, task_spec=task_spec, config=config, backend=backend)
    artifact = CompileArtifact(
        task_spec=task_spec,
        best_program_ir=best_candidate.program,
        smallest_effective_program_ir=smallest_effective.program,
        program_ir=final_candidate.program,
        python_program=final_candidate.program,
        dspy_program=dspy_program,
        seed_programs=seed_programs,
        candidate_archive=archive_records,
        compile_trace=compile_traces,
        metrics=metrics,
        config=config.model_dump(mode="json"),
        output_dir=str(output_dir) if output_dir else None,
    )
    if output_dir is not None:
        write_compile_artifact(artifact, output_dir)
        trace_store.flush()
        reporter.log(f"Artifacts written to {output_dir}")
    return artifact


def evaluate_program_on_examples(
    *,
    program: PromptProgram,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_fn: MetricFn,
    backend: LLMBackend,
    config: TopoPromptConfig,
    phase: str,
    show_progress: bool = False,
    progress_verbosity: int = 1,
    progress_reporter: CompileProgressReporter | None = None,
) -> dict[str, Any]:
    budget: BudgetLedger | None = None
    reporter = progress_reporter or CompileProgressReporter(enabled=show_progress, verbosity=progress_verbosity)
    reporter.rule(f"Evaluate Program: {program.program_id}", level=1, style="bold blue")
    evaluation = _evaluate_candidate(
        program=program,
        task_spec=task_spec,
        examples=examples,
        metric_fn=metric_fn,
        backend=backend,
        config=config,
        budget=budget,
        phase=phase,
        stage=phase,
        max_examples=len(examples),
        fewshot_examples=[],
        reporter=reporter,
    )
    assert evaluation is not None
    reporter.log_candidate(evaluation, prefix="Evaluation ", level=1)
    return {
        "score": evaluation.score,
        "mean_invocations": evaluation.mean_invocations,
        "mean_tokens": evaluation.mean_tokens,
        "parse_failure_rate": evaluation.parse_failure_rate,
        "traces": [trace.model_dump(mode="json") for trace in evaluation.traces],
    }


def _resolve_metric(metric: str | MetricFn | None) -> tuple[str, MetricFn]:
    if callable(metric):
        return getattr(metric, "__name__", "custom_metric"), metric
    metric_name = canonical_metric_name(metric)
    return metric_name, metric_for_name(metric_name)


def _resolve_partitions(
    *,
    examples: list[Example],
    config: TopoPromptConfig,
    search_examples: list[Example] | None,
    validation_examples: list[Example] | None,
    fewshot_examples: list[Example] | None,
) -> DatasetPartitions:
    if search_examples is not None and validation_examples is not None:
        fs = fewshot_examples or search_examples[: min(len(search_examples), config.data.fewshot_pool_max_examples)]
        return DatasetPartitions(
            compile_examples=(fewshot_examples or []) + search_examples,
            fewshot_examples=fs,
            search_examples=search_examples,
            validation_examples=validation_examples,
            test_examples=[],
        )
    return partition_examples(examples, data_config=config.data, create_test_split=False)


def _infer_task_spec(*, task_description: str, examples: list[Example], task_id: str | None) -> TaskSpec:
    python_to_jsonschema = {
        "dict": "object",
        "list": "array",
        "str": "string",
        "int": "number",
        "float": "number",
        "bool": "boolean",
    }
    input_schema = {key: type(value).__name__ for key, value in (examples[0].input.items() if examples else [])}
    output_type = (
        python_to_jsonschema.get(type(examples[0].target).__name__, "string")
        if examples and examples[0].target is not None
        else "string"
    )
    output_schema = {"type": output_type}
    return TaskSpec(
        task_id=task_id or "compiled_task",
        description=task_description,
        input_schema=input_schema,
        output_schema=output_schema,
        task_family=None,
        metadata={},
    )


def _run_analysis(
    *,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_name: str,
    backend: LLMBackend,
    config: TopoPromptConfig,
    budget: BudgetLedger,
    reporter: CompileProgressReporter | None = None,
) -> TaskAnalysis:
    if not budget.spend("analyzer", 1, allow_reserve=True):
        if reporter is not None:
            reporter.log("Analyzer budget unavailable. Falling back to direct seed.")
        return TaskAnalysis(initial_seed_templates=["direct_finalize"])
    if reporter is not None:
        reporter.log(f"Running task analyzer on {min(len(examples), config.data.representative_examples_for_analysis)} representative examples.")
    analysis = analyze_task(task_spec=task_spec, examples=examples, metric_name=metric_name, backend=backend, config=config)
    task_spec.task_family = analysis.task_family
    return analysis


def _evaluate_candidate(
    *,
    program: PromptProgram,
    task_spec: TaskSpec,
    examples: list[Example],
    metric_fn: MetricFn,
    backend: LLMBackend,
    config: TopoPromptConfig,
    budget: BudgetLedger | None,
    phase: str,
    stage: str,
    max_examples: int,
    fewshot_examples: list[Example],
    parent_id: str | None = None,
    edit_applied: str | None = None,
    reporter: CompileProgressReporter | None = None,
) -> CandidateEvaluation | None:
    executor = ProgramExecutor(backend=backend, config=config, budget_ledger=budget, reporter=reporter)
    traces = []
    scores = []
    scoped_examples = examples[:max_examples]
    for example in scoped_examples:
        try:
            result = executor.run_program(
                program=program,
                task_spec=task_spec,
                example_id=example.example_id,
                task_input=example.input,
                phase=phase,
            )
        except BudgetExhausted:
            break
        trace = result.trace
        trace.correctness = metric_fn(trace.final_output, example)
        traces.append(trace)
        scores.append(trace.correctness)
        if reporter is not None:
            reporter.log_example_result(
                program_id=program.program_id,
                example_id=example.example_id,
                score=trace.correctness,
                invocations=trace.total_invocations,
                parse_failures=trace.parse_failures,
            )
    fully_evaluated = len(traces) == len(scoped_examples)
    if not traces:
        if reporter is not None:
            reporter.log(f"{program.program_id} produced no traces during {stage}.")
        return None
    if reporter is not None and not fully_evaluated:
        reporter.log(
            f"{program.program_id} completed {len(traces)}/{len(scoped_examples)} examples during {stage}; budget exhausted.",
            level=1,
        )

    route_diagnostics = _induce_route_diagnostics(
        program=program,
        task_spec=task_spec,
        examples=examples[: min(3, len(traces))],
        base_traces=traces,
        metric_fn=metric_fn,
        backend=backend,
        config=config,
        budget=budget,
    )
    for diagnostic in route_diagnostics:
        for trace in traces:
            if trace.example_id == diagnostic.example_id:
                trace.route_diagnostics.append(diagnostic)

    complexity = description_length(program, config.program)
    mean_invocations = mean(trace.total_invocations for trace in traces)
    mean_tokens = mean(trace.total_tokens for trace in traces)
    coverage_ratio = len(traces) / max(len(scoped_examples), 1)
    parse_failure_rate = sum(trace.parse_failures for trace in traces) / max(
        1, sum(len(trace.node_traces) for trace in traces)
    )
    perf = mean(scores)
    candidate = CandidateEvaluation(
        program=program,
        topology_fingerprint=topology_fingerprint(program),
        family_signature=family_signature(program),
        stage=stage,
        parent_id=parent_id,
        edit_applied=edit_applied,
        example_scores=scores,
        score=perf,
        search_score=search_score(
            perf=perf,
            mean_invocations=mean_invocations,
            complexity=complexity,
            parse_failure_rate=parse_failure_rate,
            coverage_ratio=coverage_ratio,
            objective_config=config.objective,
            program_config=config.program,
        ),
        mean_invocations=mean_invocations,
        mean_tokens=mean_tokens,
        complexity=complexity,
        parse_failure_rate=parse_failure_rate,
        traces=traces,
        route_diagnostics=route_diagnostics,
        metadata={
            "fewshot_pool_size": len(fewshot_examples),
            "examples_evaluated": len(traces),
            "target_examples": len(scoped_examples),
            "fully_evaluated": fully_evaluated,
            "coverage_ratio": coverage_ratio,
        },
    )
    if reporter is not None:
        reporter.log_candidate(candidate, prefix=f"{stage} ", level=2)
    return candidate


def _evaluate_candidates_multi_fidelity(
    *,
    proposals: list[tuple[PromptProgram, str, str]],
    task_spec: TaskSpec,
    examples: list[Example],
    metric_fn: MetricFn,
    backend: LLMBackend,
    config: TopoPromptConfig,
    budget: BudgetLedger,
    fewshot_examples: list[Example],
    reporter: CompileProgressReporter | None = None,
) -> list[CandidateEvaluation]:
    screened: list[CandidateEvaluation] = []
    proposal_iterable = reporter.track(
        proposals,
        desc="Screening candidates",
        total=len(proposals),
        leave=False,
        level=1,
    ) if reporter is not None else proposals
    for index, (program, parent_id, edit_applied) in enumerate(proposal_iterable):
        screening_examples = _budgeted_stage_example_cap(
            program=program,
            phase="screening",
            configured_examples=config.compile.screening_examples,
            budget=budget,
            remaining_candidates=len(proposals) - index,
            config=config,
        )
        if screening_examples <= 0:
            if reporter is not None:
                reporter.log("Screening budget is exhausted before all candidates could be scored.", level=1)
            break
        evaluation = _evaluate_candidate(
            program=program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            backend=backend,
            config=config,
            budget=budget,
            phase="screening",
            stage="screening",
            max_examples=screening_examples,
            fewshot_examples=fewshot_examples,
            parent_id=parent_id,
            edit_applied=edit_applied,
            reporter=reporter,
        )
        if evaluation is not None:
            screened.append(evaluation)
    if not screened:
        return []
    best_screen = max(candidate.search_score for candidate in screened)
    promising = [
        candidate
        for candidate in screened
        if candidate.search_score >= best_screen - 0.15 and candidate.parse_failure_rate <= 0.5
    ]
    promising = sorted(promising, key=lambda item: item.search_score, reverse=True)
    narrowed: list[CandidateEvaluation] = []
    limit = max(config.compile.beam_width * 2, config.compile.min_structural_families)
    narrowing_pool = promising[:limit]
    narrowing_iterable = reporter.track(
        narrowing_pool,
        desc="Narrowing candidates",
        total=len(narrowing_pool),
        leave=False,
        level=1,
    ) if reporter is not None else narrowing_pool
    for index, candidate in enumerate(narrowing_iterable):
        narrowing_examples = _budgeted_stage_example_cap(
            program=candidate.program,
            phase="narrowing",
            configured_examples=config.compile.narrowing_examples,
            budget=budget,
            remaining_candidates=len(narrowing_pool) - index,
            config=config,
        )
        if narrowing_examples <= 0:
            if reporter is not None:
                reporter.log("Narrowing budget is exhausted before all promising candidates could be rescored.", level=1)
            break
        narrowed_eval = _evaluate_candidate(
            program=candidate.program,
            task_spec=task_spec,
            examples=examples,
            metric_fn=metric_fn,
            backend=backend,
            config=config,
            budget=budget,
            phase="narrowing",
            stage="narrowing",
            max_examples=narrowing_examples,
            fewshot_examples=fewshot_examples,
            parent_id=candidate.parent_id,
            edit_applied=candidate.edit_applied,
            reporter=reporter,
        )
        if narrowed_eval is not None:
            narrowed.append(narrowed_eval)
    return narrowed or screened


def _confirm_candidates(
    *,
    candidates: list[CandidateEvaluation],
    task_spec: TaskSpec,
    validation_examples: list[Example],
    metric_fn: MetricFn,
    backend: LLMBackend,
    config: TopoPromptConfig,
    budget: BudgetLedger,
    reporter: CompileProgressReporter | None = None,
) -> list[CandidateEvaluation]:
    confirmed: list[CandidateEvaluation] = []
    ranked_candidates = sorted(candidates, key=lambda item: item.search_score, reverse=True)
    for candidate in (reporter.track(
        ranked_candidates,
        desc="Confirmation",
        total=len(ranked_candidates),
        leave=False,
        level=1,
    ) if reporter is not None else ranked_candidates):
        evaluation = _evaluate_candidate(
            program=candidate.program,
            task_spec=task_spec,
            examples=validation_examples,
            metric_fn=metric_fn,
            backend=backend,
            config=config,
            budget=budget,
            phase="confirmation",
            stage="confirmation",
            max_examples=min(len(validation_examples), config.compile.confirmation_examples),
            fewshot_examples=[],
            parent_id=candidate.parent_id,
            edit_applied=candidate.edit_applied,
            reporter=reporter,
        )
        if evaluation is not None:
            confirmed.append(evaluation)
    return confirmed


def _estimate_program_invocations_per_example(program: PromptProgram, config: TopoPromptConfig) -> int:
    invocations = 0
    for node in program.nodes:
        if node.execution_mode == "pass_through":
            continue
        if node.execution_mode == "decompose_macro":
            invocations += 1 + config.program.max_subquestions_per_decompose
        else:
            invocations += 1
    return max(invocations, 1)


def _budgeted_stage_example_cap(
    *,
    program: PromptProgram,
    phase: str,
    configured_examples: int,
    budget: BudgetLedger | None,
    remaining_candidates: int,
    config: TopoPromptConfig,
) -> int:
    if configured_examples <= 0:
        return 0
    if budget is None:
        return configured_examples
    remaining_calls = budget.remaining(phase)
    if remaining_calls <= 0 or remaining_candidates <= 0:
        return 0
    estimated_calls_per_example = _estimate_program_invocations_per_example(program, config)
    if remaining_calls < estimated_calls_per_example:
        return 0
    fair_share_calls = max(estimated_calls_per_example, remaining_calls // remaining_candidates)
    affordable_examples = max(1, fair_share_calls // estimated_calls_per_example)
    return min(configured_examples, affordable_examples)


def _estimate_confirmation_calls(
    *,
    program: PromptProgram,
    validation_examples: list[Example],
    config: TopoPromptConfig,
) -> int:
    target_examples = min(len(validation_examples), config.compile.confirmation_examples)
    return _estimate_program_invocations_per_example(program, config) * target_examples


def _select_final_candidate(
    *,
    best_candidate: CandidateEvaluation,
    smallest_effective: CandidateEvaluation,
    config: TopoPromptConfig,
) -> CandidateEvaluation:
    if config.compile.final_program_policy == "smallest_effective":
        return smallest_effective
    return best_candidate


def _select_affordable_confirmation_candidates(
    *,
    candidates: list[CandidateEvaluation],
    validation_examples: list[Example],
    budget: BudgetLedger,
    config: TopoPromptConfig,
    reporter: CompileProgressReporter | None = None,
) -> list[CandidateEvaluation]:
    if not candidates or not validation_examples:
        return []
    available_calls = budget.remaining("confirmation") + budget.remaining("reserve")
    if available_calls <= 0:
        if reporter is not None:
            reporter.log("No confirmation budget remains for final selection.")
        return []

    selected: list[CandidateEvaluation] = []
    reserved_calls = 0
    for index, candidate in enumerate(candidates):
        estimated_calls = _estimate_confirmation_calls(
            program=candidate.program,
            validation_examples=validation_examples,
            config=config,
        )
        remaining_calls = available_calls - reserved_calls
        if estimated_calls <= remaining_calls:
            selected.append(candidate)
            reserved_calls += estimated_calls
            continue
        if index == 0:
            selected.append(candidate)
            if reporter is not None:
                reporter.log(
                    (
                        f"Confirmation budget is tight; attempting only the top candidate "
                        f"{candidate.program.program_id} (est_calls={estimated_calls}, available={available_calls})."
                    ),
                )
        elif reporter is not None:
            reporter.log(
                (
                    f"Stopping final confirmation at {candidate.program.program_id}; "
                    f"est_calls={estimated_calls} remaining={remaining_calls}."
                ),
                level=1,
            )
        break
    return selected


def _is_fully_evaluated(candidate: CandidateEvaluation) -> bool:
    return bool(candidate.metadata.get("fully_evaluated", False))


def _examples_evaluated(candidate: CandidateEvaluation) -> int:
    return int(candidate.metadata.get("examples_evaluated", len(candidate.traces)))


def _target_examples(candidate: CandidateEvaluation) -> int:
    target_examples = int(candidate.metadata.get("target_examples", 0))
    return max(target_examples, _examples_evaluated(candidate), 1)


def _fallback_evidence_threshold(candidate: CandidateEvaluation, config: TopoPromptConfig) -> int:
    if candidate.stage == "confirmation":
        configured_threshold = config.compile.min_confirmation_examples_for_fallback
    elif candidate.stage == "narrowing":
        configured_threshold = config.compile.min_narrowing_examples_for_fallback
    else:
        configured_threshold = config.compile.min_screening_examples_for_fallback
    return min(max(configured_threshold, 1), _target_examples(candidate))


def _has_fallback_evidence(candidate: CandidateEvaluation, config: TopoPromptConfig) -> bool:
    return _examples_evaluated(candidate) >= _fallback_evidence_threshold(candidate, config)


def _select_final_selection_pool(
    *,
    finalists: list[CandidateEvaluation],
    beam: list[CandidateEvaluation],
    seed_evals: list[CandidateEvaluation],
    config: TopoPromptConfig,
) -> tuple[list[CandidateEvaluation], bool]:
    fully_confirmed = _dedupe_confirmed([candidate for candidate in finalists if _is_fully_evaluated(candidate)])
    if fully_confirmed:
        return fully_confirmed, False
    partially_confirmed = _dedupe_confirmed(
        [candidate for candidate in finalists if _has_fallback_evidence(candidate, config)]
    )
    if partially_confirmed:
        return partially_confirmed, True
    evidenced_beam = _dedupe_confirmed([candidate for candidate in beam if _has_fallback_evidence(candidate, config)])
    if evidenced_beam:
        return evidenced_beam, True
    evidenced_seeds = _dedupe_confirmed(
        [candidate for candidate in seed_evals if _has_fallback_evidence(candidate, config)]
    )
    if evidenced_seeds:
        return evidenced_seeds, True
    fallback_pool = beam or finalists or seed_evals
    return fallback_pool, True


def _select_diverse_beam(candidates: list[CandidateEvaluation], *, width: int, min_families: int) -> list[CandidateEvaluation]:
    ranked = sorted(candidates, key=lambda item: (item.search_score, item.score), reverse=True)
    if not ranked:
        return []
    beam: list[CandidateEvaluation] = []
    family_buckets: dict[str, list[CandidateEvaluation]] = {}
    for candidate in ranked:
        family_buckets.setdefault(candidate.family_signature, []).append(candidate)
    for family in list(family_buckets)[:min_families]:
        beam.append(family_buckets[family][0])
    seen = {candidate.topology_fingerprint for candidate in beam}
    for candidate in ranked:
        if len(beam) >= width:
            break
        if candidate.topology_fingerprint not in seen:
            beam.append(candidate)
            seen.add(candidate.topology_fingerprint)
    return beam[:width]


def _dedupe_edits(edits: list[CandidateEdit]) -> list[CandidateEdit]:
    seen = set()
    unique = []
    for edit in edits:
        key = (
            edit.edit_type,
            edit.target_node_id,
            edit.new_node_type.value if edit.new_node_type else None,
            edit.module_role,
            tuple(edit.branch_labels or []),
            edit.rewrite_instruction,
        )
        if key not in seen:
            seen.add(key)
            unique.append(edit)
    return unique


def _dedupe_proposals(proposals: list[tuple[PromptProgram, str, str]]) -> list[tuple[PromptProgram, str, str]]:
    seen = set()
    unique = []
    for program, parent_id, edit_applied in proposals:
        fingerprint = topology_fingerprint(program)
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append((program, parent_id, edit_applied))
    return unique


def _dedupe_confirmed(candidates: list[CandidateEvaluation]) -> list[CandidateEvaluation]:
    seen = {}
    for candidate in candidates:
        seen[candidate.topology_fingerprint] = candidate
    return list(seen.values())


def _archive_record(candidate: CandidateEvaluation, *, round_index: int) -> CandidateArchiveRecord:
    screening_score = candidate.score if candidate.stage == "screening" else None
    narrowing_score = candidate.score if candidate.stage == "narrowing" else None
    confirmation_score = candidate.score if candidate.stage == "confirmation" else None
    return CandidateArchiveRecord(
        program_id=candidate.program.program_id,
        parent_id=candidate.parent_id,
        edit_applied=candidate.edit_applied,
        topology_fingerprint=candidate.topology_fingerprint,
        family_signature=candidate.family_signature,
        screening_score=screening_score,
        narrowing_score=narrowing_score,
        confirmation_score=confirmation_score,
        search_score=candidate.search_score,
        complexity=candidate.complexity,
        inference_cost=candidate.mean_invocations,
        parse_failure_rate=candidate.parse_failure_rate,
        round_index=round_index,
        metadata={"stage": candidate.stage},
    )


def _induce_route_diagnostics(
    *,
    program: PromptProgram,
    task_spec: TaskSpec,
    examples: list[Example],
    base_traces: list,
    metric_fn: MetricFn,
    backend: LLMBackend,
    config: TopoPromptConfig,
    budget: BudgetLedger | None,
) -> list[RouteDiagnostic]:
    route_nodes = [node for node in program.nodes if node.node_type.value == "route" and node.route_spec]
    if not route_nodes:
        return []
    route_node = route_nodes[0]
    base_by_example = {trace.example_id: trace for trace in base_traces}
    diagnostics: list[RouteDiagnostic] = []
    executor = ProgramExecutor(backend=backend, config=config, budget_ledger=budget)
    for example in examples:
        base_trace = base_by_example.get(example.example_id)
        chosen_branch = None
        chosen_score = 0.0
        if base_trace is not None:
            for node_trace in base_trace.node_traces:
                if node_trace.node_id == route_node.node_id:
                    chosen_branch = node_trace.route_choice
                    break
            chosen_score = base_trace.correctness or 0.0
        branch_scores = {}
        for branch in route_node.route_spec.branch_labels:
            if budget is not None and not budget.can_spend("reserve", 1):
                break
            try:
                forced = executor.run_program(
                    program=program,
                    task_spec=task_spec,
                    example_id=example.example_id,
                    task_input=example.input,
                    phase="reserve",
                    force_route_choices={route_node.node_id: branch},
                )
            except BudgetExhausted:
                break
            branch_scores[branch] = metric_fn(forced.trace.final_output, example)
        if branch_scores:
            oracle_branch = max(branch_scores, key=branch_scores.get)
            diagnostics.append(
                RouteDiagnostic(
                    example_id=example.example_id,
                    route_node_id=route_node.node_id,
                    chosen_branch=chosen_branch,
                    oracle_branch=oracle_branch,
                    branch_scores=branch_scores,
                    regret=branch_scores[oracle_branch] - chosen_score,
                    confidence=None,
                )
            )
    return diagnostics


def _llm_guided_edit_proposals(
    *,
    parent: CandidateEvaluation,
    analysis: TaskAnalysis,
    backend: LLMBackend,
    config: TopoPromptConfig,
    budget: BudgetLedger,
) -> list[CandidateEdit]:
    if not budget.spend("screening", 1):
        return []
    program_summary = {
        "program_id": parent.program.program_id,
        "family": parent.family_signature,
        "nodes": [node.node_type.value for node in parent.program.nodes],
        "search_score": parent.search_score,
    }
    diagnostics = {
        "score": parent.score,
        "parse_failure_rate": parent.parse_failure_rate,
        "mean_invocations": parent.mean_invocations,
    }
    schema = {
        "type": "object",
        "properties": {
            "proposals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "edit_type": {"type": "string"},
                        "target_node_id": {"type": ["string", "null"]},
                        "new_node_type": {"type": ["string", "null"]},
                        "module_role": {"type": ["string", "null"]},
                        "branch_labels": {"type": ["array", "null"]},
                        "rewrite_instruction": {"type": ["string", "null"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["edit_type", "reason"],
                },
            }
        },
        "required": ["proposals"],
    }
    try:
        response = backend.generate_structured(
            system_prompt="You are proposing the next structural edit for a prompt-program compiler.",
            user_prompt=(
                f"Task summary:\n{analysis.model_dump_json(indent=2)}\n\n"
                f"Current program summary:\n{json.dumps(program_summary, indent=2)}\n\n"
                f"Current diagnostics:\n{json.dumps(diagnostics, indent=2)}\n\n"
                "Allowed edit operators:\n"
                "- add_node\n- delete_node\n- replace_node_type\n- insert_verify_after\n- insert_plan_before\n"
                "- split_with_route\n- remove_route\n- swap_branch_target\n- rewrite_prompt_module\n"
                "- add_fewshot_module\n- drop_fewshot_module\n- change_finalize_format\n\n"
                f"Hard constraints:\n- graph must remain a DAG\n- exactly one finalize node\n- max nodes: {config.program.max_nodes}\n"
                f"- max route nodes: {config.program.max_route_nodes}\n- max branch fanout: {config.program.max_branch_fanout}\n"
            ),
            schema=schema,
            model=config.model.name,
            temperature=0.0,
            max_output_tokens=config.model.max_output_tokens,
        )
        payload = response.structured or json.loads(response.text)
        proposals = []
        for raw in payload.get("proposals", [])[: config.compile.llm_edit_proposals_per_parent]:
            if raw.get("new_node_type"):
                raw["new_node_type"] = raw["new_node_type"]
            proposals.append(CandidateEdit.model_validate(raw))
        return proposals
    except Exception:
        return []


def _validate_candidate(program: PromptProgram, config: TopoPromptConfig) -> bool:
    try:
        validate_program(program, config.program)
        return True
    except ProgramValidationError:
        return False
