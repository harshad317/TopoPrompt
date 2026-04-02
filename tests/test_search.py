from __future__ import annotations

from pathlib import Path

from topoprompt.compiler.search import (
    _budgeted_stage_example_cap,
    _select_affordable_confirmation_candidates,
    _select_final_candidate,
    _select_final_selection_pool,
    compile_task,
    evaluate_program_on_examples,
)
from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.config import TopoPromptConfig
from topoprompt.compiler.objective import search_score
from topoprompt.eval.budget import BudgetLedger
from topoprompt.eval.metrics import numeric_metric
from topoprompt.schemas import CandidateEvaluation, Example, PromptProgram, TaskAnalysis


def test_compile_task_runs_end_to_end(fake_backend, small_config, gsm8k_examples, tmp_path):
    artifact = compile_task(
        task_description="Answer simple arithmetic questions accurately.",
        examples=gsm8k_examples,
        metric="gsm8k",
        backend=fake_backend,
        config=small_config,
        output_dir=tmp_path / "run",
        task_id="gsm8k_smoke",
    )
    assert artifact.program_ir.program_id
    assert artifact.metrics.best_validation_score >= 0.0
    assert artifact.metrics.final_program_id == artifact.program_ir.program_id
    assert (tmp_path / "run" / "final_program.json").exists()
    assert (tmp_path / "run" / "best_program.json").exists()
    assert (tmp_path / "run" / "smallest_effective_program.json").exists()
    assert artifact.candidate_archive


def test_compile_budget_override_rebalances_phase_budgets(fake_backend, gsm8k_examples, tmp_path):
    artifact = compile_task(
        task_description="Answer simple arithmetic questions accurately.",
        examples=gsm8k_examples,
        metric="gsm8k",
        backend=fake_backend,
        compile_budget=48,
        output_dir=tmp_path / "budget_run",
        task_id="gsm8k_budget_override",
    )

    assert artifact.metrics.planned_budget_calls == 48
    assert sum(phase.planned_calls for phase in artifact.metrics.planned_budget_by_phase) == 48


def test_evaluate_program_runs_full_dataset_without_compile_budget_cap(fake_backend, small_config, simple_task_spec):
    analysis = TaskAnalysis(needs_reasoning=False, initial_seed_templates=["direct_finalize"])
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    examples = [
        Example(
            example_id=f"eval_{index}",
            input={"question": f"What is {index} + 1?"},
            target=str(index + 1),
        )
        for index in range(25)
    ]

    result = evaluate_program_on_examples(
        program=program,
        task_spec=simple_task_spec,
        examples=examples,
        metric_fn=numeric_metric,
        backend=fake_backend,
        config=small_config,
        phase="confirmation",
    )

    assert len(result["traces"]) == len(examples)


def test_final_selection_falls_back_to_beam_instead_of_screening_seed():
    config = TopoPromptConfig()
    baseline_program = PromptProgram(
        program_id="direct_finalize",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    beam_program = PromptProgram(
        program_id="plan_solve_finalize_best",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    partial_confirmation_program = PromptProgram(
        program_id="plan_solve_finalize_partial",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )

    screening_seed = CandidateEvaluation(
        program=baseline_program,
        topology_fingerprint="seed",
        family_signature="direct",
        stage="screening",
        score=0.375,
        search_score=0.3592,
        mean_invocations=1.0,
        mean_tokens=390.0,
        complexity=0.086,
        metadata={"examples_evaluated": 4, "target_examples": 4},
    )
    beam_candidate = CandidateEvaluation(
        program=beam_program,
        topology_fingerprint="beam",
        family_signature="plan-solve",
        stage="narrowing",
        score=0.875,
        search_score=0.8610,
        mean_invocations=2.0,
        mean_tokens=799.0,
        complexity=0.140,
        metadata={"examples_evaluated": 8, "target_examples": 8},
    )
    partial_confirmation = CandidateEvaluation(
        program=partial_confirmation_program,
        topology_fingerprint="partial",
        family_signature="plan-solve",
        stage="confirmation",
        score=0.900,
        search_score=0.8800,
        mean_invocations=2.0,
        mean_tokens=810.0,
        complexity=0.141,
        metadata={"fully_evaluated": False, "examples_evaluated": 3, "target_examples": 12},
    )

    pool, used_fallback = _select_final_selection_pool(
        finalists=[partial_confirmation],
        beam=[beam_candidate],
        seed_evals=[screening_seed],
        config=config,
    )

    assert used_fallback is True
    assert [candidate.program.program_id for candidate in pool] == ["plan_solve_finalize_best"]


def test_final_selection_prefers_partial_confirmation_with_sufficient_evidence():
    config = TopoPromptConfig()
    beam_program = PromptProgram(
        program_id="screening_fluke",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    partial_confirmation_program = PromptProgram(
        program_id="partially_confirmed",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )

    beam_candidate = CandidateEvaluation(
        program=beam_program,
        topology_fingerprint="beam",
        family_signature="plan-solve",
        stage="screening",
        score=1.0,
        search_score=0.95,
        mean_invocations=2.0,
        mean_tokens=800.0,
        complexity=0.14,
        metadata={"examples_evaluated": 1, "target_examples": 1},
    )
    partial_confirmation = CandidateEvaluation(
        program=partial_confirmation_program,
        topology_fingerprint="partial",
        family_signature="plan-solve",
        stage="confirmation",
        score=0.6,
        search_score=0.58,
        mean_invocations=2.0,
        mean_tokens=810.0,
        complexity=0.14,
        metadata={"fully_evaluated": False, "examples_evaluated": 10, "target_examples": 12},
    )

    pool, used_fallback = _select_final_selection_pool(
        finalists=[partial_confirmation],
        beam=[beam_candidate],
        seed_evals=[],
        config=config,
    )

    assert used_fallback is True
    assert [candidate.program.program_id for candidate in pool] == ["partially_confirmed"]


def test_final_program_policy_defaults_to_best_candidate():
    config = TopoPromptConfig()
    best_program = PromptProgram(
        program_id="best_program",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    smallest_program = PromptProgram(
        program_id="smallest_program",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    best_candidate = CandidateEvaluation(
        program=best_program,
        topology_fingerprint="best",
        family_signature="plan",
        stage="confirmation",
        score=0.8,
        search_score=0.8,
        mean_invocations=3.0,
        mean_tokens=100.0,
        complexity=0.18,
    )
    smallest_candidate = CandidateEvaluation(
        program=smallest_program,
        topology_fingerprint="smallest",
        family_signature="direct",
        stage="confirmation",
        score=0.79,
        search_score=0.79,
        mean_invocations=1.0,
        mean_tokens=80.0,
        complexity=0.08,
    )

    selected = _select_final_candidate(
        best_candidate=best_candidate,
        smallest_effective=smallest_candidate,
        config=config,
    )

    assert selected.program.program_id == "best_program"


def test_final_program_policy_can_export_smallest_effective():
    config = TopoPromptConfig.model_validate({"compile": {"final_program_policy": "smallest_effective"}})
    best_program = PromptProgram(
        program_id="best_program",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    smallest_program = PromptProgram(
        program_id="smallest_program",
        task_id="task",
        nodes=[],
        edges=[],
        entry_node_id="entry",
        finalize_node_id="finalize",
    )
    best_candidate = CandidateEvaluation(
        program=best_program,
        topology_fingerprint="best",
        family_signature="plan",
        stage="confirmation",
        score=0.8,
        search_score=0.8,
        mean_invocations=3.0,
        mean_tokens=100.0,
        complexity=0.18,
    )
    smallest_candidate = CandidateEvaluation(
        program=smallest_program,
        topology_fingerprint="smallest",
        family_signature="direct",
        stage="confirmation",
        score=0.79,
        search_score=0.79,
        mean_invocations=1.0,
        mean_tokens=80.0,
        complexity=0.08,
    )

    selected = _select_final_candidate(
        best_candidate=best_candidate,
        smallest_effective=smallest_candidate,
        config=config,
    )

    assert selected.program.program_id == "smallest_program"


def test_affordable_confirmation_selection_preserves_top_ranked_prefix(simple_task_spec):
    config = TopoPromptConfig.model_validate(
        {
            "model": {"name": "fake-model", "repair_model": "fake-model"},
            "compile": {
                "confirmation_budget_calls": 12,
                "reserve_budget_calls": 0,
                "confirmation_examples": 4,
            },
            "runtime": {"cache_enabled": False},
        }
    )
    analysis = TaskAnalysis(
        needs_reasoning=True,
        initial_seed_templates=["plan_solve_finalize", "solve_verify_finalize", "direct_finalize"],
    )
    plan_program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="plan_solve_finalize")
    verify_program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="solve_verify_finalize")
    direct_program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert plan_program is not None and verify_program is not None and direct_program is not None

    candidates = [
        CandidateEvaluation(
            program=plan_program,
            topology_fingerprint="plan",
            family_signature="plan",
            stage="narrowing",
            score=0.8,
            search_score=0.8,
            mean_invocations=2.0,
            mean_tokens=100.0,
            complexity=0.1,
        ),
        CandidateEvaluation(
            program=verify_program,
            topology_fingerprint="verify",
            family_signature="verify",
            stage="narrowing",
            score=0.79,
            search_score=0.79,
            mean_invocations=2.0,
            mean_tokens=110.0,
            complexity=0.11,
        ),
        CandidateEvaluation(
            program=direct_program,
            topology_fingerprint="direct",
            family_signature="direct",
            stage="narrowing",
            score=0.7,
            search_score=0.7,
            mean_invocations=1.0,
            mean_tokens=80.0,
            complexity=0.08,
        ),
    ]
    validation_examples = [
        Example(example_id=f"v{index}", input={"question": f"What is {index} + 1?"}, target=str(index + 1))
        for index in range(4)
    ]
    budget = BudgetLedger.from_compile_config(config.compile)

    selected = _select_affordable_confirmation_candidates(
        candidates=candidates,
        validation_examples=validation_examples,
        budget=budget,
        config=config,
    )

    assert [candidate.program.program_id for candidate in selected] == ["plan_solve_finalize"]


def test_budgeted_stage_example_cap_reserves_budget_for_remaining_candidates(simple_task_spec):
    config = TopoPromptConfig.model_validate(
        {
            "compile": {
                "screening_budget_calls": 12,
                "narrowing_budget_calls": 0,
                "confirmation_budget_calls": 0,
                "reserve_budget_calls": 0,
            }
        }
    )
    analysis = TaskAnalysis(needs_reasoning=True, initial_seed_templates=["plan_solve_finalize"])
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="plan_solve_finalize")
    assert program is not None

    budget = BudgetLedger.from_compile_config(config.compile)
    cap = _budgeted_stage_example_cap(
        program=program,
        phase="screening",
        configured_examples=8,
        budget=budget,
        remaining_candidates=3,
        config=config,
    )

    assert cap == 2


def test_search_score_penalizes_partial_coverage():
    config = TopoPromptConfig()

    full_score = search_score(
        perf=0.75,
        mean_invocations=2.0,
        complexity=0.14,
        parse_failure_rate=0.0,
        coverage_ratio=1.0,
        objective_config=config.objective,
        program_config=config.program,
    )
    partial_score = search_score(
        perf=0.75,
        mean_invocations=2.0,
        complexity=0.14,
        parse_failure_rate=0.0,
        coverage_ratio=0.25,
        objective_config=config.objective,
        program_config=config.program,
    )

    assert partial_score < full_score
