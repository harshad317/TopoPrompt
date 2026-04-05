from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from topoprompt.backends.llm_client import FakeBackend
from topoprompt.compiler.search import (
    _budgeted_stage_example_cap,
    _synthesize_failure_grounded_rewrite_edit,
    _select_affordable_confirmation_candidates,
    _select_final_candidate,
    _select_final_selection_pool,
    _infer_task_spec,
    _resolve_metric,
    compile_task,
    evaluate_program_on_examples,
)
from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.config import TopoPromptConfig
from topoprompt.compiler.objective import search_score
from topoprompt.eval.budget import BudgetLedger
from topoprompt.eval.metrics import numeric_metric
from topoprompt.ir import family_signature
from topoprompt.progress import CompileProgressReporter
from topoprompt.schemas import CandidateEvaluation, Example, ProgramExecutionTrace, PromptProgram, TaskAnalysis
from topoprompt.transfer.features import extract_compile_winner_record, extract_transfer_features
from topoprompt.transfer.store import TraceStore


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


def test_compile_task_logs_embedding_status(gsm8k_examples, small_config, tmp_path):
    buffer = StringIO()
    reporter = CompileProgressReporter(
        enabled=True,
        verbosity=1,
        console=Console(file=buffer, force_terminal=False, color_system=None),
    )
    semantic_backend = FakeBackend(
        embed_handler=lambda _text, _model: [0.1, 0.2, 0.3],
        embeddings_are_real=True,
    )

    compile_task(
        task_description="Answer simple arithmetic questions accurately.",
        examples=gsm8k_examples,
        metric="gsm8k",
        backend=semantic_backend,
        config=small_config,
        output_dir=tmp_path / "embedding_status_run",
        task_id="gsm8k_embedding_status",
        progress_reporter=reporter,
    )

    assert "Task embedding: real (3d)" in buffer.getvalue()


def test_resolve_metric_canonicalizes_benchmark_alias():
    metric_name, metric_fn = _resolve_metric("gsm8k")

    assert metric_name == "numeric"
    assert metric_fn is numeric_metric


def test_infer_task_spec_normalizes_python_targets_to_jsonschema_types():
    task_spec = _infer_task_spec(
        task_description="Extract entities and return structured data.",
        examples=[
            Example(
                example_id="structured_1",
                input={"prompt": "Ada founded Example Labs."},
                target={"people": ["Ada"], "organizations": ["Example Labs"]},
            )
        ],
        task_id="structured_task",
    )

    assert task_spec.output_schema == {"type": "object"}


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


def test_search_score_uses_family_conditioned_objective_weights():
    config = TopoPromptConfig()

    classification_score = search_score(
        perf=0.75,
        mean_invocations=3.0,
        complexity=0.20,
        parse_failure_rate=0.0,
        coverage_ratio=1.0,
        objective_config=config.objective,
        program_config=config.program,
        task_family="classification",
    )
    math_score = search_score(
        perf=0.75,
        mean_invocations=3.0,
        complexity=0.20,
        parse_failure_rate=0.0,
        coverage_ratio=1.0,
        objective_config=config.objective,
        program_config=config.program,
        task_family="math_reasoning",
    )
    code_score = search_score(
        perf=0.75,
        mean_invocations=3.0,
        complexity=0.20,
        parse_failure_rate=0.0,
        coverage_ratio=1.0,
        objective_config=config.objective,
        program_config=config.program,
        task_family="code",
    )

    assert classification_score < math_score < code_score


def test_failure_grounded_rewrite_uses_failed_examples_in_prompt(small_config, simple_task_spec):
    analysis = TaskAnalysis(task_family="instruction_following", output_format="short_answer")
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    candidate = CandidateEvaluation(
        program=program,
        topology_fingerprint="failure-grounded",
        family_signature="direct",
        stage="narrowing",
        score=0.5,
        search_score=0.5,
        mean_invocations=1.0,
        mean_tokens=50.0,
        complexity=0.08,
        traces=[
            ProgramExecutionTrace(
                example_id="fail_1",
                program_id=program.program_id,
                node_traces=[],
                final_output="London",
                correctness=0.0,
            ),
            ProgramExecutionTrace(
                example_id="pass_1",
                program_id=program.program_id,
                node_traces=[],
                final_output="Tokyo",
                correctness=1.0,
            ),
        ],
    )
    examples = [
        Example(example_id="fail_1", input={"question": "What is the capital of France?"}, target="Paris"),
        Example(example_id="pass_1", input={"question": "What is the capital of Japan?"}, target="Tokyo"),
    ]
    captured_prompt: dict[str, str] = {}

    def structured_handler(system_prompt: str, user_prompt: str, schema: dict[str, object]) -> dict[str, str]:
        captured_prompt["user_prompt"] = user_prompt
        return {
            "target_node_id": "direct_1",
            "module_role": "instruction",
            "rewrite_instruction": "Answer the question exactly and match the expected factual answer.",
            "reason": "The model confused the target fact on failed examples.",
        }

    backend = FakeBackend(structured_handler=structured_handler)
    budget = BudgetLedger.from_compile_config(small_config.compile)

    edit = _synthesize_failure_grounded_rewrite_edit(
        parent=candidate,
        analysis=analysis,
        examples=examples,
        backend=backend,
        config=small_config,
        budget=budget,
    )

    assert edit is not None
    assert edit.edit_type == "rewrite_prompt_module"
    assert edit.target_node_id == "direct_1"
    assert "capital of France" in captured_prompt["user_prompt"]
    assert '"expected_output": "Paris"' in captured_prompt["user_prompt"]
    assert '"observed_output": "London"' in captured_prompt["user_prompt"]


def test_extract_transfer_features_records_real_embedding_when_provided(simple_task_spec):
    analysis = TaskAnalysis(task_family="reasoning", initial_seed_templates=["direct_finalize"])
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    features = extract_transfer_features(
        task_spec=simple_task_spec,
        program=program,
        metric_name="numeric",
        task_embedding=[0.25, 0.75],
        task_embedding_is_real=True,
    )

    assert features["task_features"]["task_embedding"] == [0.25, 0.75]
    assert features["task_features"]["task_embedding_is_real"] is True


def test_trace_store_top_warm_starts_prefers_structural_overlap_across_families(simple_task_spec):
    solve_verify_analysis = TaskAnalysis(
        task_family="code",
        needs_verification=True,
        initial_seed_templates=["solve_verify_finalize"],
    )
    solve_verify_program = instantiate_seed_program(
        task_spec=simple_task_spec,
        analysis=solve_verify_analysis,
        template_name="solve_verify_finalize",
    )
    direct_program = instantiate_seed_program(
        task_spec=simple_task_spec,
        analysis=TaskAnalysis(task_family="code", initial_seed_templates=["direct_finalize"]),
        template_name="direct_finalize",
    )
    assert solve_verify_program is not None and direct_program is not None

    query_signature = family_signature(solve_verify_program)
    store = TraceStore()
    store.append(
        {
            "record_type": "compile_winner",
            "task_family": "classification",
            "metric_name": "numeric",
            "family_signature": query_signature,
            "winning_topology_fingerprint": "cross_family_structural_match",
            "winning_score": 0.90,
            "program": solve_verify_program.model_dump(mode="json"),
        }
    )
    store.append(
        {
            "record_type": "compile_winner",
            "task_family": "code",
            "metric_name": "exact_match",
            "family_signature": family_signature(direct_program),
            "winning_topology_fingerprint": "same_family_weaker_shape",
            "winning_score": 0.85,
            "program": direct_program.model_dump(mode="json"),
        }
    )

    results = store.top_warm_starts(
        task_family="code",
        metric_name="numeric",
        family_signature=query_signature,
        limit=2,
    )

    assert [record["winning_topology_fingerprint"] for record in results] == [
        "cross_family_structural_match",
        "same_family_weaker_shape",
    ]
    assert results[0]["warm_start_similarity_score"] == 3
    assert results[1]["warm_start_similarity_score"] == 3
    assert results[0]["warm_start_rank_score"] > results[1]["warm_start_rank_score"]


def test_trace_store_top_warm_starts_uses_embedding_similarity_to_break_structural_ties(simple_task_spec):
    store = TraceStore()
    shared_signature = "solve-verify-finalize|routes=0|fanout=0|verify=1|decompose=0"
    for fingerprint, embedding, score in [
        ("semantic_match", [1.0, 0.0], 0.80),
        ("semantic_mismatch", [0.0, 1.0], 0.80),
    ]:
        store.append(
            {
                "record_type": "compile_winner",
                "task_family": "code",
                "metric_name": "numeric",
                "family_signature": shared_signature,
                "winning_topology_fingerprint": fingerprint,
                "winning_score": score,
                "task_features": {
                    "task_embedding": embedding,
                    "task_embedding_is_real": True,
                },
                "program": {},
            }
        )

    results = store.top_warm_starts(
        task_family="code",
        metric_name="numeric",
        family_signature=shared_signature,
        task_embedding=[1.0, 0.0],
        task_embedding_is_real=True,
        limit=2,
    )

    assert [record["winning_topology_fingerprint"] for record in results] == [
        "semantic_match",
        "semantic_mismatch",
    ]
    assert results[0]["warm_start_similarity_score"] > results[1]["warm_start_similarity_score"]


def test_extract_compile_winner_record_persists_task_description(simple_task_spec):
    analysis = TaskAnalysis(task_family="reasoning", initial_seed_templates=["direct_finalize"])
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    candidate = CandidateEvaluation(
        program=program,
        topology_fingerprint="winner",
        family_signature="direct-finalize",
        stage="confirmation",
        score=0.9,
        search_score=0.88,
        mean_invocations=1.0,
        mean_tokens=20.0,
        complexity=0.05,
    )

    record = extract_compile_winner_record(
        task_spec=simple_task_spec,
        candidate=candidate,
        metric_name="numeric",
        task_embedding=[0.1, 0.2],
        task_embedding_is_real=True,
    )

    assert record["task_description"] == simple_task_spec.description


def test_compile_task_loads_transfer_warm_start_seeds(fake_backend, small_config, gsm8k_examples, simple_task_spec, tmp_path: Path):
    analysis = TaskAnalysis(task_family="math_reasoning", initial_seed_templates=["direct_self_consistency_x3"])
    warm_start_program = instantiate_seed_program(
        task_spec=simple_task_spec,
        analysis=analysis,
        template_name="direct_self_consistency_x3",
    )
    assert warm_start_program is not None

    store = TraceStore(tmp_path / "transfer_trace_store.jsonl")
    store.append(
        {
            "record_type": "compile_winner",
            "task_family": "math_reasoning",
            "metric_name": "numeric",
            "winning_topology_fingerprint": "warm-self-consistency",
            "winning_score": 0.91,
            "program": warm_start_program.model_dump(mode="json"),
        }
    )
    store.flush()

    artifact = compile_task(
        task_description="Answer simple arithmetic questions accurately.",
        examples=gsm8k_examples,
        metric="gsm8k",
        backend=fake_backend,
        config=small_config,
        output_dir=tmp_path / "run",
        task_id="transfer_warm_start",
    )

    assert any(program.program_id.startswith("warm_start_") for program in artifact.seed_programs)
    assert any(program.metadata.get("warm_start_source") == "transfer_trace_store" for program in artifact.seed_programs)
    assert any(program.metadata.get("warm_start_rank_score", 0.0) > 0.0 for program in artifact.seed_programs)
