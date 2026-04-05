from __future__ import annotations

from topoprompt.compiler.edits import apply_edit, generate_heuristic_edits
from topoprompt.compiler.seeds import instantiate_seed_program
from topoprompt.compiler.validator import validate_program
from topoprompt.runtime.executor import ProgramExecutor
from topoprompt.schemas import CandidateEdit, TaskAnalysis


def _edit_config(small_config, *, max_candidates: int = 6):
    config = small_config.model_copy(deep=True)
    config.compile.max_candidates_per_parent = max_candidates
    return config


def test_generate_heuristic_edits_prioritizes_lightweight_classification_edits(small_config, simple_task_spec):
    config = _edit_config(small_config)
    analysis = TaskAnalysis(
        task_family="classification",
        output_format="label",
        input_heterogeneity="high",
        initial_seed_templates=["direct_finalize"],
    )
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    edits = generate_heuristic_edits(program=program, analysis=analysis, config=config)
    edit_types = [edit.edit_type for edit in edits]

    assert edit_types[:4] == [
        "split_with_route",
        "rewrite_prompt_module",
        "change_finalize_format",
        "add_fewshot_module",
    ]
    assert "insert_verify_after" not in edit_types
    assert "insert_plan_before" not in edit_types


def test_generate_heuristic_edits_prioritizes_critique_loop_for_generation(small_config, simple_task_spec):
    config = _edit_config(small_config)
    analysis = TaskAnalysis(
        task_family="generation",
        output_format="paragraph",
        initial_seed_templates=["direct_finalize"],
    )
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    edits = generate_heuristic_edits(program=program, analysis=analysis, config=config)
    edit_types = [edit.edit_type for edit in edits]

    assert edit_types[:3] == [
        "insert_critique_revise_after",
        "rewrite_prompt_module",
        "add_fewshot_module",
    ]
    assert "insert_plan_before" not in edit_types
    assert "insert_verify_after" not in edit_types


def test_generate_heuristic_edits_prioritizes_formatting_for_extraction(small_config, simple_task_spec):
    config = _edit_config(small_config)
    analysis = TaskAnalysis(
        task_family="extraction",
        output_format="json",
        needs_verification=True,
        initial_seed_templates=["direct_finalize"],
    )
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    edits = generate_heuristic_edits(program=program, analysis=analysis, config=config)
    edit_types = [edit.edit_type for edit in edits]

    assert edit_types[0] == "insert_format_after"
    assert "insert_verify_after" in edit_types
    assert "change_finalize_format" in edit_types


def test_generate_heuristic_edits_prioritizes_code_specific_structure(small_config, simple_task_spec):
    config = _edit_config(small_config)
    analysis = TaskAnalysis(
        task_family="code",
        output_format="code",
        needs_verification=True,
        initial_seed_templates=["direct_finalize"],
    )
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    edits = generate_heuristic_edits(program=program, analysis=analysis, config=config)
    edit_types = [edit.edit_type for edit in edits]

    assert edit_types[:3] == [
        "insert_critique_revise_after",
        "insert_verify_after",
        "replace_node_type",
    ]


def test_apply_edit_insert_format_after_produces_valid_program(small_config, simple_task_spec):
    analysis = TaskAnalysis(
        task_family="extraction",
        output_format="json",
        initial_seed_templates=["direct_finalize"],
    )
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    edited = apply_edit(
        program=program,
        edit=CandidateEdit(edit_type="insert_format_after", target_node_id="direct_1"),
        analysis=analysis,
    )

    validate_program(edited, small_config.program)
    assert "format_1" in edited.node_map()
    assert edited.node_map()[edited.finalize_node_id].config["source_key"] == "formatted_answer"
    assert any(edge.source == "direct_1" and edge.target == "format_1" for edge in edited.edges)


def test_apply_edit_insert_critique_revise_after_executes(fake_backend, small_config, simple_task_spec, gsm8k_examples):
    analysis = TaskAnalysis(
        task_family="generation",
        output_format="paragraph",
        initial_seed_templates=["direct_finalize"],
    )
    program = instantiate_seed_program(task_spec=simple_task_spec, analysis=analysis, template_name="direct_finalize")
    assert program is not None

    edited = apply_edit(
        program=program,
        edit=CandidateEdit(edit_type="insert_critique_revise_after", target_node_id="direct_1"),
        analysis=analysis,
    )

    validate_program(edited, small_config.program)
    executor = ProgramExecutor(backend=fake_backend, config=small_config)
    result = executor.run_program(
        program=edited,
        task_spec=simple_task_spec,
        example_id=gsm8k_examples[0].example_id,
        task_input=gsm8k_examples[0].input,
        phase="confirmation",
    )

    assert result.trace.final_output == "4"
    assert [trace.node_id for trace in result.trace.node_traces] == [
        "direct_1",
        "critique_1",
        "solve_1",
        "finalize_1",
    ]
