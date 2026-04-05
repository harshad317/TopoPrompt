from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from topoprompt.config import TopoPromptConfig
from topoprompt.eval.dspy_baselines import (
    _build_student_program,
    _restore_topoprompt_example,
    _to_dspy_examples,
    apply_dspy_program_state,
    compare_dspy_programs,
    compare_topoprompt_vs_dspy,
    extract_dspy_program_state,
)
from topoprompt.schemas import Example, ProgramEdge, ProgramNode, PromptProgram, TaskSpec


dspy = pytest.importorskip("dspy")


def test_to_dspy_examples_preserves_source_example():
    examples = [
        Example(
            example_id="ex_1",
            input={
                "question": "Which option is correct?",
                "choices": [
                    {"label": "A", "text": "Alpha"},
                    {"label": "B", "text": "Beta"},
                ],
            },
            target="A",
            metadata={"subject": "toy"},
        )
    ]

    dspy_examples = _to_dspy_examples(dspy, examples, input_keys=["question", "choices"])

    assert dspy_examples[0].inputs()["choices"] == "A. Alpha\nB. Beta"
    restored = _restore_topoprompt_example(dspy_examples[0])
    assert restored.example_id == "ex_1"
    assert restored.input == examples[0].input
    assert restored.target == "A"
    assert restored.metadata == {"subject": "toy"}


def test_dspy_program_state_round_trip(small_config: TopoPromptConfig):
    program = _build_student_program(
        dspy=dspy,
        signature="question -> answer",
        strategy="predict",
        input_keys=["question"],
        output_field="answer",
        config=small_config,
    )
    program._topoprompt_program_id = "dspy_mipro_predict"
    predictor = next(iter(program.named_predictors()))[1]
    predictor.signature.instructions = "Custom optimized instruction."
    predictor.demos = [dspy.Example(question="Q1", answer="A1").with_inputs("question")]

    state = extract_dspy_program_state(program)

    restored = _build_student_program(
        dspy=dspy,
        signature="question -> answer",
        strategy="predict",
        input_keys=["question"],
        output_field="answer",
        config=small_config,
    )
    apply_dspy_program_state(dspy=dspy, program=restored, state=state)

    restored_predictor = next(iter(restored.named_predictors()))[1]
    assert restored_predictor.signature.instructions == "Custom optimized instruction."
    assert len(restored_predictor.demos) == 1
    assert restored_predictor.demos[0].question == "Q1"
    assert restored_predictor.demos[0].answer == "A1"


def test_compare_topoprompt_vs_dspy_writes_summary(monkeypatch, tmp_path: Path, fake_backend, small_config: TopoPromptConfig):
    topoprompt_program = PromptProgram(
        program_id="topoprompt_prog",
        task_id="task",
        nodes=[ProgramNode(node_id="n1", node_type="finalize", name="Finalize")],
        edges=[],
        entry_node_id="n1",
        finalize_node_id="n1",
    )
    dspy_program = SimpleNamespace(_topoprompt_program_id="dspy_prog")
    task_spec = TaskSpec(task_id="task", description="Toy task.")
    examples = [
        Example(example_id="ex1", input={"question": "Q1"}, target="1"),
        Example(example_id="ex2", input={"question": "Q2"}, target="2"),
    ]

    monkeypatch.setattr(
        "topoprompt.eval.dspy_baselines.evaluate_program_on_examples",
        lambda **_: {
            "score": 0.5,
            "mean_invocations": 2.0,
            "mean_tokens": 10.0,
            "parse_failure_rate": 0.0,
            "traces": [
                {"example_id": "ex1", "final_output": "1", "correctness": 1.0},
                {"example_id": "ex2", "final_output": "0", "correctness": 0.0},
            ],
        },
    )
    monkeypatch.setattr(
        "topoprompt.eval.dspy_baselines.evaluate_dspy_program_on_examples",
        lambda **_: {
            "score": 0.0,
            "mean_invocations": 1.0,
            "mean_tokens": 0.0,
            "parse_failure_rate": 0.0,
            "traces": [
                {"example_id": "ex1", "final_output": "0", "correctness": 0.0},
                {"example_id": "ex2", "final_output": "0", "correctness": 0.0},
            ],
        },
    )

    result = compare_topoprompt_vs_dspy(
        topoprompt_program=topoprompt_program,
        dspy_program=dspy_program,
        task_spec=task_spec,
        examples=examples,
        metric_fn=lambda pred, ex: 1.0 if pred == ex.target else 0.0,
        backend=fake_backend,
        config=small_config,
        output_dir=tmp_path,
    )

    assert result["score_a_mean"] == 0.5
    assert result["score_b_mean"] == 0.0
    assert (tmp_path / "compare_summary.json").exists()
    assert (tmp_path / "significance_summary.json").exists()


def test_compare_dspy_programs_writes_summary(monkeypatch, tmp_path: Path, small_config: TopoPromptConfig):
    dspy_program_a = SimpleNamespace(_topoprompt_program_id="dspy_mipro_predict")
    dspy_program_b = SimpleNamespace(_topoprompt_program_id="dspy_gepa_predict")
    task_spec = TaskSpec(task_id="task", description="Toy task.")
    examples = [
        Example(example_id="ex1", input={"question": "Q1"}, target="1"),
        Example(example_id="ex2", input={"question": "Q2"}, target="2"),
    ]

    scores = {
        "dspy_mipro_predict": {
            "score": 0.5,
            "mean_invocations": 1.0,
            "mean_tokens": 0.0,
            "parse_failure_rate": 0.0,
            "traces": [
                {"example_id": "ex1", "final_output": "1", "correctness": 1.0},
                {"example_id": "ex2", "final_output": "0", "correctness": 0.0},
            ],
        },
        "dspy_gepa_predict": {
            "score": 0.0,
            "mean_invocations": 1.0,
            "mean_tokens": 0.0,
            "parse_failure_rate": 0.0,
            "traces": [
                {"example_id": "ex1", "final_output": "0", "correctness": 0.0},
                {"example_id": "ex2", "final_output": "0", "correctness": 0.0},
            ],
        },
    }

    monkeypatch.setattr(
        "topoprompt.eval.dspy_baselines.evaluate_dspy_program_on_examples",
        lambda **kwargs: scores[getattr(kwargs["program"], "_topoprompt_program_id")],
    )

    result = compare_dspy_programs(
        program_a=dspy_program_a,
        program_b=dspy_program_b,
        task_spec=task_spec,
        examples=examples,
        metric_fn=lambda pred, ex: 1.0 if pred == ex.target else 0.0,
        config=small_config,
        output_dir=tmp_path,
    )

    assert result["score_a_mean"] == 0.5
    assert result["score_b_mean"] == 0.0
    assert (tmp_path / "compare_summary.json").exists()
    assert (tmp_path / "significance_summary.json").exists()
