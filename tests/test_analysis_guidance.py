from __future__ import annotations

from topoprompt.compiler.analyzer import stabilize_task_analysis
from topoprompt.compiler.search import _build_compile_warnings
from topoprompt.schemas import Example, TaskAnalysis, TaskSpec


def test_stabilize_task_analysis_relabels_low_confidence_math_tasks():
    task_spec = TaskSpec(
        task_id="gsm8k_like",
        description="Solve grade-school math word problems. Return only the final numeric answer.",
        input_schema={"question": "str"},
        output_schema={"type": "string"},
    )
    analysis = TaskAnalysis(
        task_family="mixed",
        output_format="short_answer",
        needs_reasoning=True,
        needs_verification=True,
        analyzer_confidence=0.46,
        initial_seed_templates=["direct_finalize"],
    )

    refined = stabilize_task_analysis(
        task_spec=task_spec,
        examples=[
            Example(
                example_id="ex1",
                input={"question": "If Alice has 2 apples and gets 3 more, how many apples does she have?"},
                target="5",
            )
        ],
        metric_name="numeric",
        analysis=analysis,
    )

    assert refined.task_family == "math_reasoning"
    assert refined.output_format == "numeric"


def test_stabilize_task_analysis_prefers_bullets_for_bullet_summaries():
    task_spec = TaskSpec(
        task_id="cnn_like",
        description="Summarize the news article into concise bullet-point highlights.",
        input_schema={"article": "str"},
        output_schema={"type": "string"},
    )
    analysis = TaskAnalysis(
        task_family="summarization",
        output_format="paragraph",
        needs_reasoning=True,
        needs_verification=True,
        analyzer_confidence=0.46,
        initial_seed_templates=["direct_finalize"],
    )

    refined = stabilize_task_analysis(
        task_spec=task_spec,
        examples=[
            Example(
                example_id="ex1",
                input={"article": "A long article about a merger and its market impact."},
                target="The companies announced a merger.",
            )
        ],
        metric_name="exact_match",
        analysis=analysis,
    )

    assert refined.task_family == "summarization"
    assert refined.output_format == "bullet_points"


def test_compile_warnings_flag_exact_match_for_free_form_bullet_summaries():
    task_spec = TaskSpec(
        task_id="cnn_like",
        description="Summarize the news article into concise bullet-point highlights.",
        input_schema={"article": "str"},
        output_schema={"type": "string"},
    )
    analysis = TaskAnalysis(
        task_family="summarization",
        output_format="bullet_points",
        analyzer_confidence=0.8,
        initial_seed_templates=["direct_finalize"],
    )
    examples = [
        Example(
            example_id="ex1",
            input={"article": "Article one"},
            target="The company reported stronger earnings and raised guidance for the year.",
        ),
        Example(
            example_id="ex2",
            input={"article": "Article two"},
            target="Officials said evacuation orders remained in effect after the storm passed.",
        ),
    ]

    warnings = _build_compile_warnings(
        task_spec=task_spec,
        examples=examples,
        metric_name="exact_match",
        analysis=analysis,
    )

    assert len(warnings) == 2
    assert "strict normalized string equality" in warnings[0]
    assert "bullet-point output" in warnings[1]
