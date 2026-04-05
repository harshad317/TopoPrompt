from __future__ import annotations

from topoprompt.backends.llm_client import FakeBackend
from topoprompt.compiler.analyzer import analyze_task, heuristic_task_analysis
from topoprompt.schemas import Example, TaskSpec


def test_heuristic_task_analysis_recognizes_classification_tasks():
    task_spec = TaskSpec(
        task_id="sentiment",
        description="Classify each review as positive or negative.",
    )
    examples = [
        Example(example_id="pos", input={"text": "I loved this movie."}, target="positive"),
        Example(example_id="neg", input={"text": "This was painfully slow."}, target="negative"),
    ]

    analysis = heuristic_task_analysis(task_spec=task_spec, examples=examples, metric_name="exact_match")

    assert analysis.task_family == "classification"
    assert analysis.output_format == "label"
    assert analysis.initial_seed_templates[0] == "direct_finalize"
    assert "plan_solve_verify_finalize" not in analysis.initial_seed_templates


def test_heuristic_task_analysis_recognizes_extraction_tasks():
    task_spec = TaskSpec(
        task_id="entities",
        description="Extract people and organizations from the passage and return JSON.",
    )
    examples = [
        Example(
            example_id="ent_1",
            input={"prompt": "Ada joined Analytical Engines in London."},
            target={"people": ["Ada"], "organizations": ["Analytical Engines"]},
        )
    ]

    analysis = heuristic_task_analysis(task_spec=task_spec, examples=examples, metric_name="exact_match")

    assert analysis.task_family == "extraction"
    assert analysis.output_format == "json"
    assert "format_finalize" in analysis.initial_seed_templates
    assert "critique_revise_finalize" not in analysis.initial_seed_templates


def test_heuristic_task_analysis_prefers_classification_when_targets_include_rationales():
    task_spec = TaskSpec(
        task_id="nli",
        description="Decide whether the hypothesis is entailment, contradiction, or neutral.",
    )
    examples = [
        Example(
            example_id="nli_1",
            input={"premise": "All dogs bark.", "hypothesis": "Dogs bark."},
            target="entailment because the hypothesis follows directly from the premise.",
        ),
        Example(
            example_id="nli_2",
            input={"premise": "No birds are mammals.", "hypothesis": "Birds are mammals."},
            target="contradiction because the premise explicitly rules it out.",
        ),
    ]

    analysis = heuristic_task_analysis(task_spec=task_spec, examples=examples, metric_name="exact_match")

    assert analysis.task_family == "classification"
    assert analysis.output_format == "label"
    assert analysis.initial_seed_templates == ["direct_finalize"]


def test_fake_backend_analysis_uses_broader_code_priors(small_config):
    task_spec = TaskSpec(
        task_id="code_task",
        description="Write a Python function that reverses a string.",
    )
    examples = [
        Example(
            example_id="code_1",
            input={"prompt": "Implement reverse_string(text)."},
            target="def reverse_string(text):\n    return text[::-1]",
        )
    ]

    analysis = analyze_task(
        task_spec=task_spec,
        examples=examples,
        metric_name="exact_match",
        backend=FakeBackend(),
        config=small_config,
    )

    assert analysis.task_family == "code"
    assert "critique_revise_finalize" in analysis.initial_seed_templates
    assert "solve_verify_finalize" in analysis.initial_seed_templates
