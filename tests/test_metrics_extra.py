from __future__ import annotations

from topoprompt.eval.metrics import bbh_metric, ifeval_metric, numeric_metric
from topoprompt.schemas import Example


def test_bbh_metric_uses_exact_match_for_free_form_examples():
    example = Example(example_id="bbh_1", input={"prompt": "not (True) and True is"}, target="False")

    assert bbh_metric("False", example) == 1.0
    assert bbh_metric("True", example) == 0.0
    assert bbh_metric("The answer is False.", example) == 1.0


def test_bbh_metric_accepts_multiple_choice_labels_and_option_text():
    prompt = (
        "Today is the second day of the third month of 1966. What is the date one week ago?\n"
        "Options:\n"
        "(A) 02/26/1966\n"
        "(B) 01/13/1966\n"
        "(C) 02/02/1966\n"
    )
    example = Example(
        example_id="bbh_mc",
        input={
            "prompt": prompt,
            "choices": [
                {"label": "A", "text": "02/26/1966"},
                {"label": "B", "text": "01/13/1966"},
                {"label": "C", "text": "02/02/1966"},
            ],
        },
        target="(B)",
        metadata={"bbh_task": "date_understanding"},
    )

    assert bbh_metric("I choose B.", example) == 1.0
    assert bbh_metric("01/13/1966", example) == 1.0
    assert bbh_metric("(C)", example) == 0.0


def test_bbh_metric_handles_symbolic_dyck_outputs():
    example = Example(
        example_id="bbh_dyck",
        input={"prompt": "Complete the bracket sequence."},
        target="] ] >",
        metadata={"bbh_task": "dyck_languages"},
    )

    assert bbh_metric("The answer is ] ] >.", example) == 1.0
    assert bbh_metric("] )", example) == 0.0


def test_ifeval_metric_scores_instruction_metadata():
    example = Example(
        example_id="ifeval_1",
        input={"prompt": "Answer with exactly two parts separated by ****** and include <<title>>."},
        metadata={
            "instruction_id_list": [
                "combination:two_responses",
                "detectable_format:title",
                "punctuation:no_comma",
            ],
            "instruction_kwargs": [{}, {}, {}],
        },
    )

    assert ifeval_metric("<<Title>> one******two", example) == 1.0
    assert ifeval_metric("<<Title>>, one******two", example) == 2 / 3


def test_ifeval_metric_normalizes_language_codes_and_names():
    example = Example(
        example_id="ifeval_lang",
        input={"prompt": "Respond in French."},
        metadata={
            "instruction_id_list": ["language:response_language"],
            "instruction_kwargs": [{"language": "French"}],
        },
    )

    assert ifeval_metric("Bonjour tout le monde", example) == 1.0
    assert ifeval_metric("Hello world", example) == 0.0


def test_numeric_metric_prefers_answer_marker_over_trailing_number():
    example = Example(
        example_id="numeric_marker",
        input={"question": "dummy"},
        target="42",
    )

    assert numeric_metric("Answer: 42. Sanity check value: 7.", example) == 1.0


def test_numeric_metric_supports_first_number_strategy_from_metadata():
    example = Example(
        example_id="numeric_first",
        input={"question": "Convert 42 inches to feet."},
        target="3.5",
        metadata={"prediction_numeric_position": "first"},
    )

    assert numeric_metric("3.5 feet is the answer; the original value was 42 inches.", example) == 1.0
