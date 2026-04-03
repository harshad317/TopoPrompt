from __future__ import annotations

from topoprompt.eval.metrics import bbh_metric, ifeval_metric
from topoprompt.schemas import Example


def test_bbh_metric_uses_exact_match_for_free_form_examples():
    example = Example(example_id="bbh_1", input={"prompt": "not (True) and True is"}, target="False")

    assert bbh_metric("False", example) == 1.0
    assert bbh_metric("True", example) == 0.0


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
