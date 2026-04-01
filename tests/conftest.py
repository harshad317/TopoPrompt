from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topoprompt.backends.llm_client import FakeBackend
from topoprompt.config import TopoPromptConfig
from topoprompt.eval.datasets import load_examples_from_jsonl
from topoprompt.schemas import TaskSpec


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def small_config() -> TopoPromptConfig:
    return TopoPromptConfig.model_validate(
        {
            "model": {"name": "fake-model", "repair_model": "fake-model"},
            "compile": {
                "total_budget_calls": 120,
                "analyzer_budget_calls": 8,
                "seed_budget_calls": 20,
                "screening_budget_calls": 40,
                "narrowing_budget_calls": 30,
                "confirmation_budget_calls": 12,
                "reserve_budget_calls": 10,
                "beam_width": 4,
                "max_rounds": 2,
                "max_candidates_per_parent": 4,
                "confirm_top_k": 2,
                "screening_examples": 4,
                "narrowing_examples": 6,
                "confirmation_examples": 4,
                "llm_edit_proposals_enabled": False,
            },
            "runtime": {"cache_enabled": False},
        }
    )


@pytest.fixture
def gsm8k_examples():
    path = Path(__file__).resolve().parent / "fixtures" / "smoke" / "gsm8k_examples.jsonl"
    return load_examples_from_jsonl(path)


@pytest.fixture
def simple_task_spec() -> TaskSpec:
    return TaskSpec(
        task_id="arith_task",
        description="Answer simple arithmetic questions accurately.",
        input_schema={"question": "str"},
        output_schema={"type": "string"},
    )
