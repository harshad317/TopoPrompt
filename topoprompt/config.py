from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


COMPILE_BUDGET_PHASE_FIELDS = (
    "analyzer_budget_calls",
    "seed_budget_calls",
    "screening_budget_calls",
    "narrowing_budget_calls",
    "confirmation_budget_calls",
    "reserve_budget_calls",
)


class ModelConfig(BaseModel):
    name: str = "gpt-4.1-mini"
    repair_model: str | None = None
    embedding_model: str | None = "text-embedding-3-small"
    temperature: float = 0.0
    max_output_tokens: int = 800


class CompileConfig(BaseModel):
    budget_unit: str = "llm_invocation"
    total_budget_calls: int = 500
    analyzer_budget_calls: int = 20
    seed_budget_calls: int = 60
    screening_budget_calls: int = 240
    narrowing_budget_calls: int = 110
    confirmation_budget_calls: int = 50
    reserve_budget_calls: int = 20
    beam_width: int = 8
    max_rounds: int = 6
    max_candidates_per_parent: int = 6
    min_structural_families: int = 2
    confirm_top_k: int = 3
    early_stop_min_improvement: float = 0.005
    early_stop_patience_rounds: int = 2
    target_score: float | None = None
    screening_examples: int = 8
    narrowing_examples: int = 32
    confirmation_examples: int = 64
    min_screening_examples_for_fallback: int = 4
    min_narrowing_examples_for_fallback: int = 8
    min_confirmation_examples_for_fallback: int = 8
    always_include_direct_seed: bool = True
    reseed_margin: float = 0.04
    llm_edit_proposals_per_parent: int = 2
    llm_edit_proposals_enabled: bool = True

    def phase_budget_total(self) -> int:
        return sum(getattr(self, field_name) for field_name in COMPILE_BUDGET_PHASE_FIELDS)

    def rebalance_phase_budgets(self, total_budget_calls: int | None = None) -> None:
        target_total = self.total_budget_calls if total_budget_calls is None else total_budget_calls
        current_values = [getattr(self, field_name) for field_name in COMPILE_BUDGET_PHASE_FIELDS]
        current_total = sum(current_values)
        self.total_budget_calls = target_total
        if current_total <= 0:
            even_split = [0 for _ in COMPILE_BUDGET_PHASE_FIELDS]
            if COMPILE_BUDGET_PHASE_FIELDS:
                even_split[0] = target_total
            for field_name, value in zip(COMPILE_BUDGET_PHASE_FIELDS, even_split, strict=False):
                setattr(self, field_name, value)
            return

        scaled = [value * target_total / current_total for value in current_values]
        rebalanced = [int(value) for value in scaled]
        remaining = target_total - sum(rebalanced)
        fractional_order = sorted(
            range(len(COMPILE_BUDGET_PHASE_FIELDS)),
            key=lambda index: (scaled[index] - rebalanced[index], -index),
            reverse=True,
        )
        for index in fractional_order[:remaining]:
            rebalanced[index] += 1
        for field_name, value in zip(COMPILE_BUDGET_PHASE_FIELDS, rebalanced, strict=False):
            setattr(self, field_name, value)


class ProgramConfig(BaseModel):
    max_nodes: int = 7
    max_route_nodes: int = 2
    max_branch_fanout: int = 3
    max_subquestions_per_decompose: int = 3
    allow_loops: bool = False
    prompt_token_cap: int = 1500


class DataConfig(BaseModel):
    official_test_split: bool = True
    compile_fraction_if_no_official_split: float = 0.60
    validation_fraction_if_no_official_split: float = 0.20
    test_fraction_if_no_official_split: float = 0.20
    fewshot_pool_fraction_of_compile: float = 0.10
    fewshot_pool_max_examples: int = 32
    representative_examples_for_analysis: int = 5


class ObjectiveConfig(BaseModel):
    alpha_cost: float = 0.05
    beta_complexity: float = 0.10
    gamma_parse_failure: float = 0.20
    delta_partial_coverage: float = 0.25
    epsilon_mode: str = "variance_adaptive"
    epsilon_floor: float = 0.01
    epsilon_z: float = 1.0


class RuntimeConfig(BaseModel):
    parser_retry_limit: int = 1
    parser_repair_limit: int = 1
    cache_enabled: bool = True
    cache_dir: str = ".topoprompt_cache"
    route_mode_default: str = "self_route_llm"


class TopoPromptConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    compile: CompileConfig = Field(default_factory=CompileConfig)
    program: ProgramConfig = Field(default_factory=ProgramConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> TopoPromptConfig:
    if path is None:
        default_path = Path(__file__).resolve().parent.parent / "configs" / "topoprompt_v1.yaml"
        path = default_path
    path = Path(path)
    data: dict[str, Any] = {}
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
    if overrides:
        data = _deep_merge_dicts(data, overrides)
    return TopoPromptConfig.model_validate(data)


def _deep_merge_dicts(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
