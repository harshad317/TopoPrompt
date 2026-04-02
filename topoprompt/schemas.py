from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class NodeType(str, Enum):
    DIRECT = "direct"
    PLAN = "plan"
    DECOMPOSE = "decompose"
    SOLVE = "solve"
    VERIFY = "verify"
    CRITIQUE = "critique"
    ROUTE = "route"
    FORMAT = "format"
    FINALIZE = "finalize"


class TaskSpec(BaseModel):
    task_id: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    task_family: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Example(BaseModel):
    example_id: str
    input: dict[str, Any]
    target: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptModule(BaseModel):
    role: Literal["system", "instruction", "reasoning", "verification", "format", "fewshot"]
    text: str
    tags: list[str] = Field(default_factory=list)
    origin: str = "generated"


class RouteCandidate(BaseModel):
    label: str
    description: str


class RouteSpec(BaseModel):
    mode: Literal["self_route_llm", "rule_route", "classifier_route"]
    branch_labels: list[str]
    branch_descriptions: dict[str, str] = Field(default_factory=dict)
    confidence_threshold: float | None = None
    fallback_branch: str | None = None


class ProgramNode(BaseModel):
    node_id: str
    node_type: NodeType
    name: str
    input_keys: list[str] = Field(default_factory=list)
    output_keys: list[str] = Field(default_factory=list)
    execution_mode: Literal["llm_call", "pass_through", "decompose_macro"] = "llm_call"
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    parser_id: str = "json"
    fallback_parser_id: str = "regex_then_repair"
    prompt_modules: list[PromptModule] = Field(default_factory=list)
    route_spec: RouteSpec | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ProgramEdge(BaseModel):
    source: str
    target: str
    label: str | None = None


class PromptProgram(BaseModel):
    program_id: str
    task_id: str
    nodes: list[ProgramNode]
    edges: list[ProgramEdge]
    entry_node_id: str
    finalize_node_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def node_map(self) -> dict[str, ProgramNode]:
        return {node.node_id: node for node in self.nodes}


class TaskAnalysis(BaseModel):
    task_family: str = "other"
    output_format: str = "short_answer"
    needs_reasoning: bool = False
    needs_verification: bool = False
    needs_decomposition: bool = False
    input_heterogeneity: str = "low"
    candidate_routes: list[RouteCandidate] = Field(default_factory=list)
    initial_seed_templates: list[str] = Field(default_factory=list)
    analyzer_confidence: float = 0.0
    rationale: str = ""


class CandidateEdit(BaseModel):
    edit_type: str
    target_node_id: str | None = None
    new_node_type: NodeType | None = None
    module_role: str | None = None
    branch_labels: list[str] | None = None
    rewrite_instruction: str | None = None
    reason: str = ""


class RouteDiagnostic(BaseModel):
    example_id: str
    route_node_id: str
    chosen_branch: str | None = None
    oracle_branch: str | None = None
    branch_scores: dict[str, float] = Field(default_factory=dict)
    regret: float | None = None
    confidence: float | None = None


class NodeExecutionTrace(BaseModel):
    node_id: str
    prompt_text: str = ""
    raw_output: str = ""
    parsed_output: Any | None = None
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int | None = None
    route_choice: str | None = None
    confidence: float | None = None
    cache_hit: bool = False
    repair_used: bool = False
    parse_error: str | None = None
    backend_request_id: str | None = None
    invocation_cost: int = 0


class ProgramExecutionTrace(BaseModel):
    example_id: str
    program_id: str
    node_traces: list[NodeExecutionTrace]
    final_output: Any | None = None
    correctness: float | None = None
    total_tokens: int = 0
    total_latency_ms: int = 0
    total_invocations: int = 0
    parse_failures: int = 0
    route_diagnostics: list[RouteDiagnostic] = Field(default_factory=list)


class CandidateArchiveRecord(BaseModel):
    program_id: str
    parent_id: str | None = None
    edit_applied: str | None = None
    topology_fingerprint: str
    family_signature: str
    screening_score: float | None = None
    narrowing_score: float | None = None
    confirmation_score: float | None = None
    search_score: float | None = None
    complexity: float = 0.0
    inference_cost: float = 0.0
    parse_failure_rate: float = 0.0
    round_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    program: PromptProgram
    topology_fingerprint: str
    family_signature: str
    stage: str
    parent_id: str | None = None
    edit_applied: str | None = None
    example_scores: list[float] = Field(default_factory=list)
    score: float = 0.0
    search_score: float = 0.0
    mean_invocations: float = 0.0
    mean_tokens: float = 0.0
    complexity: float = 0.0
    parse_failure_rate: float = 0.0
    traces: list[ProgramExecutionTrace] = Field(default_factory=list)
    route_diagnostics: list[RouteDiagnostic] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BudgetPhaseSpend(BaseModel):
    phase: str
    planned_calls: int
    spent_calls: int = 0


class CompileMetrics(BaseModel):
    best_program_id: str
    best_validation_score: float
    smallest_effective_program_id: str
    smallest_effective_score: float
    epsilon: float
    planned_budget_calls: int
    spent_budget_calls: int
    planned_budget_by_phase: list[BudgetPhaseSpend] = Field(default_factory=list)
    spent_budget_by_phase: list[BudgetPhaseSpend] = Field(default_factory=list)
    winning_topology_family: str
    beam_family_count_by_round: list[int] = Field(default_factory=list)
    parser_failure_rate: float = 0.0
    route_accuracy: float | None = None
    route_regret: float | None = None


class CompileArtifact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_spec: TaskSpec
    best_program_ir: PromptProgram | None = None
    program_ir: PromptProgram
    python_program: Any | None = None
    dspy_program: Any | None = None
    seed_programs: list[PromptProgram] = Field(default_factory=list)
    candidate_archive: list[CandidateArchiveRecord] = Field(default_factory=list)
    compile_trace: list[ProgramExecutionTrace] = Field(default_factory=list)
    metrics: CompileMetrics
    config: dict[str, Any] = Field(default_factory=dict)
    output_dir: str | None = None
