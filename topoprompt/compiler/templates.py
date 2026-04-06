from __future__ import annotations

import json
from typing import Iterable

from topoprompt.schemas import Example, NodeType, ProgramNode, PromptModule, RouteSpec, TaskAnalysis


def default_prompt_modules(
    node_type: NodeType,
    *,
    task_analysis: TaskAnalysis | None = None,
    branch_label: str | None = None,
    output_format: str | None = None,
    fewshot_examples: Iterable[Example] | None = None,
) -> list[PromptModule]:
    task_family = task_analysis.task_family if task_analysis else "general"

    # Pull task-specific instructions from the analyzer when available.
    # node_instructions is keyed by node type value string (e.g. "direct",
    # "solve").  When confidence is sufficient the analyzer populates this with
    # concise, task-grounded instruction text so seeds start from a meaningful
    # baseline rather than generic placeholders.
    _node_instrs: dict[str, str] = (
        task_analysis.node_instructions if task_analysis and task_analysis.node_instructions else {}
    )

    def _instr(node_key: str, fallback: str) -> str:
        """Return task-specific instruction text if available, else the fallback."""
        text = _node_instrs.get(node_key, "").strip()
        return text if text else fallback

    modules: dict[NodeType, list[PromptModule]] = {
        NodeType.DIRECT: [
            PromptModule(
                role="instruction",
                text=_instr("direct", f"Answer the {task_family} task accurately."),
            ),
            PromptModule(role="format", text="Return only the candidate answer."),
        ],
        NodeType.PLAN: [
            PromptModule(
                role="instruction",
                text=_instr("plan", "Produce a concise plan before solving."),
            ),
            PromptModule(role="reasoning", text="Keep the plan short and actionable."),
        ],
        NodeType.DECOMPOSE: [
            PromptModule(
                role="instruction",
                text=_instr("decompose", "Break the task into a small number of subquestions."),
            ),
            PromptModule(role="reasoning", text="Prefer at most three subquestions."),
        ],
        NodeType.SOLVE: [
            PromptModule(
                role="instruction",
                text=_instr("solve", f"Solve the {task_family} task carefully."),
            ),
            PromptModule(role="reasoning", text="Work through the logic step by step, but keep it concise."),
            PromptModule(role="format", text="Output a candidate answer and a short rationale."),
        ],
        NodeType.VERIFY: [
            PromptModule(
                role="verification",
                text=_instr(
                    "verify",
                    "Check whether the candidate answer satisfies all task constraints.",
                ),
            ),
            PromptModule(role="format", text="Return PASS or FAIL with a short explanation."),
        ],
        NodeType.CRITIQUE: [
            PromptModule(
                role="instruction",
                text=_instr("critique", "Identify flaws in the current candidate answer."),
            ),
        ],
        NodeType.ROUTE: [
            PromptModule(
                role="instruction",
                text=_instr("route", "Select the best execution branch for this input."),
            ),
            PromptModule(role="format", text="Return branch, confidence, and reason as JSON."),
        ],
        NodeType.FORMAT: [
            PromptModule(
                role="format",
                text=_instr("format", f"Format the answer as {output_format or 'the requested output'}"),
            ),
        ],
        NodeType.FINALIZE: [
            PromptModule(
                role="format",
                text=_instr("finalize", f"Emit the final answer as {output_format or 'the requested output'}"),
            ),
        ],
    }
    result = list(modules[node_type])
    if branch_label:
        result.append(PromptModule(role="instruction", text=f"This node serves the '{branch_label}' branch."))
    if fewshot_examples:
        result.append(fewshot_module_from_examples(list(fewshot_examples)))
    return result


def fewshot_module_from_examples(examples: list[Example]) -> PromptModule:
    rows = [
        {"input": example.input, "target": example.target}
        for example in examples
    ]
    text = "Use these examples as style guidance:\n" + json.dumps(rows, sort_keys=True)
    return PromptModule(role="fewshot", text=text, tags=["fewshot"], origin="compiler")


def output_schema_for_node(node_type: NodeType) -> dict:
    schemas = {
        NodeType.DIRECT: {
            "type": "object",
            # reasoning appears FIRST so the model must think before committing
            # to candidate_answer — exactly like DSPy ChainOfThought.
            "properties": {
                "reasoning": {"type": "string", "description": "Step-by-step reasoning before the answer."},
                "candidate_answer": {"type": "string"},
            },
            "required": ["reasoning", "candidate_answer"],
        },
        NodeType.PLAN: {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why this plan addresses the task."},
                "plan": {"type": "string"},
            },
            "required": ["reasoning", "plan"],
        },
        NodeType.DECOMPOSE: {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why these subquestions cover the task."},
                "subquestions": {"type": "array", "items": {"type": "string"}},
                "subquestion_answers": {"type": "array", "items": {"type": "string"}},
                "decomposition_context": {"type": "string"},
            },
            "required": ["reasoning", "subquestions"],
        },
        NodeType.SOLVE: {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Step-by-step reasoning before the answer."},
                "candidate_answer": {"type": "string"},
            },
            "required": ["reasoning", "candidate_answer"],
        },
        NodeType.VERIFY: {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Step-by-step verification logic."},
                "verification_result": {"type": "string"},
                "explanation": {"type": ["string", "null"]},
            },
            "required": ["reasoning", "verification_result"],
        },
        NodeType.CRITIQUE: {
            "type": "object",
            "properties": {"critique": {"type": "string"}},
            "required": ["critique"],
        },
        NodeType.ROUTE: {
            "type": "object",
            "properties": {
                "branch": {"type": "string"},
                "confidence": {"type": ["number", "null"]},
                "reason": {"type": ["string", "null"]},
            },
            "required": ["branch"],
        },
        NodeType.FORMAT: {
            "type": "object",
            "properties": {"formatted_answer": {"type": "string"}},
            "required": ["formatted_answer"],
        },
        NodeType.FINALIZE: {
            "type": "object",
            "properties": {"final_answer": {"type": "string"}},
            "required": ["final_answer"],
        },
    }
    return schemas[node_type]


def default_output_keys(node_type: NodeType) -> list[str]:
    mapping = {
        NodeType.DIRECT: ["candidate_answer"],
        NodeType.PLAN: ["plan"],
        NodeType.DECOMPOSE: ["subquestions", "subquestion_answers", "decomposition_context"],
        NodeType.SOLVE: ["candidate_answer"],
        NodeType.VERIFY: ["verification_result"],
        NodeType.CRITIQUE: ["critique"],
        NodeType.ROUTE: ["branch", "confidence", "reason"],
        NodeType.FORMAT: ["formatted_answer"],
        NodeType.FINALIZE: ["final_answer"],
    }
    return mapping[node_type]


def default_execution_mode(node_type: NodeType) -> str:
    if node_type == NodeType.DECOMPOSE:
        return "decompose_macro"
    if node_type == NodeType.FINALIZE:
        return "pass_through"
    return "llm_call"


def create_node(
    node_id: str,
    node_type: NodeType,
    *,
    name: str | None = None,
    input_keys: list[str] | None = None,
    output_keys: list[str] | None = None,
    route_spec: RouteSpec | None = None,
    config: dict | None = None,
    prompt_modules: list[PromptModule] | None = None,
    task_analysis: TaskAnalysis | None = None,
    fewshot_examples: Iterable[Example] | None = None,
) -> ProgramNode:
    return ProgramNode(
        node_id=node_id,
        node_type=node_type,
        name=name or node_type.value,
        input_keys=input_keys or [],
        output_keys=output_keys or default_output_keys(node_type),
        execution_mode=default_execution_mode(node_type),
        expected_output_schema=output_schema_for_node(node_type),
        prompt_modules=prompt_modules or default_prompt_modules(
            node_type,
            task_analysis=task_analysis,
            fewshot_examples=fewshot_examples,
        ),
        route_spec=route_spec,
        config=config or {},
    )
