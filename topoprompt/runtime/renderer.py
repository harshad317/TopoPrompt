from __future__ import annotations

import json
from typing import Any

from topoprompt.schemas import ProgramNode, TaskSpec


NODE_ROLE_TEXT = {
    "direct": "Produce a direct candidate answer.",
    "plan": "Produce a concise plan that helps solve the task.",
    "decompose": "Break the task into bounded subquestions.",
    "solve": "Solve the task carefully using the provided context.",
    "verify": "Check whether the candidate answer satisfies the task constraints.",
    "critique": "Critique the current candidate answer.",
    "route": "Choose the best branch for execution.",
    "format": "Format the result into the required schema.",
    "finalize": "Emit the final answer.",
}


def render_node_prompt(task_spec: TaskSpec, node: ProgramNode, state: dict[str, Any]) -> tuple[str, str]:
    system_modules = [module.text for module in node.prompt_modules if module.role == "system"]
    non_system_modules = [module for module in node.prompt_modules if module.role != "system"]
    system_prompt = "\n\n".join(system_modules) if system_modules else "You are executing one node inside TopoPrompt."

    local_state: dict[str, Any] = {}
    if node.input_keys:
        for key in node.input_keys:
            if key in state:
                local_state[key] = state[key]
    else:
        local_state["task_input"] = state.get("task_input", {})
    if "task_input" not in local_state and "task_input" in state:
        local_state["task_input"] = state["task_input"]

    parts = [
        f"Task ID: {task_spec.task_id}",
        f"Task Description:\n{task_spec.description}",
        f"Node Name: {node.name}",
        f"Node Type: {node.node_type.value}",
        f"Node Role:\n{NODE_ROLE_TEXT.get(node.node_type.value, node.node_type.value)}",
    ]
    if node.route_spec is not None:
        branch_lines = "\n".join(
            f"- {label}: {node.route_spec.branch_descriptions.get(label, '')}".strip()
            for label in node.route_spec.branch_labels
        )
        parts.append(f"Available Branches:\n{branch_lines}")
    if non_system_modules:
        parts.append(
            "Prompt Modules:\n"
            + "\n".join(f"- [{module.role}] {module.text}" for module in non_system_modules)
        )
    parts.append(f"Context JSON:\n{json.dumps(local_state, sort_keys=True)}")
    parts.append(f"Output JSON Schema:\n{json.dumps(node.expected_output_schema, sort_keys=True)}")
    parts.append("Formatting Guardrails:\n- Return strict JSON only.\n- Do not omit required fields.")
    user_prompt = "\n\n".join(parts)
    return system_prompt, user_prompt
