from __future__ import annotations

from typing import Any

from topoprompt.schemas import ProgramNode, RouteSpec


def choose_rule_route(node: ProgramNode, state: dict[str, Any]) -> tuple[str, float]:
    route_spec = node.route_spec
    if route_spec is None:
        raise ValueError("Route node is missing route_spec.")
    keyword_rules = node.config.get("keyword_rules", {})
    haystack = str(state.get("task_input", "")).lower()
    if isinstance(state.get("task_input"), dict):
        haystack = " ".join(str(value).lower() for value in state["task_input"].values())
    for keyword, branch in keyword_rules.items():
        if keyword.lower() in haystack and branch in route_spec.branch_labels:
            return branch, 1.0
    fallback = route_spec.fallback_branch or route_spec.branch_labels[0]
    return fallback, 0.5


def resolve_route_choice(route_spec: RouteSpec, parsed_output: dict[str, Any]) -> tuple[str, float]:
    branch = parsed_output.get("branch")
    confidence = float(parsed_output.get("confidence") or 0.0)
    if branch in route_spec.branch_labels:
        return branch, confidence
    fallback = route_spec.fallback_branch or route_spec.branch_labels[0]
    return fallback, confidence

