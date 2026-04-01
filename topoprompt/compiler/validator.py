from __future__ import annotations

from collections import Counter

import networkx as nx

from topoprompt.config import ProgramConfig
from topoprompt.ir import build_graph, outgoing_edges, topological_nodes
from topoprompt.schemas import NodeType, PromptProgram


class ProgramValidationError(ValueError):
    pass


def validate_program(program: PromptProgram, config: ProgramConfig) -> None:
    graph = build_graph(program)
    node_ids = {node.node_id for node in program.nodes}
    if len(node_ids) != len(program.nodes):
        raise ProgramValidationError("Node IDs must be unique.")
    if program.entry_node_id not in node_ids:
        raise ProgramValidationError("Entry node does not exist.")
    if program.finalize_node_id not in node_ids:
        raise ProgramValidationError("Finalize node does not exist.")
    if len(program.nodes) > config.max_nodes:
        raise ProgramValidationError("Program exceeds max node count.")

    route_nodes = [node for node in program.nodes if node.node_type == NodeType.ROUTE]
    if len(route_nodes) > config.max_route_nodes:
        raise ProgramValidationError("Program exceeds max route nodes.")

    if not config.allow_loops and not nx.is_directed_acyclic_graph(graph):
        raise ProgramValidationError("Program graph must be acyclic.")

    finalize_nodes = [node for node in program.nodes if node.node_type == NodeType.FINALIZE]
    if len(finalize_nodes) != 1 or finalize_nodes[0].node_id != program.finalize_node_id:
        raise ProgramValidationError("Program must have exactly one finalize node.")

    if sum(1 for node in program.nodes if graph.in_degree(node.node_id) == 0) != 1:
        raise ProgramValidationError("Program must have exactly one entry node.")

    reachable = nx.descendants(graph, program.entry_node_id) | {program.entry_node_id}
    if reachable != node_ids:
        raise ProgramValidationError("Program contains orphan or unreachable nodes.")

    reverse_graph = graph.reverse(copy=False)
    can_reach_finalize = nx.descendants(reverse_graph, program.finalize_node_id) | {program.finalize_node_id}
    if can_reach_finalize != node_ids:
        raise ProgramValidationError("Every node must feed forward toward finalize.")

    outs = outgoing_edges(program)
    for node in route_nodes:
        if node.route_spec is None:
            raise ProgramValidationError("Route node missing route spec.")
        outgoing = outs.get(node.node_id, [])
        if len(outgoing) > config.max_branch_fanout:
            raise ProgramValidationError("Route node exceeds max branch fanout.")
        outgoing_labels = {edge.label for edge in outgoing}
        branch_labels = set(node.route_spec.branch_labels)
        if branch_labels != outgoing_labels:
            raise ProgramValidationError("Route branch labels must match outgoing edge labels.")
        if node.route_spec.fallback_branch and node.route_spec.fallback_branch not in branch_labels:
            raise ProgramValidationError("Fallback branch must be part of the route spec.")

    available_keys = {"task_input"}
    node_counts = Counter(node.node_type.value for node in program.nodes)
    if node_counts["direct"] + node_counts["solve"] == 0:
        raise ProgramValidationError("Program must contain at least one answer-producing node.")

    for node in topological_nodes(program):
        if node.input_keys and not set(node.input_keys).issubset(available_keys):
            missing = set(node.input_keys) - available_keys
            raise ProgramValidationError(f"Node {node.node_id} uses unavailable inputs: {sorted(missing)}")
        available_keys.update(node.output_keys)


def is_valid_program(program: PromptProgram, config: ProgramConfig) -> bool:
    try:
        validate_program(program, config)
    except ProgramValidationError:
        return False
    return True

