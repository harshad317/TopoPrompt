from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Iterable

import networkx as nx

from topoprompt.schemas import ProgramEdge, ProgramNode, PromptProgram


def build_graph(program: PromptProgram) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in program.nodes:
        graph.add_node(node.node_id, node=node)
    for edge in program.edges:
        graph.add_edge(edge.source, edge.target, label=edge.label)
    return graph


def topological_nodes(program: PromptProgram) -> list[ProgramNode]:
    graph = build_graph(program)
    order = list(nx.topological_sort(graph))
    nodes = program.node_map()
    return [nodes[node_id] for node_id in order]


def outgoing_edges(program: PromptProgram) -> dict[str, list[ProgramEdge]]:
    grouped: dict[str, list[ProgramEdge]] = defaultdict(list)
    for edge in program.edges:
        grouped[edge.source].append(edge)
    return grouped


def incoming_edges(program: PromptProgram) -> dict[str, list[ProgramEdge]]:
    grouped: dict[str, list[ProgramEdge]] = defaultdict(list)
    for edge in program.edges:
        grouped[edge.target].append(edge)
    return grouped


def prompt_token_count(program: PromptProgram) -> int:
    return sum(len(module.text.split()) for node in program.nodes for module in node.prompt_modules)


def branch_count(program: PromptProgram) -> int:
    return sum(1 for edge in program.edges if edge.label)


def family_signature(program: PromptProgram) -> str:
    node_types = "-".join(node.node_type.value for node in topological_nodes(program))
    route_nodes = sum(1 for node in program.nodes if node.node_type.value == "route")
    fanouts = []
    outs = outgoing_edges(program)
    for node in program.nodes:
        if node.node_type.value == "route":
            fanouts.append(str(len(outs.get(node.node_id, []))))
    fanout_sig = ",".join(sorted(fanouts)) if fanouts else "0"
    flags = Counter(node.node_type.value for node in program.nodes)
    return f"{node_types}|routes={route_nodes}|fanout={fanout_sig}|verify={int(flags['verify']>0)}|decompose={int(flags['decompose']>0)}"


def topology_fingerprint(program: PromptProgram) -> str:
    payload = {
        "nodes": [
            {
                "id": node.node_id,
                "type": node.node_type.value,
                "outputs": node.output_keys,
                "route": node.route_spec.model_dump(mode="json") if node.route_spec else None,
                "modules": [
                    {
                        "role": module.role,
                        "text": " ".join(module.text.split()),
                        "tags": sorted(module.tags),
                    }
                    for module in node.prompt_modules
                ],
            }
            for node in sorted(program.nodes, key=lambda item: item.node_id)
        ],
        "edges": [
            {"source": edge.source, "target": edge.target, "label": edge.label}
            for edge in sorted(program.edges, key=lambda item: (item.source, item.target, item.label or ""))
        ],
        "entry": program.entry_node_id,
        "finalize": program.finalize_node_id,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def clone_program(program: PromptProgram, *, program_id: str | None = None) -> PromptProgram:
    cloned = PromptProgram.model_validate(program.model_dump(mode="json"))
    if program_id:
        cloned.program_id = program_id
    return cloned


def replace_edges(program: PromptProgram, edges: Iterable[ProgramEdge]) -> None:
    program.edges = list(edges)

