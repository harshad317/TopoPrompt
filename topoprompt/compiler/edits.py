from __future__ import annotations

from topoprompt.compiler.templates import create_node, fewshot_module_from_examples
from topoprompt.config import TopoPromptConfig
from topoprompt.ir import clone_program, outgoing_edges
from topoprompt.schemas import CandidateEdit, Example, NodeType, ProgramEdge, PromptModule, PromptProgram, RouteSpec, TaskAnalysis


def generate_heuristic_edits(
    *,
    program: PromptProgram,
    analysis: TaskAnalysis,
    config: TopoPromptConfig,
) -> list[CandidateEdit]:
    node_types = {node.node_type for node in program.nodes}
    edits: list[CandidateEdit] = []

    if analysis.needs_verification and NodeType.VERIFY not in node_types:
        target = _last_answer_node(program)
        if target:
            edits.append(CandidateEdit(edit_type="insert_verify_after", target_node_id=target))

    if analysis.needs_reasoning and NodeType.PLAN not in node_types:
        solve_nodes = [node.node_id for node in program.nodes if node.node_type == NodeType.SOLVE]
        if solve_nodes:
            edits.append(CandidateEdit(edit_type="insert_plan_before", target_node_id=solve_nodes[0]))
        else:
            direct_nodes = [node.node_id for node in program.nodes if node.node_type == NodeType.DIRECT]
            if direct_nodes:
                edits.append(
                    CandidateEdit(
                        edit_type="replace_node_type",
                        target_node_id=direct_nodes[0],
                        new_node_type=NodeType.SOLVE,
                    )
                )

    if analysis.input_heterogeneity in {"medium", "high"} and NodeType.ROUTE not in node_types:
        answer_target = _entry_answerish_node(program)
        if answer_target:
            edits.append(CandidateEdit(edit_type="split_with_route", target_node_id=answer_target))

    edits.append(CandidateEdit(edit_type="rewrite_prompt_module", rewrite_instruction="Make the instructions more task-specific."))
    edits.append(CandidateEdit(edit_type="change_finalize_format", rewrite_instruction="Return only the final answer."))

    if NodeType.ROUTE in node_types:
        route_node = next(node.node_id for node in program.nodes if node.node_type == NodeType.ROUTE)
        edits.append(CandidateEdit(edit_type="swap_branch_target", target_node_id=route_node))
        edits.append(CandidateEdit(edit_type="remove_route", target_node_id=route_node))

    if NodeType.VERIFY in node_types:
        verify_node = next(node.node_id for node in program.nodes if node.node_type == NodeType.VERIFY)
        edits.append(CandidateEdit(edit_type="delete_node", target_node_id=verify_node))

    if NodeType.VERIFY not in node_types:
        target = _last_answer_node(program)
        if target:
            edits.append(CandidateEdit(edit_type="add_node", target_node_id=target, new_node_type=NodeType.VERIFY))

    edits.append(CandidateEdit(edit_type="add_fewshot_module"))
    edits.append(CandidateEdit(edit_type="drop_fewshot_module"))

    return edits[: config.compile.max_candidates_per_parent]


def apply_edit(
    *,
    program: PromptProgram,
    edit: CandidateEdit,
    analysis: TaskAnalysis,
    fewshot_pool: list[Example] | None = None,
) -> PromptProgram:
    candidate = clone_program(program, program_id=f"{program.program_id}__{edit.edit_type}")
    node_map = candidate.node_map()

    match edit.edit_type:
        case "add_node":
            if edit.target_node_id is None or edit.new_node_type is None:
                raise ValueError("add_node requires target_node_id and new_node_type")
            new_node = create_node(
                node_id=_next_node_id(candidate, edit.new_node_type),
                node_type=edit.new_node_type,
                input_keys=["task_input", "candidate_answer"] if edit.new_node_type == NodeType.VERIFY else ["task_input"],
                task_analysis=analysis,
            )
            _insert_after(candidate, edit.target_node_id, new_node)
        case "delete_node":
            if edit.target_node_id is None:
                raise ValueError("delete_node requires target_node_id")
            _delete_node(candidate, edit.target_node_id)
        case "replace_node_type":
            if edit.target_node_id is None or edit.new_node_type is None:
                raise ValueError("replace_node_type requires target_node_id and new_node_type")
            node = node_map[edit.target_node_id]
            replacement = create_node(
                node_id=node.node_id,
                node_type=edit.new_node_type,
                input_keys=node.input_keys,
                route_spec=node.route_spec,
                config=node.config,
                task_analysis=analysis,
            )
            candidate.nodes = [replacement if existing.node_id == node.node_id else existing for existing in candidate.nodes]
        case "insert_verify_after":
            target = edit.target_node_id or _last_answer_node(candidate)
            new_node = create_node(
                node_id=_next_node_id(candidate, NodeType.VERIFY),
                node_type=NodeType.VERIFY,
                input_keys=["task_input", "candidate_answer"],
                task_analysis=analysis,
            )
            _insert_after(candidate, target, new_node)
        case "insert_plan_before":
            if edit.target_node_id is None:
                raise ValueError("insert_plan_before requires target_node_id")
            new_node = create_node(
                node_id=_next_node_id(candidate, NodeType.PLAN),
                node_type=NodeType.PLAN,
                input_keys=["task_input"],
                task_analysis=analysis,
            )
            _insert_before(candidate, edit.target_node_id, new_node)
            candidate.node_map()[edit.target_node_id].input_keys = list(
                dict.fromkeys([*candidate.node_map()[edit.target_node_id].input_keys, "plan"])
            )
        case "split_with_route":
            if edit.target_node_id is None:
                raise ValueError("split_with_route requires target_node_id")
            _split_with_route(candidate, edit.target_node_id, analysis)
        case "remove_route":
            if edit.target_node_id is None:
                raise ValueError("remove_route requires target_node_id")
            _remove_route(candidate, edit.target_node_id)
        case "swap_branch_target":
            if edit.target_node_id is None:
                raise ValueError("swap_branch_target requires target_node_id")
            _swap_branch_target(candidate, edit.target_node_id)
        case "rewrite_prompt_module":
            _rewrite_prompt_module(candidate, edit)
        case "add_fewshot_module":
            if fewshot_pool:
                _add_fewshot_module(candidate, fewshot_pool[: min(3, len(fewshot_pool))])
        case "drop_fewshot_module":
            _drop_fewshot_module(candidate)
        case "change_finalize_format":
            _change_finalize_format(candidate, edit.rewrite_instruction or "Return only the final answer.")
        case "remove_verify":
            verify_nodes = [node.node_id for node in candidate.nodes if node.node_type == NodeType.VERIFY]
            if verify_nodes:
                _delete_node(candidate, verify_nodes[0])
        case _:
            raise ValueError(f"Unsupported edit: {edit.edit_type}")

    candidate.metadata = dict(candidate.metadata)
    candidate.metadata["parent_id"] = program.program_id
    candidate.metadata["edit_applied"] = edit.model_dump(mode="json")
    return candidate


def _next_node_id(program: PromptProgram, node_type: NodeType) -> str:
    count = sum(1 for node in program.nodes if node.node_type == node_type) + 1
    return f"{node_type.value}_{count}"


def _insert_after(program: PromptProgram, target_node_id: str, new_node) -> None:
    outgoing = [edge for edge in program.edges if edge.source == target_node_id]
    kept = [edge for edge in program.edges if edge.source != target_node_id]
    kept.append(ProgramEdge(source=target_node_id, target=new_node.node_id))
    for edge in outgoing:
        kept.append(ProgramEdge(source=new_node.node_id, target=edge.target, label=edge.label))
    program.nodes.append(new_node)
    program.edges = kept


def _insert_before(program: PromptProgram, target_node_id: str, new_node) -> None:
    incoming = [edge for edge in program.edges if edge.target == target_node_id]
    kept = [edge for edge in program.edges if edge.target != target_node_id]
    if not incoming:
        program.entry_node_id = new_node.node_id
        kept.append(ProgramEdge(source=new_node.node_id, target=target_node_id))
    else:
        for edge in incoming:
            kept.append(ProgramEdge(source=edge.source, target=new_node.node_id, label=edge.label))
        kept.append(ProgramEdge(source=new_node.node_id, target=target_node_id))
    program.nodes.append(new_node)
    program.edges = kept


def _delete_node(program: PromptProgram, node_id: str) -> None:
    if node_id in {program.entry_node_id, program.finalize_node_id}:
        return
    incoming = [edge for edge in program.edges if edge.target == node_id]
    outgoing = [edge for edge in program.edges if edge.source == node_id]
    kept = [edge for edge in program.edges if edge.source != node_id and edge.target != node_id]
    for parent in incoming:
        for child in outgoing:
            kept.append(ProgramEdge(source=parent.source, target=child.target, label=parent.label or child.label))
    program.edges = _dedupe_edges(kept)
    program.nodes = [node for node in program.nodes if node.node_id != node_id]
    _prune_unreachable(program)


def _split_with_route(program: PromptProgram, target_node_id: str, analysis: TaskAnalysis) -> None:
    target = program.node_map()[target_node_id]
    alt_type = NodeType.SOLVE if target.node_type == NodeType.DIRECT else NodeType.DIRECT
    route = create_node(
        node_id=_next_node_id(program, NodeType.ROUTE),
        node_type=NodeType.ROUTE,
        input_keys=["task_input"],
        route_spec=RouteSpec(
            mode="self_route_llm",
            branch_labels=["direct", "solve"],
            branch_descriptions={
                "direct": "Direct answer branch.",
                "solve": "Reasoning branch.",
            },
            fallback_branch="direct",
        ),
        task_analysis=analysis,
    )
    alt_node = create_node(
        node_id=_next_node_id(program, alt_type),
        node_type=alt_type,
        input_keys=["task_input"],
        task_analysis=analysis,
    )
    incoming = [edge for edge in program.edges if edge.target == target_node_id]
    outgoing = [edge for edge in program.edges if edge.source == target_node_id]
    kept = [edge for edge in program.edges if edge.target != target_node_id and edge.source != target_node_id]

    if not incoming:
        program.entry_node_id = route.node_id
    else:
        for edge in incoming:
            kept.append(ProgramEdge(source=edge.source, target=route.node_id, label=edge.label))

    primary_label = "direct" if target.node_type == NodeType.DIRECT else "solve"
    alt_label = "solve" if primary_label == "direct" else "direct"
    kept.append(ProgramEdge(source=route.node_id, target=target.node_id, label=primary_label))
    kept.append(ProgramEdge(source=route.node_id, target=alt_node.node_id, label=alt_label))
    for edge in outgoing:
        kept.append(ProgramEdge(source=target.node_id, target=edge.target, label=edge.label))
        kept.append(ProgramEdge(source=alt_node.node_id, target=edge.target, label=edge.label))

    program.nodes.extend([route, alt_node])
    program.edges = _dedupe_edges(kept)


def _remove_route(program: PromptProgram, route_node_id: str) -> None:
    route_node = program.node_map()[route_node_id]
    if route_node.route_spec is None:
        return
    chosen_label = route_node.route_spec.fallback_branch or route_node.route_spec.branch_labels[0]
    incoming = [edge for edge in program.edges if edge.target == route_node_id]
    outgoing = [edge for edge in program.edges if edge.source == route_node_id and edge.label == chosen_label]
    kept = [edge for edge in program.edges if edge.source != route_node_id and edge.target != route_node_id]
    if outgoing:
        chosen_target = outgoing[0].target
        if not incoming:
            program.entry_node_id = chosen_target
        else:
            for edge in incoming:
                kept.append(ProgramEdge(source=edge.source, target=chosen_target, label=edge.label))
    program.edges = _dedupe_edges(kept)
    program.nodes = [node for node in program.nodes if node.node_id != route_node_id]
    _prune_unreachable(program)


def _swap_branch_target(program: PromptProgram, route_node_id: str) -> None:
    branch_edges = [edge for edge in program.edges if edge.source == route_node_id]
    if len(branch_edges) < 2:
        return
    first, second = branch_edges[0], branch_edges[1]
    first.target, second.target = second.target, first.target


def _rewrite_prompt_module(program: PromptProgram, edit: CandidateEdit) -> None:
    target_nodes = program.nodes if edit.target_node_id is None else [program.node_map()[edit.target_node_id]]
    instruction = edit.rewrite_instruction or "Make the prompt more task-specific."
    for node in target_nodes:
        for module in node.prompt_modules:
            if edit.module_role is None or module.role == edit.module_role:
                module.text = f"{module.text.rstrip()} {instruction}".strip()
                return


def _add_fewshot_module(program: PromptProgram, fewshot_examples: list[Example]) -> None:
    module = fewshot_module_from_examples(fewshot_examples)
    for node in program.nodes:
        if node.node_type in {NodeType.DIRECT, NodeType.SOLVE}:
            if all(existing.role != "fewshot" for existing in node.prompt_modules):
                node.prompt_modules.append(module)
                break


def _drop_fewshot_module(program: PromptProgram) -> None:
    for node in program.nodes:
        node.prompt_modules = [module for module in node.prompt_modules if module.role != "fewshot"]


def _change_finalize_format(program: PromptProgram, instruction: str) -> None:
    finalize = program.node_map()[program.finalize_node_id]
    replaced = False
    for module in finalize.prompt_modules:
        if module.role == "format":
            module.text = instruction
            replaced = True
            break
    if not replaced:
        finalize.prompt_modules.append(PromptModule(role="format", text=instruction, origin="edit"))


def _prune_unreachable(program: PromptProgram) -> None:
    outs = outgoing_edges(program)
    reachable = set()
    stack = [program.entry_node_id]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(edge.target for edge in outs.get(node_id, []))
    program.nodes = [node for node in program.nodes if node.node_id in reachable]
    program.edges = [edge for edge in program.edges if edge.source in reachable and edge.target in reachable]


def _dedupe_edges(edges: list[ProgramEdge]) -> list[ProgramEdge]:
    seen = set()
    deduped = []
    for edge in edges:
        key = (edge.source, edge.target, edge.label)
        if key not in seen:
            seen.add(key)
            deduped.append(edge)
    return deduped


def _last_answer_node(program: PromptProgram) -> str | None:
    for node in reversed(program.nodes):
        if node.node_type in {NodeType.DIRECT, NodeType.SOLVE, NodeType.FORMAT}:
            return node.node_id
    return None


def _entry_answerish_node(program: PromptProgram) -> str | None:
    if program.entry_node_id in program.node_map() and program.node_map()[program.entry_node_id].node_type in {NodeType.DIRECT, NodeType.SOLVE}:
        return program.entry_node_id
    for node in program.nodes:
        if node.node_type in {NodeType.DIRECT, NodeType.SOLVE}:
            return node.node_id
    return None
