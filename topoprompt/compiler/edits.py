from __future__ import annotations

from topoprompt.compiler.templates import create_node, fewshot_module_from_examples
from topoprompt.config import TopoPromptConfig
from topoprompt.ir import clone_program, outgoing_edges
from topoprompt.schemas import CandidateEdit, Example, NodeType, ProgramEdge, PromptModule, PromptProgram, RouteSpec, TaskAnalysis


_DESTRUCTIVE_STRUCTURAL_EDIT_TYPES: frozenset[str] = frozenset({
    "replace_node_type",
    "split_with_route",
    "remove_route",
    "swap_branch_target",
    "add_node",
    "delete_node",
    "insert_plan_before",
    "change_finalize_format",
    # Dropping fewshots from a high-scoring program almost always degrades it;
    # block it so the optimizer can't produce degenerate add→drop no-op cycles.
    "drop_fewshot_module",
})

# Programs scoring at or above this threshold are considered high-performing.
# Destructive structural edits are suppressed to avoid mutating them away from
# their optimum — only conservative edits (fewshot, rewrite, verify) are allowed.
_HIGH_SCORE_CONSERVATIVE_THRESHOLD = 0.85


def generate_heuristic_edits(
    *,
    program: PromptProgram,
    analysis: TaskAnalysis,
    config: TopoPromptConfig,
    incumbent_score: float = 0.0,
) -> list[CandidateEdit]:
    _reset_edit_dedupe()
    family = analysis.task_family or "other"
    edits: list[CandidateEdit] = []
    if family == "classification":
        _add_route_split_edit(edits, program, analysis)
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_last_candidate_node(program),
            instruction="Choose the best label from the task's label space and avoid extra explanation.",
        )
        _add_finalize_format_edit(edits, "Return only the final label.")
        _add_route_tuning_edits(edits, program)
        _add_verify_cleanup_edit(edits, program)
        _add_fewshot_edits(edits, program)
    elif family == "extraction":
        _add_format_insertion_edit(edits, program)
        _add_route_split_edit(edits, program, analysis)
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_last_candidate_node(program),
            module_role="format",
            instruction="Preserve the requested field names and keep extracted values faithful to the input.",
        )
        _add_finalize_format_edit(edits, "Return only valid JSON matching the requested schema.")
        _add_verify_insertion_edit(edits, program, analysis)
        _add_fewshot_edits(edits, program)
    elif family == "instruction_following":
        if analysis.output_format == "json":
            _add_format_insertion_edit(edits, program)
        _add_verify_insertion_edit(edits, program, analysis)
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_last_candidate_node(program),
            instruction="Satisfy every stated instruction before answering.",
        )
        _add_finalize_format_edit(
            edits,
            f"Return only the requested {analysis.output_format.replace('_', ' ')} response.",
        )
        _add_route_split_edit(edits, program, analysis)
        _add_fewshot_edits(edits, program)
    elif family in {"generation", "summarization"}:
        _add_critique_revise_insertion_edit(edits, program)
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_last_candidate_node(program),
            instruction="Improve clarity, specificity, and overall answer quality while staying faithful to the task.",
        )
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_first_node_of_type(program, NodeType.CRITIQUE),
            instruction="Identify the most important weaknesses before revising.",
        )
        _add_route_split_edit(edits, program, analysis)
        _add_verify_cleanup_edit(edits, program)
        _add_fewshot_edits(edits, program)
    elif family == "code":
        _add_critique_revise_insertion_edit(edits, program)
        _add_verify_insertion_edit(edits, program, analysis)
        _add_solve_upgrade_edit(edits, program)
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_last_candidate_node(program),
            instruction="Return correct, executable code that handles the important edge cases.",
        )
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_first_node_of_type(program, NodeType.VERIFY),
            module_role="verification",
            instruction="Check correctness, edge cases, and required output format before finalizing.",
        )
        _add_fewshot_edits(edits, program)
    elif family in {"math_reasoning", "reasoning"}:
        _add_verify_insertion_edit(edits, program, analysis)
        _add_reasoning_structure_edits(edits, program, analysis)
        _add_route_split_edit(edits, program, analysis)
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_last_candidate_node(program),
            instruction="Reason carefully and keep the final answer exact.",
        )
        _add_finalize_format_edit(edits, "Return only the final answer.")
        _add_route_tuning_edits(edits, program)
        _add_fewshot_edits(edits, program)
    elif family == "mixed":
        _add_format_insertion_edit(edits, program)
        _add_critique_revise_insertion_edit(edits, program)
        _add_verify_insertion_edit(edits, program, analysis)
        _add_route_split_edit(edits, program, analysis)
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_last_candidate_node(program),
            instruction="Handle the heterogeneous cases explicitly and preserve correctness across branches.",
        )
        _add_fewshot_edits(edits, program)
    else:
        _add_verify_insertion_edit(edits, program, analysis)
        _add_reasoning_structure_edits(edits, program, analysis)
        _add_route_split_edit(edits, program, analysis)
        _add_targeted_rewrite(
            edits,
            program,
            target_node_id=_last_candidate_node(program),
            instruction="Make the instructions more task-specific.",
        )
        _add_finalize_format_edit(edits, "Return only the final answer.")
        _add_route_tuning_edits(edits, program)
        _add_fewshot_edits(edits, program)

    # When the parent program is already high-scoring, suppress destructive
    # structural edits that are likely to mutate it away from its optimum.
    # Conservative edits (fewshot tuning, prompt rewrites, verify insertion)
    # are still allowed — they tend to refine rather than restructure.
    if incumbent_score >= _HIGH_SCORE_CONSERVATIVE_THRESHOLD:
        edits = [e for e in edits if e.edit_type not in _DESTRUCTIVE_STRUCTURAL_EDIT_TYPES]

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
                input_keys=_default_insert_input_keys(edit.new_node_type),
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
        case "insert_format_after":
            target = edit.target_node_id or _last_candidate_node(candidate)
            if target is None:
                raise ValueError("insert_format_after requires target_node_id or an answer-producing node")
            new_node = create_node(
                node_id=_next_node_id(candidate, NodeType.FORMAT),
                node_type=NodeType.FORMAT,
                input_keys=["task_input", "candidate_answer"],
                task_analysis=analysis,
            )
            _insert_after(candidate, target, new_node)
            candidate.node_map()[candidate.finalize_node_id].config["source_key"] = "formatted_answer"
        case "insert_critique_revise_after":
            target = edit.target_node_id or _last_candidate_node(candidate)
            if target is None:
                raise ValueError("insert_critique_revise_after requires target_node_id or an answer-producing node")
            critique = create_node(
                node_id=_next_node_id(candidate, NodeType.CRITIQUE),
                node_type=NodeType.CRITIQUE,
                input_keys=["task_input", "candidate_answer"],
                prompt_modules=[
                    PromptModule(role="instruction", text="Identify the most important weaknesses in the current candidate answer."),
                    PromptModule(role="format", text="Return a concise critique grounded in the task requirements."),
                ],
                task_analysis=analysis,
            )
            _insert_after(candidate, target, critique)
            revise = create_node(
                node_id=_next_node_id(candidate, NodeType.SOLVE),
                node_type=NodeType.SOLVE,
                name="revise",
                input_keys=["task_input", "candidate_answer", "critique"],
                prompt_modules=[
                    PromptModule(
                        role="instruction",
                        text=f"Revise the current answer for the {analysis.task_family} task using the critique.",
                    ),
                    PromptModule(
                        role="reasoning",
                        text="Keep what is correct, fix what is wrong, and improve the final answer quality.",
                    ),
                    PromptModule(role="format", text="Return an improved candidate answer and a short rationale."),
                ],
                task_analysis=analysis,
            )
            _insert_after(candidate, critique.node_id, revise)
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
    rewrote_any = False
    for node in target_nodes:
        for module in node.prompt_modules:
            if edit.module_role is None or module.role == edit.module_role:
                module.text = f"{module.text.rstrip()} {instruction}".strip()
                rewrote_any = True
                # When a specific target node is given, rewrite only the first
                # matching module within that node (intentional single-module edit).
                # When targeting all nodes (target_node_id is None), rewrite the
                # first matching module in each node so the instruction propagates
                # across the whole program.
                break
    # If nothing was rewritten (e.g. no module with the requested role exists),
    # fall back to rewriting the first module of the first node.
    if not rewrote_any and target_nodes:
        node = target_nodes[0]
        if node.prompt_modules:
            module = node.prompt_modules[0]
            module.text = f"{module.text.rstrip()} {instruction}".strip()


def _default_insert_input_keys(node_type: NodeType) -> list[str]:
    if node_type in {NodeType.VERIFY, NodeType.CRITIQUE, NodeType.FORMAT}:
        return ["task_input", "candidate_answer"]
    return ["task_input"]


def _add_edit(edits: list[CandidateEdit], edit: CandidateEdit | None) -> None:
    if edit is None:
        return
    key = (
        edit.edit_type,
        edit.target_node_id,
        edit.new_node_type.value if edit.new_node_type else None,
        edit.module_role,
        tuple(edit.branch_labels or []),
        edit.rewrite_instruction,
    )
    seen = getattr(_add_edit, "_seen", None)
    if seen is None or not isinstance(seen, set):
        _add_edit._seen = set()
        seen = _add_edit._seen
    if key in seen:
        return
    seen.add(key)
    edits.append(edit)


def _reset_edit_dedupe() -> None:
    _add_edit._seen = set()


def _add_verify_insertion_edit(edits: list[CandidateEdit], program: PromptProgram, analysis: TaskAnalysis) -> None:
    if not analysis.needs_verification or any(node.node_type == NodeType.VERIFY for node in program.nodes):
        return
    target = _last_answer_node(program)
    if target:
        _add_edit(
            edits,
            CandidateEdit(
                edit_type="insert_verify_after",
                target_node_id=target,
                reason="Add an explicit verification pass for constraint-heavy tasks.",
            ),
        )


def _add_reasoning_structure_edits(edits: list[CandidateEdit], program: PromptProgram, analysis: TaskAnalysis) -> None:
    if not analysis.needs_reasoning or any(node.node_type == NodeType.PLAN for node in program.nodes):
        return
    solve_nodes = [node.node_id for node in program.nodes if node.node_type == NodeType.SOLVE]
    if solve_nodes:
        _add_edit(
            edits,
            CandidateEdit(
                edit_type="insert_plan_before",
                target_node_id=solve_nodes[0],
                reason="Add an explicit planning step before the main solve node.",
            ),
        )
        return
    _add_solve_upgrade_edit(edits, program)


def _add_solve_upgrade_edit(edits: list[CandidateEdit], program: PromptProgram) -> None:
    direct_nodes = [node.node_id for node in program.nodes if node.node_type == NodeType.DIRECT]
    solve_nodes = [node.node_id for node in program.nodes if node.node_type == NodeType.SOLVE]
    if solve_nodes or not direct_nodes:
        return
    _add_edit(
        edits,
        CandidateEdit(
            edit_type="replace_node_type",
            target_node_id=direct_nodes[0],
            new_node_type=NodeType.SOLVE,
            reason="Upgrade the direct answer node into a solve node.",
        ),
    )


def _add_route_split_edit(edits: list[CandidateEdit], program: PromptProgram, analysis: TaskAnalysis) -> None:
    if analysis.input_heterogeneity not in {"medium", "high"} or any(node.node_type == NodeType.ROUTE for node in program.nodes):
        return
    answer_target = _entry_answerish_node(program)
    if answer_target:
        _add_edit(
            edits,
            CandidateEdit(
                edit_type="split_with_route",
                target_node_id=answer_target,
                reason="Route heterogeneous inputs to specialized answer paths.",
            ),
        )


def _add_route_tuning_edits(edits: list[CandidateEdit], program: PromptProgram) -> None:
    route_node = _first_node_of_type(program, NodeType.ROUTE)
    if route_node is None:
        return
    _add_edit(edits, CandidateEdit(edit_type="swap_branch_target", target_node_id=route_node, reason="Swap route targets to test a better branch mapping."))
    _add_edit(edits, CandidateEdit(edit_type="remove_route", target_node_id=route_node, reason="Remove routing if branching is adding unnecessary complexity."))


def _add_verify_cleanup_edit(edits: list[CandidateEdit], program: PromptProgram) -> None:
    verify_node = _first_node_of_type(program, NodeType.VERIFY)
    if verify_node is None:
        return
    _add_edit(
        edits,
        CandidateEdit(
            edit_type="delete_node",
            target_node_id=verify_node,
            reason="Remove a verification pass that may be wasting budget on low-risk tasks.",
        ),
    )


def _add_format_insertion_edit(edits: list[CandidateEdit], program: PromptProgram) -> None:
    if any(node.node_type == NodeType.FORMAT for node in program.nodes):
        return
    target = _last_candidate_node(program)
    if target:
        _add_edit(
            edits,
            CandidateEdit(
                edit_type="insert_format_after",
                target_node_id=target,
                reason="Add a dedicated formatting node for structured or schema-bound outputs.",
            ),
        )


def _add_critique_revise_insertion_edit(edits: list[CandidateEdit], program: PromptProgram) -> None:
    target = _last_candidate_node(program)
    if target is None:
        return
    _add_edit(
        edits,
        CandidateEdit(
            edit_type="insert_critique_revise_after",
            target_node_id=target,
            reason="Add a critique-and-revise loop to improve answer quality before finalization.",
        ),
    )


def _add_targeted_rewrite(
    edits: list[CandidateEdit],
    program: PromptProgram,
    *,
    target_node_id: str | None,
    instruction: str,
    module_role: str | None = "instruction",
) -> None:
    if target_node_id is None:
        return
    _add_edit(
        edits,
        CandidateEdit(
            edit_type="rewrite_prompt_module",
            target_node_id=target_node_id,
            module_role=module_role,
            rewrite_instruction=instruction,
            reason="Refine prompt text for the detected task family.",
        ),
    )


def _add_finalize_format_edit(edits: list[CandidateEdit], instruction: str) -> None:
    _add_edit(
        edits,
        CandidateEdit(
            edit_type="change_finalize_format",
            rewrite_instruction=instruction,
            reason="Tighten the final output format to match task expectations.",
        ),
    )


def _add_fewshot_edits(edits: list[CandidateEdit], program: PromptProgram) -> None:
    if _has_fewshot_module(program):
        # Only propose dropping fewshots if they were NOT recently added by the
        # optimizer itself (i.e. "add_fewshot_module" is not in the program's
        # edit chain).  This prevents the add→drop no-op seesaw where the
        # optimizer adds fewshots, evaluates them on a small screening set,
        # then immediately proposes removing them in the next round.
        if "add_fewshot_module" not in program.program_id:
            _add_edit(
                edits,
                CandidateEdit(
                    edit_type="drop_fewshot_module",
                    reason="Remove few-shot guidance if it is adding noise.",
                ),
            )
    else:
        _add_edit(
            edits,
            CandidateEdit(
                edit_type="add_fewshot_module",
                reason="Add a few-shot module to ground the model on representative examples.",
            ),
        )


def _has_fewshot_module(program: PromptProgram) -> bool:
    return any(module.role == "fewshot" for node in program.nodes for module in node.prompt_modules)


def _first_node_of_type(program: PromptProgram, node_type: NodeType) -> str | None:
    for node in program.nodes:
        if node.node_type == node_type:
            return node.node_id
    return None


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


def _last_candidate_node(program: PromptProgram) -> str | None:
    for node in reversed(program.nodes):
        if node.node_type in {NodeType.DIRECT, NodeType.SOLVE}:
            return node.node_id
    return None


def _entry_answerish_node(program: PromptProgram) -> str | None:
    if program.entry_node_id in program.node_map() and program.node_map()[program.entry_node_id].node_type in {NodeType.DIRECT, NodeType.SOLVE}:
        return program.entry_node_id
    for node in program.nodes:
        if node.node_type in {NodeType.DIRECT, NodeType.SOLVE}:
            return node.node_id
    return None
