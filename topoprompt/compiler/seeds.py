from __future__ import annotations

from topoprompt.compiler.templates import create_node, default_prompt_modules
from topoprompt.schemas import NodeType, ProgramEdge, PromptModule, PromptProgram, RouteSpec, TaskAnalysis, TaskSpec


SEED_LIBRARY = [
    "direct_finalize",
    "plan_solve_finalize",
    "decompose_solve_finalize",
    "solve_verify_finalize",
    "route_direct_or_solve_finalize",
    "plan_solve_verify_finalize",
    "route_direct_or_plan_solve_finalize",
]


def instantiate_seed_programs(
    *,
    task_spec: TaskSpec,
    analysis: TaskAnalysis,
    include_direct_baseline: bool = True,
    seed_names: list[str] | None = None,
) -> list[PromptProgram]:
    selected = seed_names or analysis.initial_seed_templates or SEED_LIBRARY[:]
    seeds: list[PromptProgram] = []
    if include_direct_baseline and "direct_finalize" not in selected:
        selected = ["direct_finalize", *selected]
    for name in list(dict.fromkeys(selected)):
        program = instantiate_seed_program(task_spec=task_spec, analysis=analysis, template_name=name)
        if program is not None:
            seeds.append(program)
    return seeds


def instantiate_direct_self_consistency_program(
    *,
    task_spec: TaskSpec,
    analysis: TaskAnalysis,
    attempts: int = 3,
    program_id: str | None = None,
) -> PromptProgram:
    if attempts < 2:
        raise ValueError("Self-consistency baseline requires at least 2 attempts.")

    nodes = []
    edges: list[ProgramEdge] = []
    answer_keys: list[str] = []
    previous_node_id: str | None = None

    for attempt_index in range(1, attempts + 1):
        answer_key = f"candidate_answer_{attempt_index}"
        rationale_key = f"rationale_{attempt_index}"
        answer_keys.append(answer_key)
        direct_node = create_node(
            node_id=f"direct_{attempt_index}",
            node_type=NodeType.DIRECT,
            input_keys=["task_input"],
            output_keys=[answer_key, rationale_key],
            config={
                "output_alias_map": {
                    "candidate_answer": answer_key,
                    "rationale": rationale_key,
                }
            },
            prompt_modules=[
                *default_prompt_modules(NodeType.DIRECT, task_analysis=analysis),
                PromptModule(
                    role="instruction",
                    text=f"This is independent attempt {attempt_index} of {attempts}. Do not reference prior attempts.",
                ),
            ],
            task_analysis=analysis,
        )
        nodes.append(direct_node)
        if previous_node_id is not None:
            edges.append(ProgramEdge(source=previous_node_id, target=direct_node.node_id))
        previous_node_id = direct_node.node_id

    finalize = create_node(
        "finalize_1",
        NodeType.FINALIZE,
        input_keys=answer_keys,
        config={"mode": "majority_vote", "source_keys": answer_keys},
    )
    nodes.append(finalize)
    assert previous_node_id is not None
    edges.append(ProgramEdge(source=previous_node_id, target="finalize_1"))

    return PromptProgram(
        program_id=program_id or f"direct_self_consistency_x{attempts}",
        task_id=task_spec.task_id,
        nodes=nodes,
        edges=edges,
        entry_node_id=nodes[0].node_id,
        finalize_node_id="finalize_1",
        metadata={"seed_template": "direct_self_consistency", "attempts": attempts},
    )


def instantiate_seed_program(*, task_spec: TaskSpec, analysis: TaskAnalysis, template_name: str) -> PromptProgram | None:
    route_candidates = analysis.candidate_routes or []
    default_branch_descriptions = {route.label: route.description for route in route_candidates}
    if not default_branch_descriptions:
        default_branch_descriptions = {
            "direct": "Use for direct or factual items.",
            "solve": "Use for reasoning or arithmetic items.",
        }

    match template_name:
        case "direct_finalize":
            direct = create_node(
                node_id="direct_1",
                node_type=NodeType.DIRECT,
                input_keys=["task_input"],
                task_analysis=analysis,
            )
            finalize = create_node(
                node_id="finalize_1",
                node_type=NodeType.FINALIZE,
                config={"source_key": "candidate_answer"},
            )
            return PromptProgram(
                program_id=template_name,
                task_id=task_spec.task_id,
                nodes=[direct, finalize],
                edges=[ProgramEdge(source="direct_1", target="finalize_1")],
                entry_node_id="direct_1",
                finalize_node_id="finalize_1",
                metadata={"seed_template": template_name},
            )
        case "direct_self_consistency_x3":
            return instantiate_direct_self_consistency_program(
                task_spec=task_spec,
                analysis=analysis,
                attempts=3,
                program_id=template_name,
            )
        case "plan_solve_finalize":
            plan = create_node("plan_1", NodeType.PLAN, input_keys=["task_input"], task_analysis=analysis)
            solve = create_node("solve_1", NodeType.SOLVE, input_keys=["task_input", "plan"], task_analysis=analysis)
            finalize = create_node("finalize_1", NodeType.FINALIZE, config={"source_key": "candidate_answer"})
            return PromptProgram(
                program_id=template_name,
                task_id=task_spec.task_id,
                nodes=[plan, solve, finalize],
                edges=[
                    ProgramEdge(source="plan_1", target="solve_1"),
                    ProgramEdge(source="solve_1", target="finalize_1"),
                ],
                entry_node_id="plan_1",
                finalize_node_id="finalize_1",
                metadata={"seed_template": template_name},
            )
        case "decompose_solve_finalize":
            decompose = create_node("decompose_1", NodeType.DECOMPOSE, input_keys=["task_input"], task_analysis=analysis)
            solve = create_node(
                "solve_1",
                NodeType.SOLVE,
                input_keys=["task_input", "decomposition_context"],
                task_analysis=analysis,
            )
            finalize = create_node("finalize_1", NodeType.FINALIZE, config={"source_key": "candidate_answer"})
            return PromptProgram(
                program_id=template_name,
                task_id=task_spec.task_id,
                nodes=[decompose, solve, finalize],
                edges=[
                    ProgramEdge(source="decompose_1", target="solve_1"),
                    ProgramEdge(source="solve_1", target="finalize_1"),
                ],
                entry_node_id="decompose_1",
                finalize_node_id="finalize_1",
                metadata={"seed_template": template_name},
            )
        case "solve_verify_finalize":
            solve = create_node("solve_1", NodeType.SOLVE, input_keys=["task_input"], task_analysis=analysis)
            verify = create_node(
                "verify_1",
                NodeType.VERIFY,
                input_keys=["task_input", "candidate_answer"],
                task_analysis=analysis,
            )
            finalize = create_node("finalize_1", NodeType.FINALIZE, config={"source_key": "candidate_answer"})
            return PromptProgram(
                program_id=template_name,
                task_id=task_spec.task_id,
                nodes=[solve, verify, finalize],
                edges=[
                    ProgramEdge(source="solve_1", target="verify_1"),
                    ProgramEdge(source="verify_1", target="finalize_1"),
                ],
                entry_node_id="solve_1",
                finalize_node_id="finalize_1",
                metadata={"seed_template": template_name},
            )
        case "route_direct_or_solve_finalize":
            route = create_node(
                node_id="route_1",
                node_type=NodeType.ROUTE,
                input_keys=["task_input"],
                route_spec=RouteSpec(
                    mode="self_route_llm",
                    branch_labels=["direct", "solve"],
                    branch_descriptions={
                        "direct": default_branch_descriptions.get("direct", "Direct answer branch"),
                        "solve": default_branch_descriptions.get("solve", "Reasoning branch"),
                    },
                    fallback_branch="direct",
                ),
                prompt_modules=default_prompt_modules(NodeType.ROUTE, task_analysis=analysis),
            )
            direct = create_node("direct_1", NodeType.DIRECT, input_keys=["task_input"], task_analysis=analysis)
            solve = create_node("solve_1", NodeType.SOLVE, input_keys=["task_input"], task_analysis=analysis)
            finalize = create_node("finalize_1", NodeType.FINALIZE, config={"source_key": "candidate_answer"})
            return PromptProgram(
                program_id=template_name,
                task_id=task_spec.task_id,
                nodes=[route, direct, solve, finalize],
                edges=[
                    ProgramEdge(source="route_1", target="direct_1", label="direct"),
                    ProgramEdge(source="route_1", target="solve_1", label="solve"),
                    ProgramEdge(source="direct_1", target="finalize_1"),
                    ProgramEdge(source="solve_1", target="finalize_1"),
                ],
                entry_node_id="route_1",
                finalize_node_id="finalize_1",
                metadata={"seed_template": template_name},
            )
        case "plan_solve_verify_finalize":
            plan = create_node("plan_1", NodeType.PLAN, input_keys=["task_input"], task_analysis=analysis)
            solve = create_node("solve_1", NodeType.SOLVE, input_keys=["task_input", "plan"], task_analysis=analysis)
            verify = create_node("verify_1", NodeType.VERIFY, input_keys=["task_input", "candidate_answer"], task_analysis=analysis)
            finalize = create_node("finalize_1", NodeType.FINALIZE, config={"source_key": "candidate_answer"})
            return PromptProgram(
                program_id=template_name,
                task_id=task_spec.task_id,
                nodes=[plan, solve, verify, finalize],
                edges=[
                    ProgramEdge(source="plan_1", target="solve_1"),
                    ProgramEdge(source="solve_1", target="verify_1"),
                    ProgramEdge(source="verify_1", target="finalize_1"),
                ],
                entry_node_id="plan_1",
                finalize_node_id="finalize_1",
                metadata={"seed_template": template_name},
            )
        case "route_direct_or_plan_solve_finalize":
            route = create_node(
                node_id="route_1",
                node_type=NodeType.ROUTE,
                input_keys=["task_input"],
                route_spec=RouteSpec(
                    mode="self_route_llm",
                    branch_labels=["direct", "plan_solve"],
                    branch_descriptions={
                        "direct": default_branch_descriptions.get("direct", "Direct answer branch"),
                        "plan_solve": "Use for reasoning tasks that benefit from a plan.",
                    },
                    fallback_branch="direct",
                ),
                prompt_modules=default_prompt_modules(NodeType.ROUTE, task_analysis=analysis),
            )
            direct = create_node("direct_1", NodeType.DIRECT, input_keys=["task_input"], task_analysis=analysis)
            plan = create_node("plan_1", NodeType.PLAN, input_keys=["task_input"], task_analysis=analysis)
            solve = create_node("solve_1", NodeType.SOLVE, input_keys=["task_input", "plan"], task_analysis=analysis)
            finalize = create_node("finalize_1", NodeType.FINALIZE, config={"source_key": "candidate_answer"})
            return PromptProgram(
                program_id=template_name,
                task_id=task_spec.task_id,
                nodes=[route, direct, plan, solve, finalize],
                edges=[
                    ProgramEdge(source="route_1", target="direct_1", label="direct"),
                    ProgramEdge(source="route_1", target="plan_1", label="plan_solve"),
                    ProgramEdge(source="direct_1", target="finalize_1"),
                    ProgramEdge(source="plan_1", target="solve_1"),
                    ProgramEdge(source="solve_1", target="finalize_1"),
                ],
                entry_node_id="route_1",
                finalize_node_id="finalize_1",
                metadata={"seed_template": template_name},
            )
    return None
