from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.config import TopoPromptConfig
from topoprompt.eval.budget import BudgetLedger
from topoprompt.ir import outgoing_edges, topological_nodes
from topoprompt.progress import CompileProgressReporter
from topoprompt.runtime.cache import RuntimeCache
from topoprompt.runtime.parser import ParseFailed, parse_structured_output
from topoprompt.runtime.renderer import render_node_prompt
from topoprompt.runtime.router import choose_rule_route, resolve_route_choice
from topoprompt.schemas import NodeExecutionTrace, NodeType, ProgramExecutionTrace, ProgramNode, PromptProgram, RouteDiagnostic, TaskSpec


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class ExecutionResult:
    state: dict[str, Any]
    trace: ProgramExecutionTrace


class ProgramExecutor:
    def __init__(
        self,
        *,
        backend: LLMBackend,
        config: TopoPromptConfig,
        budget_ledger: BudgetLedger | None = None,
        cache: RuntimeCache | None = None,
        reporter: CompileProgressReporter | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.budget_ledger = budget_ledger
        self.reporter = reporter
        self.cache = cache or (
            RuntimeCache(Path(self.config.runtime.cache_dir)) if self.config.runtime.cache_enabled else None
        )

    def run_program(
        self,
        *,
        program: PromptProgram,
        task_spec: TaskSpec,
        example_id: str,
        task_input: dict[str, Any],
        phase: str,
        force_route_choices: dict[str, str] | None = None,
    ) -> ExecutionResult:
        state: dict[str, Any] = {"task_input": task_input}
        edges_by_source = outgoing_edges(program)
        nodes = program.node_map()
        current_node_id = program.entry_node_id
        node_traces: list[NodeExecutionTrace] = []
        visited = set()

        while current_node_id:
            if current_node_id in visited:
                raise RuntimeError(f"Loop detected at node {current_node_id}")
            visited.add(current_node_id)
            node = nodes[current_node_id]
            if self.reporter is not None:
                self.reporter.log_node_event(
                    program_id=program.program_id,
                    example_id=example_id,
                    node_id=node.node_id,
                    node_type=node.node_type.value,
                )
            if node.execution_mode == "pass_through":
                trace = self._execute_pass_through(
                    program_id=program.program_id,
                    example_id=example_id,
                    node=node,
                    state=state,
                )
            elif node.execution_mode == "decompose_macro":
                trace = self._execute_decompose(
                    program_id=program.program_id,
                    example_id=example_id,
                    node=node,
                    task_spec=task_spec,
                    state=state,
                    phase=phase,
                )
            else:
                trace = self._execute_llm_node(
                    program_id=program.program_id,
                    example_id=example_id,
                    node=node,
                    task_spec=task_spec,
                    state=state,
                    phase=phase,
                )
            node_traces.append(trace)

            if isinstance(trace.parsed_output, dict):
                state.update({key: value for key, value in trace.parsed_output.items() if value is not None})

            next_edges = edges_by_source.get(node.node_id, [])
            if node.node_type == NodeType.ROUTE:
                route_spec = node.route_spec
                assert route_spec is not None
                if force_route_choices and node.node_id in force_route_choices:
                    branch = force_route_choices[node.node_id]
                    confidence = 1.0
                elif route_spec.mode == "rule_route":
                    branch, confidence = choose_rule_route(node, state)
                    trace.route_choice = branch
                    trace.confidence = confidence
                elif route_spec.mode == "classifier_route":
                    classifier = node.config.get("classifier")
                    if callable(classifier):
                        branch = classifier(state)
                        confidence = 1.0
                    else:
                        branch = route_spec.fallback_branch or route_spec.branch_labels[0]
                        confidence = 0.0
                else:
                    branch, confidence = resolve_route_choice(route_spec, trace.parsed_output or {})
                trace.route_choice = branch
                trace.confidence = confidence
                if self.reporter is not None:
                    self.reporter.log_node_event(
                        program_id=program.program_id,
                        example_id=example_id,
                        node_id=node.node_id,
                        node_type=node.node_type.value,
                        route_choice=branch,
                    )
                next_edges = [edge for edge in next_edges if edge.label == branch] or next_edges

            if not next_edges:
                current_node_id = None
            else:
                current_node_id = next_edges[0].target

        final_output = state.get("final_answer") or state.get("candidate_answer")
        program_trace = ProgramExecutionTrace(
            example_id=example_id,
            program_id=program.program_id,
            node_traces=node_traces,
            final_output=final_output,
            total_tokens=sum(sum(trace.token_usage.values()) for trace in node_traces),
            total_latency_ms=sum(trace.latency_ms or 0 for trace in node_traces),
            total_invocations=sum(trace.invocation_cost for trace in node_traces),
            parse_failures=sum(1 for trace in node_traces if trace.parse_error),
            route_diagnostics=[],
        )
        return ExecutionResult(state=state, trace=program_trace)

    def _execute_pass_through(
        self,
        *,
        program_id: str,
        example_id: str,
        node: ProgramNode,
        state: dict[str, Any],
    ) -> NodeExecutionTrace:
        parsed_output: dict[str, Any]
        if node.node_type == NodeType.FINALIZE:
            source_key = node.config.get("source_key")
            if source_key is None:
                for key in ["candidate_answer", "verification_result", "plan"]:
                    if key in state:
                        source_key = key
                        break
            parsed_output = {"final_answer": state.get(source_key) if source_key else state.get("final_answer")}
        else:
            parsed_output = {}
            for output_key in node.output_keys:
                if output_key in state:
                    parsed_output[output_key] = state[output_key]
                elif node.input_keys:
                    parsed_output[output_key] = state.get(node.input_keys[0])
        return NodeExecutionTrace(
            node_id=node.node_id,
            prompt_text="",
            raw_output=json.dumps(parsed_output),
            parsed_output=parsed_output,
            invocation_cost=0,
        )

    def _execute_decompose(
        self,
        *,
        program_id: str,
        example_id: str,
        node: ProgramNode,
        task_spec: TaskSpec,
        state: dict[str, Any],
        phase: str,
    ) -> NodeExecutionTrace:
        trace = self._execute_llm_node(
            program_id=program_id,
            example_id=example_id,
            node=node,
            task_spec=task_spec,
            state=state,
            phase=phase,
        )
        parsed = trace.parsed_output or {}
        subquestions = list(parsed.get("subquestions") or [])[: self.config.program.max_subquestions_per_decompose]
        subanswers: list[str] = []
        for subquestion in subquestions:
            self._spend_budget(phase, 1)
            response = self.backend.generate_structured(
                system_prompt="You solve bounded subquestions.",
                user_prompt=f"Solve this subquestion and return JSON with candidate_answer only.\n{subquestion}",
                schema={
                    "type": "object",
                    "properties": {"candidate_answer": {"type": "string"}},
                    "required": ["candidate_answer"],
                },
                model=self.config.model.name,
                temperature=self.config.model.temperature,
                max_output_tokens=self.config.model.max_output_tokens,
            )
            subanswers.append((response.structured or {}).get("candidate_answer") or response.text)
        parsed["subquestion_answers"] = subanswers
        parsed["decomposition_context"] = " | ".join(f"{q} => {a}" for q, a in zip(subquestions, subanswers, strict=False))
        trace.parsed_output = parsed
        trace.raw_output = json.dumps(parsed)
        trace.invocation_cost += len(subquestions)
        return trace

    def _execute_llm_node(
        self,
        *,
        program_id: str,
        example_id: str,
        node: ProgramNode,
        task_spec: TaskSpec,
        state: dict[str, Any],
        phase: str,
    ) -> NodeExecutionTrace:
        system_prompt, user_prompt = render_node_prompt(task_spec, node, state)
        cache_key = None
        cached_response = None
        if self.cache is not None:
            cache_key = self.cache.key_for(
                model=self.config.model.name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.config.model.temperature,
                max_output_tokens=self.config.model.max_output_tokens,
            )
            cached_response = self.cache.get(cache_key)
        if cached_response is None:
            self._spend_budget(phase, 1)
            response = self.backend.generate_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=node.expected_output_schema,
                model=self.config.model.name,
                temperature=self.config.model.temperature,
                max_output_tokens=self.config.model.max_output_tokens,
            )
            if self.cache is not None and cache_key is not None:
                self.cache.set(cache_key, response)
            cache_hit = False
        else:
            response = cached_response
            cache_hit = True

        try:
            parsed_output, repair_used = (
                (response.structured, False)
                if response.structured is not None
                else parse_structured_output(
                    raw_output=response.text,
                    schema=node.expected_output_schema,
                    backend=self.backend,
                    repair_model=self.config.model.repair_model or self.config.model.name,
                )
            )
        except ParseFailed as exc:
            parsed_output = None
            repair_used = False
            if self.reporter is not None:
                self.reporter.log_node_event(
                    program_id=program_id,
                    example_id=example_id,
                    node_id=node.node_id,
                    node_type=node.node_type.value,
                    parse_error=str(exc),
                )
            return NodeExecutionTrace(
                node_id=node.node_id,
                prompt_text=user_prompt,
                raw_output=response.text,
                parsed_output=None,
                token_usage=response.token_usage,
                latency_ms=response.latency_ms,
                cache_hit=cache_hit,
                repair_used=repair_used,
                parse_error=str(exc),
                backend_request_id=response.request_id,
                invocation_cost=0 if cache_hit else 1,
            )

        return NodeExecutionTrace(
            node_id=node.node_id,
            prompt_text=user_prompt,
            raw_output=response.text,
            parsed_output=parsed_output,
            token_usage=response.token_usage,
            latency_ms=response.latency_ms,
            route_choice=parsed_output.get("branch") if isinstance(parsed_output, dict) else None,
            confidence=float(parsed_output.get("confidence")) if isinstance(parsed_output, dict) and parsed_output.get("confidence") is not None else None,
            cache_hit=cache_hit,
            repair_used=repair_used,
            backend_request_id=response.request_id,
            invocation_cost=0 if cache_hit else 1,
        )

    def _spend_budget(self, phase: str, calls: int) -> None:
        if self.budget_ledger is None:
            return
        allow_reserve = phase in {"confirmation", "analyzer"}
        if not self.budget_ledger.spend(phase, calls, allow_reserve=allow_reserve):
            raise BudgetExhausted(f"Budget exhausted for phase '{phase}'")
