from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from topoprompt.backends.llm_client import LLMBackend
from topoprompt.config import TopoPromptConfig
from topoprompt.runtime.executor import ProgramExecutor
from topoprompt.schemas import PromptProgram, TaskSpec


def compile_to_dspy(
    *,
    program: PromptProgram,
    task_spec: TaskSpec,
    config: TopoPromptConfig,
    backend: LLMBackend,
) -> Any | None:
    try:
        import dspy  # type: ignore
    except ImportError:
        return None

    executor = ProgramExecutor(backend=backend, config=config)

    class DSPyProgramAdapter(dspy.Module):  # type: ignore
        def forward(self, **kwargs: Any) -> dict[str, Any]:
            result = executor.run_program(
                program=program,
                task_spec=task_spec,
                example_id="dspy_runtime",
                task_input=kwargs,
                phase="confirmation",
            )
            return {"final_answer": result.trace.final_output, "trace": result.trace.model_dump(mode="json")}

    return DSPyProgramAdapter()


@dataclass
class DSPyMappingSummary:
    node_type_map: dict[str, str]


def node_mapping_summary(program: PromptProgram) -> DSPyMappingSummary:
    mapping = {}
    for node in program.nodes:
        if node.node_type.value == "direct":
            mapping[node.node_id] = "dspy.Predict"
        elif node.node_type.value in {"plan", "solve"}:
            mapping[node.node_id] = "dspy.ChainOfThought"
        elif node.node_type.value in {"verify", "format"}:
            mapping[node.node_id] = "dspy.Predict"
        elif node.node_type.value == "route":
            mapping[node.node_id] = "custom route wrapper"
        else:
            mapping[node.node_id] = "native adapter"
    return DSPyMappingSummary(node_type_map=mapping)

