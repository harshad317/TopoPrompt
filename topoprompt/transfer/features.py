from __future__ import annotations

from statistics import mean

from topoprompt.ir import prompt_token_count
from topoprompt.schemas import CandidateEvaluation, PromptProgram, TaskSpec


def extract_transfer_features(
    *,
    task_spec: TaskSpec,
    program: PromptProgram,
    candidate: CandidateEvaluation | None = None,
) -> dict:
    task_embedding_stub = [float((sum(ord(ch) for ch in task_spec.description[:64]) + offset) % 101) / 100.0 for offset in range(8)]
    features = {
        "task_features": {
            "task_family": task_spec.task_family,
            "description_length": len(task_spec.description.split()),
            "input_schema_keys": sorted(task_spec.input_schema),
            "output_schema_keys": sorted(task_spec.output_schema),
            "task_embedding": task_embedding_stub,
        },
        "topology_features": {
            "num_nodes": len(program.nodes),
            "num_edges": len(program.edges),
            "route_nodes": sum(1 for node in program.nodes if node.node_type.value == "route"),
            "has_verify": any(node.node_type.value == "verify" for node in program.nodes),
            "has_plan": any(node.node_type.value == "plan" for node in program.nodes),
            "has_decompose": any(node.node_type.value == "decompose" for node in program.nodes),
            "prompt_token_count": prompt_token_count(program),
            "fewshot_count": sum(
                1 for node in program.nodes for module in node.prompt_modules if module.role == "fewshot"
            ),
        },
    }
    if candidate is not None:
        features["candidate_metrics"] = {
            "score": candidate.score,
            "mean_invocations": candidate.mean_invocations,
            "mean_tokens": candidate.mean_tokens,
            "parse_failure_rate": candidate.parse_failure_rate,
            "mean_example_score": mean(candidate.example_scores) if candidate.example_scores else 0.0,
        }
    return features

