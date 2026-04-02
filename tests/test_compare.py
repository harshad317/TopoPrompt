from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import orjson

from topoprompt.cli import main
from topoprompt.eval.compare import compare_programs
from topoprompt.schemas import Example, NodeType, ProgramEdge, ProgramNode, PromptModule, PromptProgram, TaskSpec


def _make_direct_program(program_id: str, *, instruction: str = "") -> PromptProgram:
    prompt_modules = [PromptModule(role="instruction", text=instruction)] if instruction else []
    return PromptProgram(
        program_id=program_id,
        task_id="arith_task",
        nodes=[
            ProgramNode(
                node_id="direct_1",
                node_type=NodeType.DIRECT,
                name="Direct",
                output_keys=["candidate_answer"],
                expected_output_schema={
                    "type": "object",
                    "properties": {"candidate_answer": {"type": "string"}},
                    "required": ["candidate_answer"],
                },
                prompt_modules=prompt_modules,
            ),
            ProgramNode(
                node_id="finalize_1",
                node_type=NodeType.FINALIZE,
                name="Finalize",
                input_keys=["candidate_answer"],
                output_keys=["final_answer"],
                execution_mode="pass_through",
                config={"source_key": "candidate_answer"},
            ),
        ],
        edges=[ProgramEdge(source="direct_1", target="finalize_1")],
        entry_node_id="direct_1",
        finalize_node_id="finalize_1",
    )


def test_compare_programs_writes_disagreements(tmp_path, small_config):
    def structured_handler(_system_prompt: str, user_prompt: str, _schema: dict[str, object]) -> dict[str, str]:
        if "Always return zero." in user_prompt:
            return {"candidate_answer": "0"}
        match = re.search(r'"question":\s*"What is (\d+) \+ (\d+)\?"', user_prompt)
        assert match is not None
        return {"candidate_answer": str(int(match.group(1)) + int(match.group(2)))}

    from topoprompt.backends.llm_client import FakeBackend

    backend = FakeBackend(structured_handler=structured_handler)
    task_spec = TaskSpec(
        task_id="arith_task",
        description="Answer simple arithmetic questions accurately.",
        input_schema={"question": "str"},
        output_schema={"type": "string"},
    )
    examples = [
        Example(example_id="e1", input={"question": "What is 1 + 1?"}, target="2"),
        Example(example_id="e2", input={"question": "What is 2 + 2?"}, target="4"),
    ]
    strong = _make_direct_program("strong")
    weak = _make_direct_program("weak", instruction="Always return zero.")

    result = compare_programs(
        program_a=strong,
        program_b=weak,
        task_spec=task_spec,
        examples=examples,
        metric_fn=lambda output, example: 1.0 if str(output) == str(example.target) else 0.0,
        backend=backend,
        config=small_config,
        label_a="strong",
        label_b="weak",
        repeats=2,
        output_dir=tmp_path / "compare",
    )

    assert result["score_a_mean"] == 1.0
    assert result["score_b_mean"] == 0.0
    assert result["a_better_count_mean"] == 2.0
    assert result["b_better_count_mean"] == 0.0
    assert (tmp_path / "compare" / "compare_summary.json").exists()
    assert (tmp_path / "compare" / "compare_summary.md").exists()
    assert (tmp_path / "compare" / "repeat_metrics.jsonl").exists()
    disagreement_path = tmp_path / "compare" / "disagreements_repeat_1.jsonl"
    assert disagreement_path.exists()
    rows = [orjson.loads(line) for line in disagreement_path.read_bytes().splitlines() if line.strip()]
    assert len(rows) == 2
    assert rows[0]["program_a_label"] == "strong"
    assert rows[0]["program_b_label"] == "weak"


def test_cli_compare_smoke(fake_backend, tmp_path, monkeypatch, capsys):
    task_spec = TaskSpec(
        task_id="arith_task",
        description="Answer simple arithmetic questions accurately.",
        input_schema={"question": "str"},
        output_schema={"type": "string"},
    )
    program = _make_direct_program("direct")
    examples = [
        {
            "example_id": "e1",
            "input": {"question": "What is 1 + 1?"},
            "target": "2",
        },
        {
            "example_id": "e2",
            "input": {"question": "What is 2 + 2?"},
            "target": "4",
        },
    ]

    program_a_path = tmp_path / "program_a.json"
    program_b_path = tmp_path / "program_b.json"
    task_spec_path = tmp_path / "task_spec.json"
    dataset_path = tmp_path / "dataset.jsonl"
    output_dir = tmp_path / "compare_output"

    program_a_path.write_text(program.model_dump_json())
    program_b_path.write_text(program.model_dump_json())
    task_spec_path.write_text(task_spec.model_dump_json())
    dataset_path.write_bytes(b"\n".join(orjson.dumps(row) for row in examples) + b"\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "topoprompt",
            "compare",
            "--program-a",
            str(program_a_path),
            "--program-b",
            str(program_b_path),
            "--dataset",
            str(dataset_path),
            "--task-spec",
            str(task_spec_path),
            "--metric",
            "gsm8k",
            "--fake-backend",
            "--output-dir",
            str(output_dir),
            "--quiet",
        ],
    )

    main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["repeats"] == 1
    assert payload["program_a_id"] == "direct"
    assert payload["program_b_id"] == "direct"
    assert payload["mean_invocations_a_mean"] == 1.0
    assert payload["mean_invocations_b_mean"] == 1.0
    assert (output_dir / "compare_summary.json").exists()
