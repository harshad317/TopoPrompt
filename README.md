# TopoPrompt

TopoPrompt is a compiler for discovering prompt-program topology in black-box LLM settings.

Instead of only optimizing prompt text inside a fixed user-authored structure, TopoPrompt searches over the structure itself: direct programs, staged programs, verified programs, and routed programs. It compiles the best candidates under a fixed budget and returns the smallest effective program among near-best finalists.

This repository contains a working V1 research prototype with:

- a typed DAG IR for prompt programs
- a runtime with prompt rendering, parsing, tracing, routing, and caching hooks
- a task analyzer and seed topology library
- a hybrid beam search over structural edits
- a smallest-effective final selector
- artifact export, CLI support, DSPy export hooks, and benchmark adapters
- V2 transfer interfaces behind no-op defaults

## Status

The repository is functional as a greenfield V1 prototype.

What is implemented:

- compile-time task analysis from task description plus examples
- seed generation for seven topology templates
- execution for `direct`, `plan`, `decompose`, `solve`, `verify`, `route`, and `finalize`
- bounded `decompose` execution as a runtime macro
- routing with `self_route_llm` and `rule_route`
- structural edits and budgeted multi-fidelity search
- variance-adaptive smallest-effective final selection
- run artifact export
- CLI compile and evaluate commands
- smoke benchmark adapters for GSM8K, MMLU, BBH, and IFEval
- fake backend support for deterministic local testing

What is intentionally deferred or partial:

- transfer and uncertainty are interface-only in V1
- `classifier_route` exists as an interface, but is not trained by default
- DSPy export is an adapter layer, not a full DSPy-native runtime
- live OpenAI execution is implemented but not exercised by CI
- `critique` is schema-valid but not part of the default V1 search operator set

## Repository Layout

```text
topoprompt/
  compiler/
    analyzer.py
    edits.py
    objective.py
    search.py
    seeds.py
    selector.py
    templates.py
    validator.py
  runtime/
    cache.py
    executor.py
    parser.py
    renderer.py
    router.py
    trace.py
  backends/
    dspy_backend.py
    llm_client.py
    openai_backend.py
  eval/
    benchmark_runner.py
    budget.py
    datasets.py
    metrics.py
  transfer/
    acquisition.py
    features.py
    posterior.py
    store.py
  artifacts.py
  cli.py
  config.py
  ir.py
  schemas.py
configs/
  topoprompt_v1.yaml
tests/
  fixtures/smoke/
```

## Installation

### Prerequisites

- Python 3.12+
- `uv` recommended

### Install dependencies

Base install:

```bash
uv sync
```

Install developer dependencies:

```bash
uv sync --extra dev
```

Install the optional DSPy extra:

```bash
uv sync --extra dspy
```

For one-off DSPy CLI runs, include the extra on the `uv run` command too:

```bash
uv run --extra dspy python -m topoprompt.cli benchmark-dspy ...
```

If you want to use the OpenAI backend, export an API key first:

```bash
export OPENAI_API_KEY=your_key_here
```

## Quick Start

### 1. Run a smoke compile with the fake backend

This requires no external API calls.

```bash
uv run python -m topoprompt.cli compile \
  --task-file ./tests/fixtures/smoke/gsm8k_task.md \
  --examples-file ./tests/fixtures/smoke/gsm8k_examples.jsonl \
  --output-dir ./runs/smoke_gsm8k \
  --metric numeric \
  --fake-backend
```

### 2. Inspect the output bundle

The compile command writes artifacts into the output directory, including:

- `task_spec.json`
- `config.yaml`
- `seed_programs.json`
- `candidate_archive.jsonl`
- `final_program.json`
- `final_program.yaml`
- `compile_trace.jsonl`
- `metrics.json`
- `summary.md`
- `transfer_trace_store.jsonl`

### 3. Evaluate the final program

```bash
uv run python -m topoprompt.cli evaluate \
  --program ./runs/smoke_gsm8k/final_program.json \
  --task-spec ./runs/smoke_gsm8k/task_spec.json \
  --dataset ./tests/fixtures/smoke/gsm8k_examples.jsonl \
  --metric numeric \
  --fake-backend
```

## Python API

The main entrypoint is `compile_task`.

```python
from topoprompt import compile_task
from topoprompt.backends.llm_client import FakeBackend
from topoprompt.schemas import Example

artifact = compile_task(
    task_description="Answer simple arithmetic questions accurately.",
    examples=[
        Example(
            example_id="ex1",
            input={"question": "What is 2 + 2?"},
            target="4",
        )
    ],
    metric="numeric",
    backend=FakeBackend(),
    output_dir="./runs/api_example",
)
```

The returned `CompileArtifact` contains:

- `task_spec`
- `program_ir`
- `python_program`
- `dspy_program`
- `seed_programs`
- `candidate_archive`
- `compile_trace`
- `metrics`

### Explicit partition control

If you want TopoPrompt to use your own partitions instead of internal splitting, pass them directly:

```python
artifact = compile_task(
    task_description="Answer questions accurately.",
    examples=compile_examples + validation_examples,
    search_examples=search_examples,
    validation_examples=validation_examples,
    fewshot_examples=fewshot_examples,
    metric="exact_match",
    backend=FakeBackend(),
)
```

When explicit partitions are provided, the compiler uses them as-is.

## Input Data Format

The JSONL loader accepts two practical shapes.

### Canonical form

```json
{"example_id":"ex1","input":{"question":"What is 2 + 2?"},"target":"4"}
```

### Convenience form

```json
{"example_id":"ex1","question":"What is 2 + 2?","target":"4"}
```

The loader normalizes benchmark-style rows into `Example` objects.

Supported convenience fields currently include:

- `question`
- `prompt`
- `choices`
- `required_phrase`
- `answer`
- `label`

## CLI

### Compile

```bash
uv run python -m topoprompt.cli compile \
  --task-file ./path/to/task.md \
  --examples-file ./path/to/examples.jsonl \
  --config ./configs/topoprompt_v1.yaml \
  --output-dir ./runs/my_run \
  --metric exact_match
```

Arguments:

- `--task-file`: markdown or text file containing the task description
- `--examples-file`: JSONL examples
- `--config`: optional config override file
- `--output-dir`: run artifact directory
- `--metric`: canonical metric name such as `exact_match`, `numeric`, `multiple_choice`, `bbh`, or `ifeval`
- Benchmark aliases like `gsm8k` and `mmlu` remain accepted for backward compatibility.
- `--fake-backend`: use the deterministic fake backend instead of OpenAI

### Evaluate

```bash
uv run python -m topoprompt.cli evaluate \
  --program ./runs/my_run/final_program.json \
  --task-spec ./runs/my_run/task_spec.json \
  --dataset ./path/to/eval.jsonl \
  --metric exact_match
```

Arguments:

- `--program`: compiled `PromptProgram` JSON
- `--task-spec`: optional serialized `TaskSpec`; if omitted, a minimal one is inferred
- `--dataset`: JSONL evaluation set
- `--config`: optional config override file
- `--metric`: evaluation metric
- `--fake-backend`: use the deterministic fake backend

### DSPy Benchmarks

DSPy commands require the optional `dspy` extra. You can either install it once with
`uv sync --extra dspy` or include it directly on the run command:

```bash
uv run --extra dspy python -m topoprompt.cli benchmark-dspy \
  --benchmark gsm8k \
  --split "train[:200]" \
  --output-dir ./runs/gsm8k_three_way \
  --optimizers topoprompt,mipro,gepa \
  --model gpt-4.1-mini \
  --reflection-model gpt-4.1-mini \
  -v
```

## Architecture

TopoPrompt is structured into five V1 layers.

### 1. Typed IR

The internal representation is a typed DAG with:

- `TaskSpec`
- `Example`
- `ProgramNode`
- `ProgramEdge`
- `PromptProgram`
- execution traces and candidate archive records

The IR is independent of DSPy. DSPy is an export target, not the source of truth.

### 2. Runtime

The runtime executes one program on one example by:

1. initializing state with `task_input`
2. traversing the graph from `entry_node_id`
3. rendering a local prompt for each node
4. calling the backend when `execution_mode == "llm_call"`
5. parsing structured output with fallback and optional repair
6. routing across labeled branches when a `route` node fires
7. writing `final_answer` at `finalize`

Core runtime node support:

- `direct`
- `solve`
- `verify`
- `route`
- `finalize`

Extended V1 support:

- `plan`
- `decompose`

Schema-only or deferred support:

- `format`
- `critique`

### 3. Task Analysis And Seeding

The compiler analyzes:

- task family
- output format
- reasoning need
- verification need
- decomposition need
- input heterogeneity
- route candidates
- initial seed templates

Seed templates implemented:

- `direct_finalize`
- `plan_solve_finalize`
- `decompose_solve_finalize`
- `solve_verify_finalize`
- `route_direct_or_solve_finalize`
- `plan_solve_verify_finalize`
- `route_direct_or_plan_solve_finalize`

### 4. Search

Search combines:

- heuristic edit generation
- optional bounded LLM edit proposals
- candidate deduplication by topology fingerprint
- multi-fidelity screening and narrowing
- beam-family diversity preservation
- round-wise incumbent confirmation
- final confirmation on validation examples

Structural edits currently implemented:

- `add_node`
- `delete_node`
- `replace_node_type`
- `insert_verify_after`
- `insert_plan_before`
- `split_with_route`
- `remove_route`
- `swap_branch_target`
- `rewrite_prompt_module`
- `add_fewshot_module`
- `drop_fewshot_module`
- `change_finalize_format`

### 5. Selection

Search-time ranking uses a soft objective:

`Perf - alpha * Cost - beta * Complexity - gamma * ParseFailureRate`

Final selection uses:

1. best confirmed validation score
2. variance-adaptive epsilon around the best
3. smallest candidate among the effective set

## Runtime Details

### Prompt rendering

Node prompts are rendered from:

- system preamble modules
- node-specific modules
- local state only
- node output schema

The renderer intentionally avoids dumping the entire execution state into every node prompt.

### Parsing and repair

Node outputs follow this parsing sequence:

1. strict JSON parse
2. regex or key extraction fallback
3. repair call through the backend
4. structured parse failure

Parse failures are recorded explicitly in node traces and folded into search-time scoring.

### Routing

Supported route modes:

- `self_route_llm`
- `rule_route`
- `classifier_route` interface only

Route diagnostics are induced during evaluation by replaying alternate branch choices on small subsets and logging:

- chosen branch
- oracle branch
- branch scores
- regret

### Decomposition

`decompose` runs as a bounded runtime macro, not arbitrary graph fan-out.

Current limits:

- maximum subquestions is controlled by config
- one bounded subcall per subquestion
- outputs are summarized into `decomposition_context`

## Configuration

The default config lives at [`configs/topoprompt_v1.yaml`](configs/topoprompt_v1.yaml).

Key sections:

- `model`
- `compile`
- `program`
- `data`
- `objective`
- `runtime`

Important defaults:

- total compile budget: `500` backend invocations
- beam width: `8`
- max search rounds: `6`
- max nodes per program: `7`
- max route nodes: `2`
- max branch fanout: `3`
- screening examples: `8`
- narrowing examples: `32`
- confirmation examples: `64`

Load config programmatically:

```python
from topoprompt.config import load_config

config = load_config("./configs/topoprompt_v1.yaml")
```

## Benchmarks

The repository includes benchmark adapters for:

- GSM8K
- MMLU
- BBH
- IFEval

There are two practical ways to use them:

- pass a local JSONL file through the CLI or `BenchmarkRunner`
- let the adapter load a Hugging Face dataset split directly

Smoke fixtures for all four benchmarks are included in [`tests/fixtures/smoke`](tests/fixtures/smoke).

## Artifact Format

Each compile run writes a research-friendly output directory.

### `final_program.json`

Canonical machine-readable `PromptProgram`.

### `final_program.yaml`

Human-readable export of the final IR.

### `candidate_archive.jsonl`

One record per evaluated candidate, including:

- `program_id`
- `parent_id`
- `edit_applied`
- `topology_fingerprint`
- `family_signature`
- stage-specific scores
- `search_score`
- `complexity`
- `inference_cost`
- `parse_failure_rate`

### `compile_trace.jsonl`

One execution trace per evaluated example, including node traces, token usage, latency, route choices, and parse failures.

### `metrics.json`

Run summary metrics, including:

- best and smallest-effective program IDs
- validation scores
- epsilon
- planned and spent budget
- per-phase budget ledger
- winning topology family
- beam family counts by round
- parser failure rate
- route accuracy and regret when available

### `transfer_trace_store.jsonl`

V1-ready trace feature store for future transfer and uncertainty work.

## DSPy Export

DSPy export is intentionally isolated from the internal IR.

If the `dspy` extra is installed, the compiler attempts to create a DSPy-compatible adapter object. If DSPy is not installed, `dspy_program` is `None`.

Current mapping intent:

- `direct` -> `dspy.Predict`
- `plan` -> `dspy.ChainOfThought`
- `solve` -> `dspy.ChainOfThought`
- `verify` -> `dspy.Predict`
- `route` -> custom wrapper

## Testing

Run the full test suite:

```bash
pytest
```

The current suite covers:

- topology fingerprint stability
- description-length behavior
- DAG and route validation
- runtime execution of hand-authored programs
- parser fallback behavior
- route behavior
- smallest-effective selection
- end-to-end compile smoke test
- smoke benchmark compilation for GSM8K, MMLU, BBH, and IFEval

All smoke tests run against the fake backend, so they do not require network access or real API calls.

## Example Program Flow

A typical compile run looks like this:

1. Read task description and examples.
2. Partition compile examples into a fixed few-shot pool, search examples, and validation examples.
3. Analyze the task to infer plausible topology priors.
4. Instantiate a seed library, always keeping a direct baseline available.
5. Evaluate seeds on a small shard.
6. Re-seed from the full library if analyzer priors underperform the direct baseline badly enough.
7. Search over structure and module edits with a hybrid beam.
8. Confirm strong candidates on validation examples.
9. Choose the smallest effective program.
10. Export the final IR, traces, archive, and metrics.

## Limitations

This repository is a V1 prototype, not a production orchestration system.

Current limitations:

- live OpenAI execution path is implemented but not part of the automated test suite
- search is intentionally small and bounded rather than exhaustive
- `classifier_route` is not yet backed by a real learned classifier
- `critique` and full standalone `format` behavior are deferred
- transfer, posterior ranking, and acquisition are stubs in V1
- benchmark adapters are lightweight and meant for research iteration, not hardened dataset ops

## Development Notes

If you extend the system, keep these boundaries intact:

- the IR should remain independent of DSPy
- topology must stay a compiler-owned variable
- complexity should influence final selection, not just prompt text
- traces and candidate archives should remain stable research artifacts
- any new backend should conform to the `LLMBackend` interface

## Roadmap

Planned next layers after V1:

- real transfer-aware ranking from stored compile traces
- uncertainty-aware edit prioritization
- stronger DSPy export coverage
- classifier-route training from induced supervision
- richer prompt-module rewriting
- broader benchmark automation and reporting

## License

This repository currently ships without a dedicated `LICENSE` file beyond the package metadata. Add one before external distribution.
