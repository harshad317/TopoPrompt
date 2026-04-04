from topoprompt.config import TopoPromptConfig, load_config
from topoprompt.eval.compare import compare_programs
from topoprompt.eval.dspy_baselines import (
    compare_topoprompt_vs_dspy,
    compile_dspy_baseline,
    evaluate_dspy_program_on_examples,
    load_dspy_program,
)
from topoprompt.eval.significance import summarize_significance_from_compare_dir
from topoprompt.schemas import CompileArtifact, Example, PromptProgram, TaskSpec


def compile_task(*args, **kwargs):
    from topoprompt.compiler.search import compile_task as _compile_task

    return _compile_task(*args, **kwargs)

__all__ = [
    "CompileArtifact",
    "Example",
    "PromptProgram",
    "TaskSpec",
    "TopoPromptConfig",
    "compare_topoprompt_vs_dspy",
    "compare_programs",
    "compile_task",
    "compile_dspy_baseline",
    "evaluate_dspy_program_on_examples",
    "load_config",
    "load_dspy_program",
    "summarize_significance_from_compare_dir",
]
