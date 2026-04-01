from topoprompt.config import TopoPromptConfig, load_config
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
    "compile_task",
    "load_config",
]
