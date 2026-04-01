def compile_task(*args, **kwargs):
    from topoprompt.compiler.search import compile_task as _compile_task

    return _compile_task(*args, **kwargs)


__all__ = ["compile_task"]
