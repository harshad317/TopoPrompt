from __future__ import annotations

from types import SimpleNamespace

from topoprompt.backends.llm_client import FakeBackend
from topoprompt.backends.openai_backend import OpenAIBackend


class _StubEmbeddingsAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def create(self, *, input: str, model: str):
        self.calls.append((input, model))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


class _StubOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _StubEmbeddingsAPI()


def test_openai_backend_embed_text_uses_embeddings_api():
    client = _StubOpenAIClient()
    backend = OpenAIBackend(client=client)

    embedding = backend.embed_text("embed this task", model="text-embedding-3-small")

    assert embedding == [0.1, 0.2, 0.3]
    assert client.embeddings.calls == [("embed this task", "text-embedding-3-small")]
    assert backend.embeddings_are_real() is True


def test_fake_backend_embeddings_are_stub_by_default_and_opt_in_when_requested():
    default_backend = FakeBackend()
    semantic_backend = FakeBackend(
        embed_handler=lambda text, model: [1.0, 0.0],
        embeddings_are_real=True,
    )

    assert default_backend.embeddings_are_real() is False
    assert semantic_backend.embed_text("semantic query", model="text-embedding-3-small") == [1.0, 0.0]
    assert semantic_backend.embeddings_are_real() is True
