from __future__ import annotations

from typing import Any

from openai import OpenAI

from topoprompt.backends.llm_client import BackendResponse, LLMBackend


class OpenAIBackend(LLMBackend):
    def __init__(self, client: OpenAI | None = None) -> None:
        self.client = client or OpenAI()

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
    ) -> BackendResponse:
        response = self.client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        usage = getattr(response, "usage", None)
        token_usage = {
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        }
        output_text = getattr(response, "output_text", "")
        return BackendResponse(
            text=output_text,
            structured=None,
            token_usage=token_usage,
            latency_ms=0,
            request_id=getattr(response, "id", None),
        )

    def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 800,
    ) -> BackendResponse:
        augmented_prompt = (
            f"{user_prompt}\n\nReturn strict JSON that matches this schema exactly:\n{schema}"
        )
        return self.generate_text(
            system_prompt=system_prompt,
            user_prompt=augmented_prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

