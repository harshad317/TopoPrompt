from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from diskcache import Cache

from topoprompt.backends.llm_client import BackendResponse


class RuntimeCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache = Cache(str(cache_dir))

    def key_for(self, *, model: str, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        payload = json.dumps(
            {
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> BackendResponse | None:
        payload = self.cache.get(key)
        if payload is None:
            return None
        return BackendResponse.model_validate(payload)

    def set(self, key: str, response: BackendResponse) -> None:
        self.cache.set(key, response.model_dump(mode="json"))

