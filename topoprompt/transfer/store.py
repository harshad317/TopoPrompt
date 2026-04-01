from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson


class TraceStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def flush(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = b"\n".join(orjson.dumps(record) for record in self.records)
        self.path.write_bytes(content + (b"\n" if self.records else b""))

