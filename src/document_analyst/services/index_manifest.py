from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from document_analyst.config import APP_DATA_DIR


class IndexManifest:
    """Atomic, thread-safe operational state for indexed source files."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (APP_DATA_DIR / "index-manifest.json")
        self._lock = threading.RLock()

    def load(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return {}
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                return {}
            entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
            return entries if isinstance(entries, dict) else {}

    def save(self, entries: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"version": 1, "updated_at": self.now(), "entries": entries},
                indent=2,
                ensure_ascii=False,
            ) + "\n"
            fd, temporary = tempfile.mkstemp(
                prefix="index-manifest-", suffix=".tmp", dir=self.path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)

    def reset(self) -> None:
        self.save({})

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
