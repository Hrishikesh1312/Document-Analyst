from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from document_analyst.config import APP_DATA_DIR


class ConversationStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (APP_DATA_DIR / "conversations.json")
        self._lock = threading.RLock()

    def list(self) -> list[dict[str, Any]]:
        return sorted(
            self._load().values(),
            key=lambda item: str(item.get("updated_at", "")),
            reverse=True,
        )

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        return self._load().get(conversation_id)

    def create(self, name: str = "New conversation") -> dict[str, Any]:
        conversations = self._load()
        now = self._now()
        item = {
            "id": uuid.uuid4().hex,
            "name": name.strip() or "New conversation",
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "sources_by_turn": {},
        }
        conversations[item["id"]] = item
        self._save(conversations)
        return item

    def save_conversation(
        self,
        conversation_id: str,
        name: str,
        messages: list[dict[str, str]],
        sources_by_turn: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        conversations = self._load()
        existing = conversations.get(conversation_id, {})
        now = self._now()
        item = {
            "id": conversation_id,
            "name": name.strip() or "New conversation",
            "created_at": existing.get("created_at", now),
            "updated_at": now,
            "messages": messages,
            "sources_by_turn": sources_by_turn,
        }
        conversations[conversation_id] = item
        self._save(conversations)
        return item

    def rename(self, conversation_id: str, name: str) -> None:
        conversations = self._load()
        if conversation_id not in conversations:
            raise KeyError("Conversation not found.")
        conversations[conversation_id]["name"] = name.strip() or "Untitled conversation"
        conversations[conversation_id]["updated_at"] = self._now()
        self._save(conversations)

    def delete(self, conversation_id: str) -> None:
        conversations = self._load()
        conversations.pop(conversation_id, None)
        self._save(conversations)

    def _load(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return {}
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                return {}
            conversations = payload.get("conversations", {}) if isinstance(payload, dict) else {}
            return conversations if isinstance(conversations, dict) else {}

    def _save(self, conversations: dict[str, dict[str, Any]]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {"version": 1, "conversations": conversations}, indent=2, ensure_ascii=False
            ) + "\n"
            fd, temporary = tempfile.mkstemp(
                prefix="conversations-", suffix=".tmp", dir=self.path.parent
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
