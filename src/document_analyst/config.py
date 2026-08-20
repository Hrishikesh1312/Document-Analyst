from __future__ import annotations

import json
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

APP_NAME = "Document Analyst"
APP_SLUG = "document-analyst"


def _user_data_dir() -> Path:
    """Return an OS-native, user-writable application data directory."""
    override = os.environ.get("DOCUMENT_ANALYST_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if platform.system() == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / APP_NAME
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_SLUG


APP_DATA_DIR = _user_data_dir()
MODELS_DIR = APP_DATA_DIR / "models"
SETTINGS_PATH = APP_DATA_DIR / "settings.json"
CHROMA_DIR = APP_DATA_DIR / "chroma"

EMBEDDING_REPO_OPTIONS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
]
LLM_REPO_OPTIONS = [
    "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
    "lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF",
    "lmstudio-community/Phi-3.5-mini-instruct-GGUF",
]


def repo_storage_name(repo_id: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in repo_id).strip(".-")


@dataclass(slots=True)
class AppSettings:
    documents_dir: str = ""
    chroma_dir: str = str(CHROMA_DIR)
    embeddings_repo: str = EMBEDDING_REPO_OPTIONS[0]
    embeddings_dir: str = ""
    llm_repo: str = LLM_REPO_OPTIONS[0]
    llm_dir: str = ""
    llm_filename: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 4
    max_history_turns: int = 5
    max_file_size_mb: int = 50
    semantic_threshold: float = 0.5
    enable_ocr: bool = False
    ocr_min_text_chars: int = 80
    ocr_zoom: float = 2.0
    tesseract_cmd: str = ""
    system_prompt: str = (
        "You are a privacy-first local document assistant. Answer using the provided "
        "document context. When the context is incomplete, say so clearly. Cite sources "
        "inline using [S1], [S2], etc. Avoid making up facts."
    )
    supported_extensions: list[str] = field(
        default_factory=lambda: [".pdf", ".md", ".markdown", ".txt"]
    )

    def __post_init__(self) -> None:
        self.validate()
        sync_model_paths(self)

    def validate(self) -> None:
        self.chunk_size = max(100, min(int(self.chunk_size), 20_000))
        self.chunk_overlap = max(0, min(int(self.chunk_overlap), self.chunk_size - 1))
        self.top_k = max(1, min(int(self.top_k), 50))
        self.max_history_turns = max(0, min(int(self.max_history_turns), 50))
        self.max_file_size_mb = max(1, min(int(self.max_file_size_mb), 2_048))
        self.semantic_threshold = max(-1.0, min(float(self.semantic_threshold), 1.0))
        self.ocr_min_text_chars = max(0, min(int(self.ocr_min_text_chars), 10_000))
        self.ocr_zoom = max(1.0, min(float(self.ocr_zoom), 5.0))
        self.supported_extensions = sorted({
            ext if ext.startswith(".") else f".{ext}"
            for ext in (str(value).lower() for value in self.supported_extensions)
        })


def ensure_dirs() -> None:
    for path in (APP_DATA_DIR, MODELS_DIR, CHROMA_DIR, APP_DATA_DIR / "cache"):
        path.mkdir(parents=True, exist_ok=True)


def load_settings() -> AppSettings:
    ensure_dirs()
    if not SETTINGS_PATH.exists():
        settings = AppSettings()
        save_settings(settings)
        return settings
    try:
        raw: Any = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("settings must contain a JSON object")
        allowed = {item.name for item in fields(AppSettings)}
        settings = AppSettings(**{key: value for key, value in raw.items() if key in allowed})
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        _backup_invalid_settings()
        settings = AppSettings()
        save_settings(settings)
    return settings


def save_settings(settings: AppSettings) -> None:
    ensure_dirs()
    settings.validate()
    sync_model_paths(settings)
    payload = json.dumps(asdict(settings), indent=2, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="settings-", suffix=".tmp", dir=APP_DATA_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, SETTINGS_PATH)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sync_model_paths(settings: AppSettings) -> None:
    settings.embeddings_dir = str(MODELS_DIR / "embeddings" / repo_storage_name(settings.embeddings_repo))
    settings.llm_dir = str(MODELS_DIR / "llm" / repo_storage_name(settings.llm_repo))
    configured = Path(settings.chroma_dir).expanduser() if settings.chroma_dir else CHROMA_DIR
    if not configured.is_absolute() or not _is_writable_parent(configured):
        configured = CHROMA_DIR
    settings.chroma_dir = str(configured)


def _is_writable_parent(path: Path) -> bool:
    parent = next((candidate for candidate in (path, *path.parents) if candidate.exists()), None)
    return bool(parent and parent.is_dir() and os.access(parent, os.W_OK))


def _backup_invalid_settings() -> None:
    try:
        SETTINGS_PATH.replace(SETTINGS_PATH.with_suffix(".json.invalid"))
    except OSError:
        pass
