from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SETTINGS_PATH = DATA_DIR / "settings.json"
CHROMA_DIR = DATA_DIR / "chroma"


@dataclass(slots=True)
class AppSettings:
    documents_dir: str = ""
    chroma_dir: str = str(CHROMA_DIR)
    embeddings_dir: str = str(MODELS_DIR / "embeddings" / "all-MiniLM-L6-v2")
    embeddings_repo: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_dir: str = str(MODELS_DIR / "llm")
    llm_repo: str = "lmstudio-community/Llama-3.2-1B-Instruct-GGUF"
    llm_filename: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 4
    max_history_turns: int = 5
    max_file_size_mb: int = 50
    semantic_threshold: float = 0.5
    system_prompt: str = (
        "You are a privacy-first local document assistant. Answer using the provided "
        "document context. When the context is incomplete, say so clearly. Cite sources "
        "inline using [S1], [S2], etc. Avoid making up facts."
    )
    supported_extensions: list[str] = field(
        default_factory=lambda: [".pdf", ".md", ".markdown", ".txt"]
    )


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "cache").mkdir(parents=True, exist_ok=True)


def load_settings() -> AppSettings:
    ensure_dirs()
    if not SETTINGS_PATH.exists():
        settings = AppSettings()
        save_settings(settings)
        return settings
    return AppSettings(**json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))


def save_settings(settings: AppSettings) -> None:
    ensure_dirs()
    SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
