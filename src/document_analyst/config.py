from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
SETTINGS_PATH = DATA_DIR / "settings.json"
CHROMA_DIR = DATA_DIR / "chroma"

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
    return repo_id.replace("/", "--").replace(" ", "-")


@dataclass(slots=True)
class AppSettings:
    documents_dir: str = ""
    chroma_dir: str = str(CHROMA_DIR)
    embeddings_repo: str = EMBEDDING_REPO_OPTIONS[0]
    embeddings_dir: str = str(MODELS_DIR / "embeddings" / repo_storage_name(EMBEDDING_REPO_OPTIONS[0]))
    llm_repo: str = LLM_REPO_OPTIONS[0]
    llm_dir: str = str(MODELS_DIR / "llm" / repo_storage_name(LLM_REPO_OPTIONS[0]))
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
    settings = AppSettings(**json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    sync_model_paths(settings)
    return settings


def save_settings(settings: AppSettings) -> None:
    ensure_dirs()
    sync_model_paths(settings)
    SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def sync_model_paths(settings: AppSettings) -> None:
    settings.embeddings_dir = str(MODELS_DIR / "embeddings" / repo_storage_name(settings.embeddings_repo))
    settings.llm_dir = str(MODELS_DIR / "llm" / repo_storage_name(settings.llm_repo))
