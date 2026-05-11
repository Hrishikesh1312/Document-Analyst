from __future__ import annotations

import warnings
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

# `sentence-transformers` can trigger a noisy upstream Transformers warning about
# `zoedepth.__path__` on import. It is harmless for this app's text-only usage.
warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `__path__` from `.models\.zoedepth\.image_processing_zoedepth`.*",
)
from sentence_transformers import SentenceTransformer

from document_analyst.config import AppSettings


class ModelManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def ensure_embedding_model(self) -> Path:
        target = Path(self.settings.embeddings_dir)
        if target.exists() and any(target.iterdir()):
            return target

        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.settings.embeddings_repo,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
        return target

    def ensure_llm_model(self) -> Path:
        llm_dir = Path(self.settings.llm_dir)
        llm_dir.mkdir(parents=True, exist_ok=True)

        if self.settings.llm_filename:
            target = llm_dir / self.settings.llm_filename
            if target.exists():
                return target

        repo_files = list_repo_files(self.settings.llm_repo)
        candidates = [name for name in repo_files if name.lower().endswith(".gguf")]
        if not candidates:
            raise FileNotFoundError(f"No GGUF files found in repo {self.settings.llm_repo}")

        preferred = self._select_preferred_gguf(candidates)
        downloaded = hf_hub_download(
            repo_id=self.settings.llm_repo,
            filename=preferred,
            local_dir=str(llm_dir),
            local_dir_use_symlinks=False,
        )
        return Path(downloaded)

    def local_embedding_model_exists(self) -> bool:
        target = Path(self.settings.embeddings_dir)
        return target.exists() and any(target.iterdir())

    def local_llm_model_path(self) -> Path | None:
        llm_dir = Path(self.settings.llm_dir)
        if self.settings.llm_filename:
            target = llm_dir / self.settings.llm_filename
            if target.exists():
                return target
        for candidate in llm_dir.glob("*.gguf"):
            return candidate
        return None

    def load_embedder(self, download_if_missing: bool = True) -> SentenceTransformer:
        model_dir = self.ensure_embedding_model() if download_if_missing else Path(self.settings.embeddings_dir)
        if not model_dir.exists():
            raise FileNotFoundError("Embedding model not found locally.")
        return SentenceTransformer(str(model_dir), device="cpu", local_files_only=True)

    def _select_preferred_gguf(self, files: list[str]) -> str:
        ranked_patterns = ("Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_0")
        lowered = {name: name.lower() for name in files}
        for pattern in ranked_patterns:
            for original, lower in lowered.items():
                if pattern.lower() in lower:
                    return original
        return sorted(files)[0]
