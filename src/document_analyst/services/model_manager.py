from __future__ import annotations

import logging
import warnings
from pathlib import Path
from collections.abc import Callable
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
from tqdm.auto import tqdm

# `sentence-transformers` can trigger noisy upstream Transformers messages about
# `__path__` on import (e.g., from zoedepth, maskformer). These are harmless for
# this app's text-only usage. Depending on the library version, the message may
# arrive as a warning or as a logger record, so we filter both forms.
warnings.filterwarnings(
    "ignore",
    message=r".*Accessing `__path__` from `.models\..*image_processing_.*`.*",
)


class _TransformersPathWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "Accessing `__path__` from `.models." not in message


logging.getLogger("transformers").addFilter(_TransformersPathWarningFilter())

from document_analyst.config import AppSettings

DownloadProgress = Callable[[str, str, int, int | None], None]


def _progress_tqdm(progress: DownloadProgress, phase: str) -> type[tqdm]:
    """Build the tqdm class Hugging Face uses for byte-level file transfers."""

    class ProgressTqdm(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            progress(phase, str(self.desc or "Preparing download"), int(self.n), self.total)

        def update(self, n: int = 1) -> bool | None:
            displayed = super().update(n)
            progress(phase, str(self.desc or "Downloading"), int(self.n), self.total)
            return displayed

        def close(self) -> None:
            if not self.disable:
                progress(phase, str(self.desc or "Download complete"), int(self.n), self.total)
            super().close()

    return ProgressTqdm


class ModelManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def ensure_embedding_model(self, progress: DownloadProgress | None = None) -> Path:
        target = Path(self.settings.embeddings_dir)
        if self._embedding_download_complete(target):
            if progress:
                progress("embeddings", "Already downloaded", 1, 1)
            return target

        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.settings.embeddings_repo,
            local_dir=str(target),
            max_workers=1 if progress else 8,
            tqdm_class=_progress_tqdm(progress, "embeddings") if progress else None,
        )
        if progress:
            progress("embeddings", "Embedding model ready", 1, 1)
        return target

    def ensure_llm_model(self, progress: DownloadProgress | None = None) -> Path:
        llm_dir = Path(self.settings.llm_dir)
        llm_dir.mkdir(parents=True, exist_ok=True)

        if self.settings.llm_filename:
            target = self._safe_model_path(llm_dir, self.settings.llm_filename)
            if target.exists():
                if progress:
                    progress("llm", "Already downloaded", 1, 1)
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
            tqdm_class=_progress_tqdm(progress, "llm") if progress else None,
        )
        if progress:
            progress("llm", "Language model ready", 1, 1)
        return Path(downloaded)

    def local_embedding_model_exists(self) -> bool:
        target = Path(self.settings.embeddings_dir)
        return self._embedding_download_complete(target)

    def local_llm_model_path(self) -> Path | None:
        llm_dir = Path(self.settings.llm_dir)
        if self.settings.llm_filename:
            target = self._safe_model_path(llm_dir, self.settings.llm_filename)
            if target.exists():
                return target
        for candidate in sorted(llm_dir.rglob("*.gguf")):
            return candidate
        return None

    def load_embedder(self, download_if_missing: bool = True) -> Any:
        # Importing sentence-transformers eagerly loads PyTorch and native runtime
        # libraries. Delay that work so settings and document management still open
        # on systems where the optional ML runtime is not ready yet.
        from sentence_transformers import SentenceTransformer

        model_dir = self.ensure_embedding_model() if download_if_missing else Path(self.settings.embeddings_dir)
        if not model_dir.exists():
            raise FileNotFoundError("Embedding model not found locally.")
        return SentenceTransformer(str(model_dir), device="cpu", local_files_only=True)

    @staticmethod
    def _safe_model_path(directory: Path, filename: str) -> Path:
        target = (directory / filename).resolve()
        try:
            target.relative_to(directory.resolve())
        except ValueError as exc:
            raise ValueError("GGUF filename must stay inside the model directory.") from exc
        return target

    def _select_preferred_gguf(self, files: list[str]) -> str:
        ranked_patterns = ("Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_0")
        lowered = {name: name.lower() for name in files}
        for pattern in ranked_patterns:
            for original, lower in lowered.items():
                if pattern.lower() in lower:
                    return original
        return sorted(files)[0]

    @staticmethod
    def _embedding_download_complete(directory: Path) -> bool:
        if not directory.is_dir():
            return False
        return any(item.name != ".cache" for item in directory.iterdir())
