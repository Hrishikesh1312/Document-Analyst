from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Any

from document_analyst.config import AppSettings

# Presets affect document preparation; batch size is a separate hardware control.
INDEXING_PRESETS = {
    "Fast": {"chunking_mode": "fast", "chunk_size": 1000, "chunk_overlap": 100},
    "Balanced": {"chunking_mode": "fast", "chunk_size": 700, "chunk_overlap": 100},
    "Quality": {"chunking_mode": "semantic", "chunk_size": 500, "chunk_overlap": 100},
}


def apply_indexing_preset(settings: AppSettings, preset: str, batch_size: int) -> AppSettings:
    changes = {} if preset == "Keep current" else INDEXING_PRESETS[preset]
    return replace(settings, **changes, embedding_batch_size=batch_size)


@dataclass(frozen=True)
class BatchBenchmark:
    batch_size: int
    seconds: float | None
    passages_per_second: float | None
    error: str = ""


def benchmark_embedding_batches(
    embedder: Any,
    update: Callable[[int, int, int], None] | None = None,
    candidates: tuple[int, ...] = (8, 16, 32, 64, 128),
) -> list[BatchBenchmark]:
    """Compare batches on the same synthetic workload without changing settings."""
    texts = [
        ("This document explains local search, evidence, and indexing performance. " * (1 + i % 6)).strip()
        for i in range(128)
    ]
    options = dict(show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    # Warm up separately so model startup does not penalize the first candidate.
    embedder.encode(texts[:8], batch_size=8, **options)
    results = []
    for index, size in enumerate(candidates, start=1):
        if update:
            update(index, len(candidates), size)
        try:
            started = time.perf_counter()
            # NumPy conversion transfers results to host memory before returning,
            # so measurements include completed CUDA/MPS work, not dispatch time.
            for _ in range(2):
                embedder.encode(texts, batch_size=size, **options)
            seconds = max(time.perf_counter() - started, 1e-9)
            results.append(BatchBenchmark(size, seconds, 2 * len(texts) / seconds))
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            results.append(BatchBenchmark(size, None, None, "Insufficient device memory"))
            # Larger batches will require at least as much memory.
            break
    return results
