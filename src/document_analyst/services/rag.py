from __future__ import annotations

import gc
import hashlib
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from llama_cpp import Llama

from document_analyst.config import AppSettings
from document_analyst.models import ChunkRecord, RetrievalDiagnostics, SourceRecord
from document_analyst.services.chroma_store import ChromaStore
from document_analyst.services.ingestion import DocumentIngestor
from document_analyst.services.ingestion import IndexProgress
from document_analyst.services.index_manifest import IndexManifest
from document_analyst.services.model_manager import ModelManager
from document_analyst.services.model_manager import DownloadProgress


@dataclass(slots=True)
class IndexResult:
    document_count: int
    chunk_count: int
    warnings: list[str]
    unchanged_count: int = 0
    duplicate_count: int = 0
    failed_count: int = 0
    removed_count: int = 0
    cancelled: bool = False


@dataclass(slots=True)
class AnswerResult:
    answer: str
    sources: list[SourceRecord]
    diagnostics: RetrievalDiagnostics | None = None


@dataclass(slots=True)
class AnswerStream:
    chunks: Iterator[str]
    sources: list[SourceRecord]
    diagnostics: RetrievalDiagnostics | None = None


class RagService:
    LLM_CONTEXT_WINDOW = 4096
    LLM_MAX_OUTPUT_TOKENS = 700
    PROMPT_TOKEN_BUDGET = 3000

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.models = ModelManager(settings)
        self.ingestor = DocumentIngestor(settings)
        self.store = ChromaStore(settings)
        self.manifest = IndexManifest()
        self._embedder = None
        self._llm = None

    def download_models(self, progress: DownloadProgress | None = None) -> tuple[str, str]:
        embedding_dir = self.models.ensure_embedding_model(progress=progress)
        llm_path = self.models.ensure_llm_model(progress=progress)
        return str(embedding_dir), str(llm_path)

    def ensure_embedder(self):
        if self._embedder is None:
            self._embedder = self.models.load_embedder(download_if_missing=True)
        return self._embedder

    def ensure_llm(self) -> Any:
        if self._llm is not None:
            return self._llm

        llm_path = self.models.local_llm_model_path()
        if llm_path is None:
            llm_path = self.models.ensure_llm_model()

        threads = max(2, (os.cpu_count() or 4) - 1)
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(llm_path),
            n_ctx=self.LLM_CONTEXT_WINDOW,
            n_threads=threads,
            n_batch=512,
            verbose=False,
        )
        return self._llm

    def index_documents(
        self,
        directory: str,
        replace_existing: bool = False,
        progress: IndexProgress | None = None,
        should_cancel: Callable[[], bool] | None = None,
        retry_failed_only: bool = False,
    ) -> IndexResult:
        if replace_existing:
            self.store.reset()
            self.manifest.reset()

        root = Path(directory).expanduser().resolve()
        paths = self.ingestor.discover(directory)
        entries = self.manifest.load()
        warnings: list[str] = []
        discovered_paths = {str(path.resolve()) for path in paths}
        removed_count = 0
        for source_path, entry in list(entries.items()):
            try:
                is_inside_root = Path(source_path).is_relative_to(root)
            except (OSError, ValueError):
                is_inside_root = False
            if is_inside_root and source_path not in discovered_paths and entry.get("status") != "removed":
                self.store.delete_document(source_path)
                entry.update(status="removed", error="File no longer exists", updated_at=self.manifest.now())
                removed_count += 1

        total = len(paths)
        indexed_count = unchanged_count = duplicate_count = failed_count = chunk_count = 0
        cancelled = False
        hashes: dict[str, str] = {}
        for index, path in enumerate(paths, start=1):
            if should_cancel and should_cancel():
                cancelled = True
                break
            if progress:
                progress("hashing", path.name, index, total, False)
            try:
                hashes[str(path.resolve())] = self._hash_file(path, should_cancel)
            except RuntimeError as exc:
                if "cancelled" not in str(exc).lower():
                    raise
                cancelled = True
                break
            if progress:
                progress("hashing", path.name, index, total, True)

        canonical_by_hash: dict[str, str] = {}
        for path in paths:
            source_path = str(path.resolve())
            previous = entries.get(source_path, {})
            current_hash = hashes.get(source_path)
            if (
                current_hash
                and previous.get("status") == "indexed"
                and previous.get("file_hash") == current_hash
            ):
                canonical_by_hash.setdefault(current_hash, source_path)
        for index, path in enumerate(paths, start=1):
            source_path = str(path.resolve())
            if source_path not in hashes:
                break
            if should_cancel and should_cancel():
                cancelled = True
                break
            file_hash = hashes[source_path]
            previous = entries.get(source_path, {})
            if retry_failed_only and previous.get("status") not in {"failed", "cancelled"}:
                unchanged_count += 1
                continue
            duplicate_of = canonical_by_hash.get(file_hash)
            if duplicate_of and duplicate_of != source_path:
                self.store.delete_document(source_path)
                entries[source_path] = self._manifest_entry(
                    path, file_hash, "duplicate", duplicate_of=duplicate_of
                )
                duplicate_count += 1
                self.manifest.save(entries)
                continue
            if (
                not replace_existing
                and previous.get("file_hash") == file_hash
                and previous.get("status") == "indexed"
            ):
                unchanged_count += 1
                canonical_by_hash.setdefault(file_hash, source_path)
                continue

            try:
                if progress:
                    progress("reading", path.name, index, total, False)
                document = self.ingestor.load_document(path)
                document.file_hash = file_hash
                if progress:
                    progress("reading", path.name, index, total, True)
                if not document.text:
                    raise ValueError("no readable text found")
                chunks = self.ingestor.build_chunks([document], self._embed_texts)
                if not chunks:
                    raise ValueError("no searchable chunks were produced")
                indexed_at = self.manifest.now()
                for chunk in chunks:
                    chunk.indexed_at = indexed_at
                if progress:
                    progress("embedding", path.name, index, total, False)
                embeddings = self._embed_texts([chunk.text for chunk in chunks])
                if progress:
                    progress("embedding", path.name, index, total, True)
                if should_cancel and should_cancel():
                    entries[source_path] = self._manifest_entry(
                        path, file_hash, "cancelled", error="Cancelled before database write"
                    )
                    cancelled = True
                    self.manifest.save(entries)
                    break
                if progress:
                    progress("writing", path.name, index, total, False)
                self.store.upsert_chunks(chunks, embeddings)
                if progress:
                    progress("writing", path.name, index, total, True)
                entries[source_path] = self._manifest_entry(
                    path, file_hash, "indexed", chunks=len(chunks), indexed_at=indexed_at
                )
                canonical_by_hash[file_hash] = source_path
                indexed_count += 1
                chunk_count += len(chunks)
            except Exception as exc:  # Keep other files indexable and record a retryable failure.
                if previous.get("status") == "indexed" and previous.get("file_hash") != file_hash:
                    self.store.delete_document(source_path)
                entries[source_path] = self._manifest_entry(
                    path, file_hash, "failed", error=str(exc)
                )
                warnings.append(f"Failed {path.name}: {exc}")
                failed_count += 1
            self.manifest.save(entries)

        self.manifest.save(entries)
        return IndexResult(
            document_count=indexed_count,
            chunk_count=chunk_count,
            warnings=warnings,
            unchanged_count=unchanged_count,
            duplicate_count=duplicate_count,
            failed_count=failed_count,
            removed_count=removed_count,
            cancelled=cancelled,
        )

    def _hash_file(
        self, path: Path, should_cancel: Callable[[], bool] | None = None
    ) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                if should_cancel and should_cancel():
                    raise RuntimeError("Indexing cancelled while hashing")
                digest.update(block)
        return digest.hexdigest()

    def _manifest_entry(
        self,
        path: Path,
        file_hash: str,
        status: str,
        *,
        chunks: int = 0,
        error: str = "",
        duplicate_of: str = "",
        indexed_at: str = "",
    ) -> dict[str, object]:
        stat = path.stat()
        return {
            "document_name": path.name,
            "source_path": str(path.resolve()),
            "file_hash": file_hash,
            "size_bytes": stat.st_size,
            "modified_at": stat.st_mtime,
            "indexed_at": indexed_at,
            "updated_at": self.manifest.now(),
            "status": status,
            "chunks": chunks,
            "error": error,
            "duplicate_of": duplicate_of,
        }

    def retrieve_sources(
        self,
        question: str,
        source_paths: list[str] | None = None,
        pinned_source_paths: list[str] | None = None,
        excluded_source_paths: list[str] | None = None,
    ) -> list[SourceRecord]:
        sources, _ = self.retrieve_with_diagnostics(
            question, source_paths, pinned_source_paths, excluded_source_paths
        )
        return sources

    def retrieve_with_diagnostics(
        self,
        question: str,
        source_paths: list[str] | None = None,
        pinned_source_paths: list[str] | None = None,
        excluded_source_paths: list[str] | None = None,
    ) -> tuple[list[SourceRecord], RetrievalDiagnostics]:
        retrieval_started = time.perf_counter()
        embedding_started = time.perf_counter()
        self.ensure_embedder()
        query_embedding = self._embed_texts([question])[0]
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        result = self.store.hybrid_query(
            question,
            query_embedding,
            self.settings.top_k,
            source_paths=source_paths,
            pinned_source_paths=pinned_source_paths,
            excluded_source_paths=excluded_source_paths,
            min_score=self.settings.retrieval_min_score,
        )
        total_ms = (time.perf_counter() - retrieval_started) * 1000
        diagnostics = RetrievalDiagnostics(
            query=question,
            scope=list(source_paths or []),
            candidate_count=len(result.candidates),
            selected_count=len(result.sources),
            documents_covered=len({source.source_path for source in result.sources}),
            embedding_ms=embedding_ms,
            semantic_ms=result.semantic_ms,
            lexical_ms=result.lexical_ms,
            rerank_ms=result.rerank_ms,
            total_ms=total_ms,
            candidates=result.candidates,
        )
        return result.sources, diagnostics

    def answer_question(
        self,
        question: str,
        history: list[dict[str, str]],
        source_paths: list[str] | None = None,
        pinned_source_paths: list[str] | None = None,
        excluded_source_paths: list[str] | None = None,
    ) -> AnswerResult:
        streamed = self.answer_question_stream(
            question, history, source_paths, pinned_source_paths, excluded_source_paths
        )
        return AnswerResult(
            answer="".join(streamed.chunks),
            sources=streamed.sources,
            diagnostics=streamed.diagnostics,
        )

    def answer_question_stream(
        self,
        question: str,
        history: list[dict[str, str]],
        source_paths: list[str] | None = None,
        pinned_source_paths: list[str] | None = None,
        excluded_source_paths: list[str] | None = None,
    ) -> AnswerStream:
        sources, diagnostics = self.retrieve_with_diagnostics(
            question, source_paths, pinned_source_paths, excluded_source_paths
        )
        if not sources:
            message = (
                "I could not find sufficiently relevant content in the selected documents. "
                "Try different wording, select more documents, or lower the minimum relevance score in Models & Settings."
            )
            return AnswerStream(
                chunks=iter([message]),
                sources=[],
                diagnostics=diagnostics,
            )

        llm_path = self.models.local_llm_model_path()
        if llm_path is None:
            return AnswerStream(
                chunks=iter([self._fallback_answer(question, sources)]),
                sources=sources,
                diagnostics=diagnostics,
            )

        llm = self.ensure_llm()
        prompt = self._fit_prompt_to_context(llm, question, sources, history)

        def generate() -> Iterator[str]:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.settings.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=self.LLM_MAX_OUTPUT_TOKENS,
                stream=True,
            )
            for event in response:
                choices = event.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if content:
                    yield str(content)

        return AnswerStream(chunks=generate(), sources=sources, diagnostics=diagnostics)

    def indexed_documents(self) -> list[dict[str, object]]:
        return self.store.indexed_documents()

    def document_statuses(self) -> list[dict[str, object]]:
        entries = self.manifest.load()
        if not entries:
            return self.store.indexed_documents()
        return sorted(
            entries.values(), key=lambda item: str(item.get("document_name", "")).lower()
        )

    def delete_document(self, source_path: str) -> None:
        self.store.delete_document(source_path)
        entries = self.manifest.load()
        entry = entries.get(source_path)
        if entry:
            entry.update(status="removed", error="Removed by user", updated_at=self.manifest.now())
            self.manifest.save(entries)

    def stats(self) -> dict[str, int]:
        return self.store.stats()

    def unload_models(self) -> None:
        """Release in-memory model handles before their files are removed."""
        self._embedder = None
        self._llm = None
        gc.collect()

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embedder = self.ensure_embedder()
        matrix = embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return matrix.tolist()

    def _build_prompt(
        self,
        question: str,
        sources: list[SourceRecord],
        history: list[dict[str, str]],
        source_char_limit: int | None = None,
        history_char_limit: int | None = None,
    ) -> str:
        history_window = history[-self.settings.max_history_turns * 2 :]
        history_text = "\n".join(
            f"{item['role'].title()}: {self._truncate_text(item['content'], history_char_limit)}"
            for item in history_window
        )
        source_text = "\n\n".join(
            f"[{source.source_id}] {source.document_name} (page {source.approx_page})\n"
            f"{self._truncate_text(source.text, source_char_limit)}"
            for source in sources
        )
        sections: list[str] = []
        if history_text:
            sections.append(f"Conversation history:\n{history_text}")
        sections.append(f"Retrieved context:\n{source_text}")
        sections.append(f"Question: {question}")
        sections.append(
            "Answer grounded in the retrieved context. Do not mention internal prompt structure. Cite source ids inline."
        )
        return "\n\n".join(sections)

    def _fallback_answer(self, question: str, sources: list[SourceRecord]) -> str:
        lines = [
            "The local GGUF chat model has not been downloaded yet, so this is a retrieval-only response.",
            f"I found {len(sources)} relevant source snippets for: '{question}'",
            "",
        ]
        for source in sources[:3]:
            lines.append(
                f"[{source.source_id}] {source.document_name} (page {source.approx_page}): {source.text[:260]}..."
            )
        lines.append("")
        lines.append("Open `Models & Settings` and download the recommended GGUF to enable full local generation.")
        return "\n".join(lines)

    def _fit_prompt_to_context(
        self,
        llm: Any,
        question: str,
        sources: list[SourceRecord],
        history: list[dict[str, str]],
    ) -> str:
        source_limit = 900
        history_limit = 350
        prompt = self._build_prompt(
            question,
            sources,
            history,
            source_char_limit=source_limit,
            history_char_limit=history_limit,
        )

        while self._token_count(llm, prompt) > self.PROMPT_TOKEN_BUDGET:
            if source_limit > 220:
                source_limit = max(220, source_limit - 160)
            elif history_limit > 120:
                history_limit = max(120, history_limit - 60)
            elif len(sources) > 2:
                sources = sources[: len(sources) - 1]
            elif len(history) > 2:
                history = history[2:]
            else:
                prompt = self._build_prompt(
                    question,
                    sources[:2],
                    [],
                    source_char_limit=180,
                    history_char_limit=80,
                )
                break

            prompt = self._build_prompt(
                question,
                sources,
                history,
                source_char_limit=source_limit,
                history_char_limit=history_limit,
            )
        return prompt

    def _token_count(self, llm: Any, text: str) -> int:
        return len(llm.tokenize(text.encode("utf-8")))

    def _truncate_text(self, text: str, limit: int | None) -> str:
        if limit is None or len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."
