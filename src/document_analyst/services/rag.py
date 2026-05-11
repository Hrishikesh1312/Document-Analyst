from __future__ import annotations

import os
from dataclasses import dataclass

from llama_cpp import Llama

from document_analyst.config import AppSettings
from document_analyst.models import ChunkRecord, SourceRecord
from document_analyst.services.chroma_store import ChromaStore
from document_analyst.services.ingestion import DocumentIngestor
from document_analyst.services.model_manager import ModelManager


@dataclass(slots=True)
class IndexResult:
    document_count: int
    chunk_count: int
    warnings: list[str]


@dataclass(slots=True)
class AnswerResult:
    answer: str
    sources: list[SourceRecord]


class RagService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.models = ModelManager(settings)
        self.ingestor = DocumentIngestor(settings)
        self.store = ChromaStore(settings)
        self._embedder = None
        self._llm = None

    def download_models(self) -> tuple[str, str]:
        embedding_dir = self.models.ensure_embedding_model()
        llm_path = self.models.ensure_llm_model()
        return str(embedding_dir), str(llm_path)

    def ensure_embedder(self):
        if self._embedder is None:
            self._embedder = self.models.load_embedder(download_if_missing=True)
        return self._embedder

    def ensure_llm(self) -> Llama:
        if self._llm is not None:
            return self._llm

        llm_path = self.models.local_llm_model_path()
        if llm_path is None:
            llm_path = self.models.ensure_llm_model()

        threads = max(2, (os.cpu_count() or 4) - 1)
        self._llm = Llama(
            model_path=str(llm_path),
            n_ctx=4096,
            n_threads=threads,
            n_batch=512,
            verbose=False,
        )
        return self._llm

    def index_documents(self, directory: str, replace_existing: bool = False) -> IndexResult:
        embedder = self.ensure_embedder()
        documents, warnings = self.ingestor.load_documents(directory)
        chunks = self.ingestor.build_chunks(documents, self._embed_texts)
        embeddings = self._embed_texts([chunk.text for chunk in chunks])

        if replace_existing:
            self.store.reset()
        self.store.upsert_chunks(chunks, embeddings)
        return IndexResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            warnings=warnings,
        )

    def answer_question(self, question: str, history: list[dict[str, str]]) -> AnswerResult:
        embedder = self.ensure_embedder()
        _ = embedder
        query_embedding = self._embed_texts([question])[0]
        sources = self.store.query(query_embedding, self.settings.top_k)
        if not sources:
            return AnswerResult(
                answer="I could not find relevant indexed content yet. Add documents in the Manage Documents tab and try again.",
                sources=[],
            )

        prompt = self._build_prompt(question, sources, history)
        llm_path = self.models.local_llm_model_path()
        if llm_path is None:
            text = self._fallback_answer(question, sources)
        else:
            llm = self.ensure_llm()
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.settings.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=700,
            )
            text = response["choices"][0]["message"]["content"].strip()
        return AnswerResult(answer=text, sources=sources)

    def indexed_documents(self) -> list[dict[str, object]]:
        return self.store.indexed_documents()

    def delete_document(self, source_path: str) -> None:
        self.store.delete_document(source_path)

    def stats(self) -> dict[str, int]:
        return self.store.stats()

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
    ) -> str:
        history_window = history[-self.settings.max_history_turns * 2 :]
        history_text = "\n".join(
            f"{item['role'].title()}: {item['content']}" for item in history_window
        )
        source_text = "\n\n".join(
            f"[{source.source_id}] {source.document_name} (page {source.approx_page})\n{source.text}"
            for source in sources
        )
        return (
            "Conversation history:\n"
            f"{history_text or 'No prior conversation.'}\n\n"
            "Retrieved context:\n"
            f"{source_text}\n\n"
            f"Question: {question}\n\n"
            "Answer grounded in the retrieved context. Cite source ids inline."
        )

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
