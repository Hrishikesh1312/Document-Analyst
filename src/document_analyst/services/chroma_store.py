from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from document_analyst.config import AppSettings
from document_analyst.models import CandidateDiagnostic, ChunkRecord, SourceRecord


@dataclass(slots=True)
class HybridQueryResult:
    sources: list[SourceRecord]
    candidates: list[CandidateDiagnostic]
    semantic_ms: float
    lexical_ms: float
    rerank_ms: float


@dataclass(slots=True)
class _CorpusChunk:
    chunk_id: str
    text: str
    metadata: dict[str, object]
    tokens: list[str]


@dataclass(slots=True)
class _RankedChunk:
    chunk: _CorpusChunk
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    combined_score: float = 0.0


class ChromaStore:
    LEXICAL_STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
        "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
        "what", "when", "where", "which", "who", "why", "with",
    }

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        storage = Path(settings.chroma_dir).expanduser()
        storage.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(storage.resolve()))
        self.collection = self._get_collection()
        self._corpus_cache: list[_CorpusChunk] | None = None

    def _get_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection("document_chunks")
        except (NotFoundError, ValueError):
            pass
        self.collection = self._get_collection()
        self._corpus_cache = None

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Every chunk must have exactly one embedding.")
        metadatas = [
                {
                    "source_path": chunk.source_path,
                    "document_name": chunk.document_name,
                    "sequence": chunk.sequence,
                    "char_count": chunk.char_count,
                    "approx_page": chunk.approx_page,
                    "file_hash": chunk.file_hash,
                    "indexed_at": chunk.indexed_at,
                }
                for chunk in chunks
            ]
        max_batch = max(1, int(self.client.get_max_batch_size()))
        for start, stop in self._batches(len(chunks), max_batch):
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks[start:stop]],
                documents=[chunk.text for chunk in chunks[start:stop]],
                embeddings=embeddings[start:stop],
                metadatas=metadatas[start:stop],
            )
        # Delete stale trailing chunks only after every replacement batch succeeds.
        # A failed write therefore leaves the previous index queryable.
        ids_by_source: dict[str, set[str]] = {}
        for chunk in chunks:
            ids_by_source.setdefault(chunk.source_path, set()).add(chunk.chunk_id)
        for source_path, current_ids in ids_by_source.items():
            existing = self.collection.get(where={"source_path": source_path}, include=[])
            stale_ids = [item for item in existing.get("ids", []) if item not in current_ids]
            if stale_ids:
                self.collection.delete(ids=stale_ids)
        self._corpus_cache = None

    def hybrid_query(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int,
        source_paths: list[str] | None = None,
        pinned_source_paths: list[str] | None = None,
        excluded_source_paths: list[str] | None = None,
        min_score: float | None = None,
        candidate_multiplier: int = 4,
    ) -> HybridQueryResult:
        available = self.collection.count()
        normalized_paths = sorted({path for path in (source_paths or []) if path})
        pinned_paths = {path for path in (pinned_source_paths or []) if path}
        excluded_paths = {path for path in (excluded_source_paths or []) if path}
        if available == 0 or (source_paths is not None and not normalized_paths):
            return HybridQueryResult([], [], 0.0, 0.0, 0.0)

        candidate_limit = min(available, max(top_k, top_k * candidate_multiplier))
        conditions: list[dict[str, object]] = []
        if normalized_paths:
            conditions.append({"source_path": {"$in": normalized_paths}})
        if excluded_paths:
            conditions.append({"source_path": {"$nin": sorted(excluded_paths)}})
        where = conditions[0] if len(conditions) == 1 else {"$and": conditions} if conditions else None
        semantic_started = time.perf_counter()
        query_kwargs: dict[str, object] = {
            "query_embeddings": [query_embedding],
            "n_results": candidate_limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where
        response = self.collection.query(**query_kwargs)
        semantic_ms = (time.perf_counter() - semantic_started) * 1000

        ranked: dict[str, _RankedChunk] = {}
        ids = response.get("ids", [[]])[0]
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        for chunk_id, text, meta, distance in zip(ids, docs, metas, distances):
            if text is None or meta is None or distance is None:
                continue
            chunk = _CorpusChunk(str(chunk_id), str(text), dict(meta), self._tokenize(str(text)))
            ranked[chunk.chunk_id] = _RankedChunk(
                chunk=chunk,
                semantic_score=1.0 - float(distance),
            )

        lexical_started = time.perf_counter()
        corpus = self._corpus()
        if normalized_paths:
            allowed = set(normalized_paths)
            corpus = [item for item in corpus if str(item.metadata.get("source_path", "")) in allowed]
        if excluded_paths:
            corpus = [
                item for item in corpus
                if str(item.metadata.get("source_path", "")) not in excluded_paths
            ]
        corpus_by_id = {item.chunk_id: item for item in corpus}
        lexical_scores = self._bm25_scores(query_text, corpus)
        for chunk_id, score in sorted(
            lexical_scores.items(), key=lambda item: item[1], reverse=True
        )[:candidate_limit]:
            item = corpus_by_id.get(chunk_id)
            if item is None:
                continue
            candidate = ranked.setdefault(chunk_id, _RankedChunk(chunk=item))
            candidate.lexical_score = score
        lexical_ms = (time.perf_counter() - lexical_started) * 1000

        rerank_started = time.perf_counter()
        candidates = list(ranked.values())
        for candidate in candidates:
            semantic = max(0.0, candidate.semantic_score)
            both_bonus = 0.05 if semantic > 0 and candidate.lexical_score > 0 else 0.0
            candidate.combined_score = min(
                1.0, (0.45 * semantic) + (0.55 * candidate.lexical_score) + both_bonus
            )
        candidates.sort(key=lambda item: item.combined_score, reverse=True)
        eligible = [
            item for item in candidates
            if min_score is None
            or item.semantic_score >= min_score
            or item.lexical_score >= 0.25
        ]
        eligible_ids = {item.chunk.chunk_id for item in eligible}
        selected = self._diversified_selection(eligible, top_k, pinned_paths)
        selected_ids = {item.chunk.chunk_id for item in selected}
        rerank_ms = (time.perf_counter() - rerank_started) * 1000

        sources = [
            SourceRecord(
                source_id=f"S{index}",
                document_name=str(item.chunk.metadata.get("document_name", "Unknown")),
                source_path=str(item.chunk.metadata.get("source_path", "")),
                text=item.chunk.text,
                score=item.combined_score,
                approx_page=int(item.chunk.metadata.get("approx_page", 1)),
                semantic_score=item.semantic_score,
                lexical_score=item.lexical_score,
                matched_passage=self._supporting_passage(item.chunk.text, query_text),
            )
            for index, item in enumerate(selected, start=1)
        ]
        diagnostics = [
            CandidateDiagnostic(
                rank=index,
                selected=item.chunk.chunk_id in selected_ids,
                document_name=str(item.chunk.metadata.get("document_name", "Unknown")),
                source_path=str(item.chunk.metadata.get("source_path", "")),
                approx_page=int(item.chunk.metadata.get("approx_page", 1)),
                semantic_score=item.semantic_score,
                lexical_score=item.lexical_score,
                combined_score=item.combined_score,
                excerpt=item.chunk.text[:240].replace("\n", " "),
                decision=(
                    "Selected"
                    if item.chunk.chunk_id in selected_ids
                    else "Not in diversified top-k"
                    if item.chunk.chunk_id in eligible_ids
                    else "Below relevance threshold"
                ),
            )
            for index, item in enumerate(candidates, start=1)
        ]
        return HybridQueryResult(sources, diagnostics, semantic_ms, lexical_ms, rerank_ms)

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        source_paths: list[str] | None = None,
        min_score: float | None = None,
    ) -> list[SourceRecord]:
        available = self.collection.count()
        if available == 0:
            return []
        normalized_paths = sorted({path for path in (source_paths or []) if path})
        if source_paths is not None and not normalized_paths:
            return []
        where = {"source_path": {"$in": normalized_paths}} if normalized_paths else None
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(1, top_k), available),
            include=["documents", "metadatas", "distances"],
            where=where,
        )
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        sources: list[SourceRecord] = []
        for index, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
            if doc is None or meta is None or distance is None:
                continue
            score = 1.0 - float(distance)
            if min_score is not None and score < min_score:
                continue
            sources.append(
                SourceRecord(
                    source_id="",
                    document_name=str(meta.get("document_name", "Unknown")),
                    source_path=str(meta.get("source_path", "")),
                    text=doc,
                    score=score,
                    approx_page=int(meta.get("approx_page", 1)),
                )
            )
        for index, source in enumerate(sources, start=1):
            source.source_id = f"S{index}"
        return sources

    def delete_document(self, source_path: str) -> None:
        self.collection.delete(where={"source_path": source_path})
        self._corpus_cache = None

    def _corpus(self) -> list[_CorpusChunk]:
        if self._corpus_cache is None:
            payload = self.collection.get(include=["documents", "metadatas"])
            self._corpus_cache = [
                _CorpusChunk(str(chunk_id), str(text), dict(meta), self._tokenize(str(text)))
                for chunk_id, text, meta in zip(
                    payload.get("ids", []),
                    payload.get("documents", []),
                    payload.get("metadatas", []),
                )
                if text is not None and meta is not None
            ]
        return self._corpus_cache

    @classmethod
    def _bm25_scores(cls, query: str, corpus: list[_CorpusChunk]) -> dict[str, float]:
        query_terms = cls._tokenize(query)
        if not query_terms or not corpus:
            return {}
        document_frequency = Counter(
            token for item in corpus for token in set(item.tokens)
        )
        average_length = sum(len(item.tokens) for item in corpus) / len(corpus) or 1.0
        raw_scores: dict[str, float] = {}
        query_counts = Counter(query_terms)
        for item in corpus:
            counts = Counter(item.tokens)
            score = 0.0
            for term, query_frequency in query_counts.items():
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                frequency_in_docs = document_frequency[term]
                inverse_frequency = math.log(
                    1 + ((len(corpus) - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5))
                )
                denominator = frequency + 1.5 * (
                    1 - 0.75 + 0.75 * (len(item.tokens) / average_length)
                )
                score += query_frequency * inverse_frequency * ((frequency * 2.5) / denominator)
            if score > 0:
                raw_scores[item.chunk_id] = score
        maximum = max(raw_scores.values(), default=0.0)
        return {
            chunk_id: score / maximum for chunk_id, score in raw_scores.items()
        } if maximum else {}

    @staticmethod
    def _diversified_selection(
        candidates: list[_RankedChunk], top_k: int, pinned_paths: set[str] | None = None
    ) -> list[_RankedChunk]:
        remaining = candidates.copy()
        selected: list[_RankedChunk] = []
        document_counts: Counter[str] = Counter()
        for pinned_path in sorted(pinned_paths or set()):
            matches = [
                item for item in remaining
                if str(item.chunk.metadata.get("source_path", "")) == pinned_path
            ]
            if matches and len(selected) < top_k:
                best_pinned = max(matches, key=lambda item: item.combined_score)
                selected.append(best_pinned)
                remaining.remove(best_pinned)
                document_counts[pinned_path] += 1
        while remaining and len(selected) < top_k:
            def adjusted(item: _RankedChunk) -> float:
                source_path = str(item.chunk.metadata.get("source_path", ""))
                duplicate_penalty = 0.0
                item_terms = set(item.chunk.tokens)
                for chosen in selected:
                    chosen_terms = set(chosen.chunk.tokens)
                    union = item_terms | chosen_terms
                    if union:
                        duplicate_penalty = max(
                            duplicate_penalty,
                            0.15 * (len(item_terms & chosen_terms) / len(union)),
                        )
                return item.combined_score - (0.07 * document_counts[source_path]) - duplicate_penalty

            best = max(remaining, key=adjusted)
            remaining.remove(best)
            selected.append(best)
            document_counts[str(best.chunk.metadata.get("source_path", ""))] += 1
        return selected

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        return [
            token for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in cls.LEXICAL_STOP_WORDS
        ]

    @classmethod
    def _supporting_passage(cls, text: str, query: str) -> str:
        query_terms = set(cls._tokenize(query))
        passages = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        if not passages:
            return text[:300]
        if not query_terms:
            return passages[0]
        return max(
            passages,
            key=lambda passage: (
                len(query_terms & set(cls._tokenize(passage))),
                -len(passage),
            ),
        )

    def indexed_documents(self) -> list[dict[str, object]]:
        payload = self.collection.get(include=["metadatas"])
        documents: dict[str, dict[str, object]] = {}
        for meta in payload.get("metadatas", []):
            source_path = str(meta["source_path"])
            item = documents.setdefault(
                source_path,
                {
                    "document_name": meta["document_name"],
                    "source_path": source_path,
                    "chunks": 0,
                    "file_hash": meta.get("file_hash", ""),
                    "indexed_at": meta.get("indexed_at", ""),
                },
            )
            item["chunks"] = int(item["chunks"]) + 1
        return sorted(documents.values(), key=lambda item: str(item["document_name"]).lower())

    @staticmethod
    def _batches(length: int, size: int) -> Iterator[tuple[int, int]]:
        for start in range(0, length, size):
            yield start, min(start + size, length)

    def stats(self) -> dict[str, int]:
        docs = self.indexed_documents()
        return {"documents": len(docs), "chunks": self.collection.count()}
