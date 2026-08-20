from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from document_analyst.config import AppSettings
from document_analyst.models import ChunkRecord, DocumentRecord, SourceRecord


class ChromaStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        storage = Path(settings.chroma_dir).expanduser()
        storage.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(storage.resolve()))
        self.collection = self._get_collection()

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

    def query(self, query_embedding: list[float], top_k: int) -> list[SourceRecord]:
        available = self.collection.count()
        if available == 0:
            return []
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(max(1, top_k), available),
            include=["documents", "metadatas", "distances"],
        )
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        sources: list[SourceRecord] = []
        for index, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
            if doc is None or meta is None or distance is None:
                continue
            score = 1.0 - float(distance)
            sources.append(
                SourceRecord(
                    source_id=f"S{index}",
                    document_name=str(meta.get("document_name", "Unknown")),
                    source_path=str(meta.get("source_path", "")),
                    text=doc,
                    score=score,
                    approx_page=int(meta.get("approx_page", 1)),
                )
            )
        return sources

    def delete_document(self, source_path: str) -> None:
        self.collection.delete(where={"source_path": source_path})

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
