from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from document_analyst.config import AppSettings
from document_analyst.models import ChunkRecord, DocumentRecord, SourceRecord


class ChromaStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(Path(settings.chroma_dir)))
        self.collection = self._get_collection()

    def _get_collection(self) -> Collection:
        return self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection("document_chunks")
        except Exception:
            pass
        self.collection = self._get_collection()

    def upsert_chunks(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "source_path": chunk.source_path,
                    "document_name": chunk.document_name,
                    "sequence": chunk.sequence,
                    "char_count": chunk.char_count,
                    "approx_page": chunk.approx_page,
                }
                for chunk in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[SourceRecord]:
        response = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = response.get("documents", [[]])[0]
        metas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        sources: list[SourceRecord] = []
        for index, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
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

    def stats(self) -> dict[str, int]:
        docs = self.indexed_documents()
        return {"documents": len(docs), "chunks": self.collection.count()}
