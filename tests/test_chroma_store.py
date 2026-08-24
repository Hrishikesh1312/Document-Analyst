from __future__ import annotations

import tempfile
import unittest

from document_analyst.config import AppSettings
from document_analyst.models import ChunkRecord
from document_analyst.services.chroma_store import ChromaStore


class ChromaStoreTests(unittest.TestCase):
    def test_query_filters_documents_thresholds_scores_and_renumbers_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaStore(AppSettings(chroma_dir=directory))
            chunks = [
                ChunkRecord("a:0", "/docs/a.txt", "a.txt", "alpha facts", 0, 11, 1),
                ChunkRecord("b:0", "/docs/b.txt", "b.txt", "beta facts", 0, 10, 2),
            ]
            store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

            filtered = store.query([1.0, 0.0], 4, source_paths=["/docs/a.txt"], min_score=0.5)
            rejected = store.query([1.0, 0.0], 4, source_paths=["/docs/b.txt"], min_score=0.5)

            self.assertEqual([(item.source_id, item.document_name) for item in filtered], [("S1", "a.txt")])
            self.assertEqual(rejected, [])

    def test_explicit_empty_filter_returns_no_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaStore(AppSettings(chroma_dir=directory))
            store.upsert_chunks(
                [ChunkRecord("a:0", "/docs/a.txt", "a.txt", "alpha", 0, 5, 1)],
                [[1.0, 0.0]],
            )
            self.assertEqual(store.query([1.0, 0.0], 4, source_paths=[]), [])

    def test_hybrid_search_recovers_exact_terms_missed_by_semantic_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaStore(AppSettings(chroma_dir=directory))
            chunks = [
                ChunkRecord("a:0", "/docs/a.txt", "a.txt", "general learning overview", 0, 25, 1),
                ChunkRecord("b:0", "/docs/b.txt", "b.txt", "The optimizer is AdaGradZX.", 0, 27, 5),
            ]
            store.upsert_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

            result = store.hybrid_query(
                "AdaGradZX optimizer", [1.0, 0.0], 2, min_score=0.9
            )

            self.assertEqual(result.sources[0].document_name, "b.txt")
            self.assertGreaterEqual(result.sources[0].lexical_score, 0.25)
            self.assertTrue(any(item.selected for item in result.candidates))

    def test_hybrid_reranking_diversifies_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaStore(AppSettings(chroma_dir=directory))
            chunks = [
                ChunkRecord("a:0", "/docs/a.txt", "a.txt", "neural training loss gradient", 0, 29, 1),
                ChunkRecord("a:1", "/docs/a.txt", "a.txt", "neural training loss gradient details", 1, 37, 2),
                ChunkRecord("b:0", "/docs/b.txt", "b.txt", "neural training loss optimization", 0, 33, 4),
            ]
            store.upsert_chunks(chunks, [[1.0, 0.0], [0.99, 0.01], [0.97, 0.03]])

            result = store.hybrid_query("neural training loss", [1.0, 0.0], 2)

            self.assertEqual({source.document_name for source in result.sources}, {"a.txt", "b.txt"})

    def test_hybrid_search_respects_document_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaStore(AppSettings(chroma_dir=directory))
            store.upsert_chunks(
                [
                    ChunkRecord("a:0", "/docs/a.txt", "a.txt", "unique optimizer alpha", 0, 22, 1),
                    ChunkRecord("b:0", "/docs/b.txt", "b.txt", "unique optimizer beta", 0, 21, 1),
                ],
                [[1.0, 0.0], [0.9, 0.1]],
            )

            result = store.hybrid_query(
                "unique optimizer", [1.0, 0.0], 4, source_paths=["/docs/b.txt"]
            )

            self.assertEqual([source.document_name for source in result.sources], ["b.txt"])
            self.assertTrue(all(item.document_name == "b.txt" for item in result.candidates))

    def test_lexical_cache_is_invalidated_after_upsert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ChromaStore(AppSettings(chroma_dir=directory))
            store.upsert_chunks(
                [ChunkRecord("a:0", "/docs/a.txt", "a.txt", "alpha", 0, 5, 1)],
                [[1.0, 0.0]],
            )
            self.assertEqual(len(store._corpus()), 1)
            store.upsert_chunks(
                [ChunkRecord("b:0", "/docs/b.txt", "b.txt", "beta", 0, 4, 1)],
                [[0.0, 1.0]],
            )
            self.assertEqual(len(store._corpus()), 2)


if __name__ == "__main__":
    unittest.main()
