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


if __name__ == "__main__":
    unittest.main()
