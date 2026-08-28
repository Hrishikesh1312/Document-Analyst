from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from document_analyst.config import AppSettings
from document_analyst.models import SourceRecord
from document_analyst.services.chroma_store import HybridQueryResult
from document_analyst.services.rag import RagService


class RagServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RagService.__new__(RagService)
        self.service.settings = AppSettings(retrieval_min_score=0.4, top_k=3)
        self.service.ensure_embedder = MagicMock()
        self.service._embed_texts = MagicMock(return_value=[[1.0, 0.0]])
        self.service.store = MagicMock()
        self.service.models = MagicMock()

    def test_retrieval_forwards_scope_and_threshold(self) -> None:
        self.service.store.hybrid_query.return_value = HybridQueryResult([], [], 1.0, 2.0, 3.0)

        sources, diagnostics = self.service.retrieve_with_diagnostics(
            "question", ["/docs/a.txt"]
        )

        self.service.store.hybrid_query.assert_called_once_with(
            "question", [1.0, 0.0], 3,
            source_paths=["/docs/a.txt"], pinned_source_paths=None,
            excluded_source_paths=None, min_score=0.4
        )
        self.assertEqual(sources, [])
        self.assertEqual(diagnostics.scope, ["/docs/a.txt"])
        self.assertEqual(diagnostics.semantic_ms, 1.0)
        self.assertEqual(diagnostics.lexical_ms, 2.0)
        self.assertEqual(diagnostics.rerank_ms, 3.0)

    def test_low_evidence_response_does_not_load_llm(self) -> None:
        self.service.store.hybrid_query.return_value = HybridQueryResult([], [], 0.0, 0.0, 0.0)

        result = self.service.answer_question_stream("unknown", [], None)

        self.assertIn("sufficiently relevant", "".join(result.chunks))
        self.assertEqual(result.sources, [])
        self.assertIsNotNone(result.diagnostics)
        self.service.models.local_llm_model_path.assert_not_called()

    def test_streamed_llm_chunks_are_forwarded(self) -> None:
        source = SourceRecord("S1", "a.txt", "/a.txt", "facts", 0.9, 1)
        self.service.store.hybrid_query.return_value = HybridQueryResult(
            [source], [], 1.0, 2.0, 3.0
        )
        self.service.models.local_llm_model_path.return_value = "/model.gguf"
        llm = MagicMock()
        llm.tokenize.return_value = [1]
        llm.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ])
        self.service.ensure_llm = MagicMock(return_value=llm)

        result = self.service.answer_question_stream("question", [])

        self.assertEqual("".join(result.chunks), "Hello world")
        self.assertEqual(result.sources, [source])
        self.assertEqual(result.diagnostics.selected_count, 1)
        self.assertTrue(llm.create_chat_completion.call_args.kwargs["stream"])


if __name__ == "__main__":
    unittest.main()
