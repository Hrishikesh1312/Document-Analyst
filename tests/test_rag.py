from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from document_analyst.config import AppSettings
from document_analyst.models import SourceRecord
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
        self.service.store.query.return_value = []

        self.service.retrieve_sources("question", ["/docs/a.txt"])

        self.service.store.query.assert_called_once_with(
            [1.0, 0.0], 3, source_paths=["/docs/a.txt"], min_score=0.4
        )

    def test_low_evidence_response_does_not_load_llm(self) -> None:
        self.service.store.query.return_value = []

        result = self.service.answer_question_stream("unknown", [], None)

        self.assertIn("sufficiently relevant", "".join(result.chunks))
        self.assertEqual(result.sources, [])
        self.service.models.local_llm_model_path.assert_not_called()

    def test_streamed_llm_chunks_are_forwarded(self) -> None:
        source = SourceRecord("S1", "a.txt", "/a.txt", "facts", 0.9, 1)
        self.service.store.query.return_value = [source]
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
        self.assertTrue(llm.create_chat_completion.call_args.kwargs["stream"])


if __name__ == "__main__":
    unittest.main()
