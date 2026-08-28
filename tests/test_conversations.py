from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader

from document_analyst.models import SourceRecord
from document_analyst.services.conversation_export import conversation_markdown, conversation_pdf
from document_analyst.services.conversation_store import ConversationStore


class ConversationTests(unittest.TestCase):
    def test_named_conversations_persist_rename_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ConversationStore(Path(directory) / "conversations.json")
            created = store.create("Research")
            store.save_conversation(
                created["id"],
                "Research",
                [{"role": "user", "content": "Question"}],
                {},
            )
            store.rename(created["id"], "Renamed")

            reloaded = ConversationStore(store.path).get(created["id"])
            self.assertEqual(reloaded["name"], "Renamed")
            self.assertEqual(reloaded["messages"][0]["content"], "Question")

            store.delete(created["id"])
            self.assertIsNone(store.get(created["id"]))

    def test_markdown_and_pdf_exports_include_citations(self) -> None:
        messages = [
            {"role": "user", "content": "What is backpropagation?"},
            {"role": "assistant", "content": "It computes gradients [S1]."},
        ]
        sources = {
            1: [SourceRecord("S1", "book.pdf", "/docs/book.pdf", "Evidence", 0.9, 42)]
        }

        markdown = conversation_markdown("ML Notes", messages, sources)
        pdf = conversation_pdf("ML Notes", messages, sources)

        self.assertIn("It computes gradients [S1].", markdown)
        self.assertIn("book.pdf, page 42", markdown)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as output:
            output.write(pdf)
            output.flush()
            reader = PdfReader(output.name)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        self.assertIn("ML Notes", text)
        self.assertIn("It computes gradients [S1].", text)
        self.assertIn("book.pdf, page 42", text)


if __name__ == "__main__":
    unittest.main()
