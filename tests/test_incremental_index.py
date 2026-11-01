from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pptx import Presentation

from document_analyst.config import AppSettings
from document_analyst.services.index_manifest import IndexManifest
from document_analyst.services.rag import RagService


class IncrementalIndexTests(unittest.TestCase):
    def _service(self, directory: str) -> RagService:
        root = Path(directory)
        service = RagService(AppSettings(chroma_dir=str(root / "chroma")))
        service.manifest = IndexManifest(root / "manifest.json")
        service._embed_texts = lambda texts: [
            [1.0, float(index + 1)] for index, _text in enumerate(texts)
        ]
        return service

    def test_unchanged_changed_duplicate_and_removed_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents"
            documents.mkdir()
            first = documents / "first.txt"
            first.write_text("Initial indexed content.", encoding="utf-8")
            service = self._service(directory)

            initial = service.index_documents(str(documents))
            unchanged = service.index_documents(str(documents))
            first.write_text("Changed indexed content.", encoding="utf-8")
            changed = service.index_documents(str(documents))
            duplicate = documents / "duplicate.txt"
            duplicate.write_text("Changed indexed content.", encoding="utf-8")
            duplicated = service.index_documents(str(documents))
            first.unlink()
            removed = service.index_documents(str(documents))

            self.assertEqual(initial.document_count, 1)
            self.assertEqual(unchanged.unchanged_count, 1)
            self.assertEqual(changed.document_count, 1)
            self.assertEqual(duplicated.duplicate_count, 1)
            self.assertEqual(removed.removed_count, 1)
            statuses = {item["document_name"]: item for item in service.document_statuses()}
            self.assertEqual(statuses["first.txt"]["status"], "removed")
            self.assertTrue(statuses["duplicate.txt"]["indexed_at"])

    def test_structured_progress_tracks_pdf_pages_and_embedding_batches(self) -> None:
        import pymupdf

        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents"
            documents.mkdir()
            with pymupdf.open() as pdf:
                for index in range(3):
                    pdf.new_page().insert_text((72, 72), f"Page {index}. Important content.")
                pdf.save(documents / "book.pdf")
            service = self._service(directory)
            events = []
            result = service.index_documents(str(documents), detail_progress=events.append)
            self.assertEqual(result.document_count, 1)
            page_events = [e for e in events if e.unit == "pages"]
            self.assertEqual(page_events[-1].completed_units, 3)
            self.assertEqual(page_events[-1].total_units, 3)
            batch_events = [e for e in events if e.unit == "passages"]
            self.assertTrue(any(e.phase == "chunking" for e in batch_events))
            self.assertTrue(any(e.phase == "embedding" for e in batch_events))
            self.assertTrue(all(e.document_name == "book.pdf" for e in events))
            self.assertEqual(events[-1].phase, "writing")
            self.assertEqual(events[-1].overall_fraction, 1)
            events.clear()
            service.index_documents(str(documents), detail_progress=events.append)
            self.assertEqual(events[-1].phase, "skipped")
            self.assertEqual(events[-1].overall_fraction, 1)

    def test_failed_file_can_be_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents"
            documents.mkdir()
            deck = documents / "deck.pptx"
            deck.write_bytes(b"not a presentation")
            service = self._service(directory)

            failed = service.index_documents(str(documents))
            presentation = Presentation()
            slide = presentation.slides.add_slide(presentation.slide_layouts[1])
            slide.shapes.title.text = "Recovered presentation"
            presentation.save(deck)
            retried = service.index_documents(str(documents), retry_failed_only=True)

            self.assertEqual(failed.failed_count, 1)
            self.assertEqual(retried.document_count, 1)
            self.assertEqual(service.document_statuses()[0]["status"], "indexed")

    def test_fast_mode_only_embeds_final_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents"
            documents.mkdir()
            (documents / "book.txt").write_text(
                " ".join(f"Sentence {i} explains indexing." for i in range(100)), encoding="utf-8"
            )
            service = self._service(directory)
            service.settings.chunking_mode = "fast"
            batches = []
            def embed(texts):
                batches.extend(texts)
                return [[1.0, 0.0] for _ in texts]
            service._embed_texts = embed
            result = service.index_documents(str(documents))
            self.assertEqual(result.document_count, 1)
            self.assertEqual(len(batches), result.chunk_count)
            self.assertGreater(result.chunk_count, 0)
            self.assertLess(result.chunk_count, 100)

    def test_large_document_reports_batched_chunking_before_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents"
            documents.mkdir()
            (documents / "large.txt").write_text(
                " ".join(f"Sentence {i} explains indexing." for i in range(100)), encoding="utf-8"
            )
            service = self._service(directory)
            batch_sizes = []
            def embed(texts):
                batch_sizes.append(len(texts))
                return [[1.0, 0.0] for _ in texts]
            service._embed_texts = embed
            events = []
            result = service.index_documents(str(documents), progress=lambda *event: events.append(event))
            self.assertEqual(result.document_count, 1)
            self.assertLessEqual(max(batch_sizes), 32)
            self.assertTrue(any(e[0] == "chunking" and "32/100" in e[1] for e in events))
            self.assertTrue(any(e[0] == "chunking" and "100/100" in e[1] for e in events))

    def test_configured_batch_size_is_used_for_sentence_and_chunk_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents"
            documents.mkdir()
            (documents / "book.txt").write_text(
                " ".join(f"Sentence {i} explains local indexing." for i in range(100)), encoding="utf-8"
            )
            service = self._service(directory)
            service.settings.embedding_batch_size = 16
            batches = []
            service._embed_texts = lambda texts: batches.append(len(texts)) or [[1.0, 0.0] for _ in texts]
            result = service.index_documents(str(documents))
            self.assertEqual(result.document_count, 1)
            self.assertEqual(batches[:7], [16, 16, 16, 16, 16, 16, 4])
            self.assertTrue(all(size <= 16 for size in batches))
            self.assertEqual(sum(batches), 100 + result.chunk_count)

    def test_cancel_during_chunking_preserves_index_and_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents"
            documents.mkdir()
            (documents / "large.txt").write_text("A sentence. " * 100, encoding="utf-8")
            service = self._service(directory)
            cancelled = False
            def embed(texts):
                nonlocal cancelled
                cancelled = True
                return [[1.0, 0.0] for _ in texts]
            service._embed_texts = embed
            result = service.index_documents(str(documents), should_cancel=lambda: cancelled)
            self.assertTrue(result.cancelled)
            self.assertEqual(result.failed_count, 0)
            self.assertEqual(service.stats()["chunks"], 0)
            self.assertEqual(service.document_statuses()[0]["status"], "cancelled")

    def test_cancellation_stops_before_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            documents = Path(directory) / "documents"
            documents.mkdir()
            (documents / "file.txt").write_text("content", encoding="utf-8")
            service = self._service(directory)

            result = service.index_documents(str(documents), should_cancel=lambda: True)

            self.assertTrue(result.cancelled)
            self.assertEqual(result.document_count, 0)
            self.assertEqual(service.stats()["chunks"], 0)


if __name__ == "__main__":
    unittest.main()
