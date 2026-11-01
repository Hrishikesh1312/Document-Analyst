from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from document_analyst.config import AppSettings
from document_analyst.services.indexing_performance import apply_indexing_preset, benchmark_embedding_batches


class IndexingPerformanceTests(unittest.TestCase):
    def test_presets_preserve_model_retrieval_and_document_settings(self):
        original = AppSettings(documents_dir="/docs", top_k=7, embeddings_repo="owner/search")
        for preset, mode, size in (("Fast", "fast", 1000), ("Balanced", "fast", 700), ("Quality", "semantic", 500)):
            with self.subTest(preset=preset):
                settings = apply_indexing_preset(original, preset, 64)
                self.assertEqual((settings.chunking_mode, settings.chunk_size), (mode, size))
                self.assertEqual(settings.embedding_batch_size, 64)
                self.assertEqual(settings.embeddings_repo, original.embeddings_repo)
                self.assertEqual(settings.documents_dir, original.documents_dir)
                self.assertEqual(settings.top_k, original.top_k)
        self.assertEqual(original.embedding_batch_size, 32)
        self.assertEqual(apply_indexing_preset(original, "Keep current", 16).chunking_mode, "semantic")

    def test_performance_settings_persist_and_reload(self):
        import tempfile
        from pathlib import Path
        import document_analyst.config as config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(config, APP_DATA_DIR=root, MODELS_DIR=root / "models",
                                CHROMA_DIR=root / "chroma", SETTINGS_PATH=root / "settings.json"):
                settings = apply_indexing_preset(AppSettings(), "Balanced", 64)
                config.save_settings(settings)
                restored = config.load_settings()
                self.assertEqual(restored.embedding_batch_size, 64)
                self.assertEqual(restored.chunking_mode, "fast")
                self.assertEqual(restored.chunk_size, 700)

    def test_batch_size_validation(self):
        self.assertEqual(AppSettings(embedding_batch_size=0).embedding_batch_size, 1)
        self.assertEqual(AppSettings(embedding_batch_size=9999).embedding_batch_size, 256)
        self.assertEqual(AppSettings().embedding_batch_size, 32)

    def test_benchmark_warms_up_compares_same_work_and_reports_rates(self):
        model = Mock()
        events = []
        with patch("document_analyst.services.indexing_performance.time.perf_counter", side_effect=[10, 12, 20, 21]):
            results = benchmark_embedding_batches(model, lambda *event: events.append(event), (16, 32))
        self.assertEqual([r.passages_per_second for r in results], [128, 256])
        self.assertEqual(events, [(1, 2, 16), (2, 2, 32)])
        calls = model.encode.call_args_list
        self.assertEqual(len(calls[0].args[0]), 8)
        self.assertEqual(calls[1].args[0], calls[3].args[0])
        self.assertTrue(all(c.kwargs["convert_to_numpy"] for c in calls))
        self.assertTrue(all(c.kwargs["normalize_embeddings"] for c in calls))

    def test_benchmark_stops_after_memory_error_and_does_not_hide_other_errors(self):
        model = Mock()
        model.encode.side_effect = [None, None, None, RuntimeError("MPS backend out of memory")]
        with patch("document_analyst.services.indexing_performance.time.perf_counter", side_effect=[0, 1, 2]):
            results = benchmark_embedding_batches(model, candidates=(16, 32, 64))
        self.assertEqual(results[-1].batch_size, 32)
        self.assertEqual(results[-1].error, "Insufficient device memory")
        self.assertEqual(len(results), 2)
        model.encode.side_effect = [None, RuntimeError("Device disconnected")]
        with self.assertRaisesRegex(RuntimeError, "Device disconnected"):
            benchmark_embedding_batches(model)
