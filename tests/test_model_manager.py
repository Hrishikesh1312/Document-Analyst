from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_analyst.config import AppSettings
from document_analyst.services.model_manager import ModelManager, discover_model_repositories


class ModelManagerTests(unittest.TestCase):
    def test_embedding_download_requires_config_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "README.md").write_text("Partial download")
            self.assertFalse(ModelManager._embedding_download_complete(directory))
            (directory / "config.json").write_text("{}")
            self.assertFalse(ModelManager._embedding_download_complete(directory))
            (directory / "model.safetensors").touch()
            self.assertFalse(ModelManager._embedding_download_complete(directory))
            (directory / "model.safetensors").write_bytes(b"weights")
            self.assertTrue(ModelManager._embedding_download_complete(directory))

    @patch("document_analyst.services.model_manager.snapshot_download")
    def test_partial_embedding_download_is_resumed_and_checked(self, download) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manager = ModelManager(AppSettings())
            with patch("document_analyst.services.model_manager.Path", return_value=directory):
                (directory / "config.json").write_text("{}")
                with self.assertRaisesRegex(FileNotFoundError, "download is incomplete"):
                    manager.ensure_embedding_model()
                download.assert_called_once()
                download.side_effect = lambda **kwargs: (directory / "model.safetensors").write_bytes(b"weights")
                self.assertEqual(manager.ensure_embedding_model(), directory)
                self.assertTrue(manager.local_embedding_model_exists())
                manager.ensure_embedding_model()
                self.assertEqual(download.call_count, 2)

    def test_sharded_embedding_requires_every_weight_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_text("{}")
            (directory / "model.safetensors.index.json").write_text(
                '{"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}}'
            )
            (directory / "part-1.safetensors").write_bytes(b"weights")
            self.assertFalse(ModelManager._embedding_download_complete(directory))
            (directory / "part-2.safetensors").write_bytes(b"weights")
            self.assertTrue(ModelManager._embedding_download_complete(directory))

    @patch("document_analyst.services.model_manager.list_repo_files")
    @patch("document_analyst.services.model_manager.list_models")
    def test_discovery_only_returns_valid_small_gguf_repositories(
        self, list_models_mock, list_files_mock
    ) -> None:
        class Model:
            def __init__(self, model_id: str) -> None:
                self.id = model_id

        list_models_mock.side_effect = [
            [Model("sentence-transformers/example")],
            [Model("owner/model-7B-Instruct-GGUF"), Model("owner/model-1.5B-Instruct-GGUF")],
        ]
        list_files_mock.return_value = ["README.md", "model-Q4_K_M.gguf"]

        embeddings, llms = discover_model_repositories(limit=5)

        self.assertEqual(embeddings, ["sentence-transformers/example"])
        self.assertEqual(llms, ["owner/model-1.5B-Instruct-GGUF"])
        list_files_mock.assert_called_once_with("owner/model-1.5B-Instruct-GGUF", token=False)

    def test_delete_downloaded_models_removes_all_managed_repositories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            models_dir = Path(temporary) / "models"
            (models_dir / "embeddings" / "current").mkdir(parents=True)
            (models_dir / "embeddings" / "current" / "model.bin").write_bytes(b"model")
            (models_dir / "llm" / "previous" / "nested").mkdir(parents=True)
            (models_dir / "llm" / "previous" / "nested" / "model.gguf").write_bytes(b"model")

            manager = ModelManager(AppSettings())
            with patch("document_analyst.services.model_manager.MODELS_DIR", models_dir):
                manager.delete_downloaded_models()

            self.assertTrue(models_dir.is_dir())
            self.assertEqual(list(models_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
