from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from document_analyst.config import AppSettings
from document_analyst.services.model_manager import ModelManager, discover_model_repositories


class ModelManagerTests(unittest.TestCase):
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
