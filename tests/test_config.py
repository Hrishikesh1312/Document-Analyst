from __future__ import annotations

import unittest

from document_analyst.config import AppSettings, repo_storage_name


class SettingsTests(unittest.TestCase):
    def test_settings_clamp_untrusted_values(self) -> None:
        settings = AppSettings(chunk_size=20, chunk_overlap=999, top_k=0,
                               semantic_threshold=9, retrieval_min_score=9,
                               supported_extensions=["PDF", ".Txt", "PDF"])
        self.assertEqual(settings.chunk_size, 100)
        self.assertEqual(settings.chunk_overlap, 99)
        self.assertEqual(settings.top_k, 1)
        self.assertEqual(settings.semantic_threshold, 1.0)
        self.assertEqual(settings.retrieval_min_score, 1.0)
        self.assertEqual(settings.supported_extensions, [".pdf", ".txt"])

    def test_office_formats_are_supported_by_default(self) -> None:
        settings = AppSettings()
        self.assertIn(".docx", settings.supported_extensions)
        self.assertIn(".pptx", settings.supported_extensions)

    def test_repo_storage_name_is_path_safe(self) -> None:
        name = repo_storage_name("owner/model name")
        self.assertEqual(name, "owner-model-name")
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)


if __name__ == "__main__":
    unittest.main()
