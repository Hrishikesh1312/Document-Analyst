from __future__ import annotations

import unittest

from document_analyst.services.model_manager import _progress_tqdm


class DownloadProgressTests(unittest.TestCase):
    def test_hugging_face_progress_is_forwarded(self) -> None:
        events: list[tuple[str, str, int, int | None]] = []
        progress_class = _progress_tqdm(
            lambda phase, description, current, total: events.append(
                (phase, description, current, total)
            ),
            "llm",
        )
        bar = progress_class(total=100, desc="model.gguf", disable=False)
        bar.update(25)
        bar.close()
        self.assertTrue(events)
        self.assertEqual(events[-1][0], "llm")
        self.assertEqual(events[-1][1], "model.gguf")
        self.assertEqual(events[-1][2], 25)
        self.assertEqual(events[-1][3], 100)


if __name__ == "__main__":
    unittest.main()
