from __future__ import annotations

import unittest
from textwrap import dedent

from streamlit.testing.v1 import AppTest

from document_analyst.app import _citation_links, _highlight_text


class RetrievalDiagnosticsUiTests(unittest.TestCase):
    def test_source_highlighting_and_citation_navigation(self) -> None:
        highlighted = _highlight_text("Gradient descent updates weights.", "updates weights")
        linked = _citation_links("The claim is supported [S2].", 7)

        self.assertIn("<mark>updates weights</mark>", highlighted)
        self.assertIn("(#turn-7-s2)", linked)

    def test_diagnostics_panel_renders_scores_scope_and_timings(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                from document_analyst.app import _diagnostics_panel
                from document_analyst.models import CandidateDiagnostic, RetrievalDiagnostics

                _diagnostics_panel(RetrievalDiagnostics(
                    query="backpropagation",
                    scope=["/docs/book.pdf"],
                    candidate_count=2,
                    selected_count=1,
                    documents_covered=1,
                    embedding_ms=1.0,
                    semantic_ms=2.0,
                    lexical_ms=3.0,
                    rerank_ms=4.0,
                    total_ms=10.0,
                    candidates=[CandidateDiagnostic(
                        rank=1,
                        selected=True,
                        document_name="book.pdf",
                        source_path="/docs/book.pdf",
                        approx_page=42,
                        semantic_score=0.8,
                        lexical_score=1.0,
                        combined_score=0.95,
                        excerpt="Backpropagation computes gradients.",
                    )],
                ))
                """
            ),
            default_timeout=15,
        ).run()

        self.assertFalse(app.exception)
        self.assertEqual([metric.value for metric in app.metric], ["2", "1", "1"])
        self.assertEqual(len(app.dataframe), 1)
        markdown = "\n".join(item.value for item in app.markdown)
        self.assertIn("BM25", markdown)
        self.assertIn("total", markdown)

    def test_legacy_ppt_dialog_lists_files_and_conversion_guidance(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                from pathlib import Path
                from document_analyst.app import _legacy_ppt_dialog

                _legacy_ppt_dialog([Path("/documents/legacy-slides.ppt")])
                """
            ),
            default_timeout=15,
        ).run()

        self.assertFalse(app.exception)
        self.assertIn("legacy `.ppt`", app.warning[0].value)
        self.assertIn("`.pptx`", app.warning[0].value)
        self.assertEqual(app.code[0].value, "/documents/legacy-slides.ppt")


if __name__ == "__main__":
    unittest.main()
