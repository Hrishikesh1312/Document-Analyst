from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from textwrap import dedent

from streamlit.testing.v1 import AppTest

from document_analyst.app import _citation_links, _highlight_text
from document_analyst.services.conversation_store import ConversationStore
from document_analyst.ui.theme import THEME_CSS


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


class UiFoundationTests(unittest.TestCase):
    def test_theme_defines_shared_tokens_and_responsive_layout(self) -> None:
        self.assertIn("--da-accent", THEME_CSS)
        self.assertIn("--da-radius", THEME_CSS)
        self.assertIn("@media (max-width: 900px)", THEME_CSS)
        self.assertNotIn("hero-card", THEME_CSS)

    def test_shared_headers_and_empty_state_render(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                from document_analyst.ui.components import app_header, empty_state, page_heading
                from document_analyst.ui.theme import inject_theme

                inject_theme()
                app_header("Library", 12, 480, True)
                page_heading("Library")
                empty_state("Your library is empty", "Choose a folder to begin.", "▤")
                """
            ),
            default_timeout=15,
        ).run()

        self.assertFalse(app.exception)
        markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("Document library", markup)
        self.assertIn("12 documents · 480 searchable chunks", markup)
        self.assertIn("Local model ready", markup)
        self.assertIn("Your library is empty", markup)

    def test_session_state_migrates_legacy_navigation(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                import streamlit as st
                from document_analyst.ui.state import initialize_session_state

                st.session_state.active_view = "Manage Documents"
                initialize_session_state()
                st.write(st.session_state.active_view)
                """
            ),
            default_timeout=15,
        ).run()

        self.assertFalse(app.exception)
        self.assertEqual(app.markdown[-1].value, "Library")

    def test_deleting_active_conversation_selects_replacement_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "conversations.json"
            app = AppTest.from_string(
                dedent(
                    f"""
                    import streamlit as st
                    from pathlib import Path
                    from document_analyst.app import _conversation_panel, _load_conversation
                    from document_analyst.services.conversation_store import ConversationStore

                    store = ConversationStore(Path({str(store_path)!r}))
                    if not st.session_state.get("conversation_loaded"):
                        first = store.create("First")
                        store.create("Second")
                        _load_conversation(first)
                    _conversation_panel(store)
                    """
                ),
                default_timeout=15,
            ).run()

            self.assertFalse(app.exception)
            delete = next(button for button in app.button if button.label == "Delete")
            delete.click().run()

            self.assertFalse(app.exception)
            self.assertEqual(len(ConversationStore(store_path).list()), 1)
            self.assertEqual(
                app.session_state.active_conversation_id,
                app.session_state.conversation_selector,
            )


if __name__ == "__main__":
    unittest.main()
