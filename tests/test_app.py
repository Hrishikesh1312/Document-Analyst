from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from textwrap import dedent

from streamlit.testing.v1 import AppTest

from document_analyst.app import _citation_links, _highlight_text
from document_analyst.services.conversation_store import ConversationStore
from document_analyst.ui.components import model_display_name
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
    def test_navigation_switches_on_every_click_and_survives_reruns(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                import streamlit as st
                from document_analyst.app import _sidebar
                from document_analyst.config import AppSettings
                from document_analyst.ui.state import initialize_session_state

                class Service:
                    def stats(self):
                        return {"documents": 0, "chunks": 0}

                initialize_session_state()
                if st.button("Go to Settings"):
                    st.session_state.active_view = "Settings"
                st.write(_sidebar(AppSettings(), Service(), None))
                """
            ),
            default_timeout=15,
        ).run()

        navigation_id = app.radio[0].id
        for view in ["Library", "Settings", "Help", "Ask", "Settings", "Library"]:
            app.radio[0].set_value(view).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.radio[0].value, view)
            self.assertEqual(app.session_state.active_view, view)
            self.assertEqual(app.main.markdown[0].value, view)
            app.run()
            self.assertEqual(app.radio[0].id, navigation_id)
            self.assertEqual(app.radio[0].value, view)
            self.assertEqual(app.session_state.active_view, view)

        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio[0].value, "Settings")
        app.radio[0].set_value("Ask").run()
        self.assertEqual(app.session_state.active_view, "Ask")

    def test_recommended_onboarding_download_and_retry(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                import streamlit as st
                import document_analyst.app as application
                from document_analyst.config import AppSettings

                class Models:
                    def local_embedding_model_exists(self):
                        return bool(st.session_state.get("installed"))
                    def local_llm_model_path(self):
                        return "model.gguf" if st.session_state.get("installed") else None
                class Service:
                    models = Models()
                def save(settings):
                    assert settings.embeddings_repo == AppSettings().embeddings_repo
                    assert settings.llm_repo == AppSettings().llm_repo
                    assert settings.llm_filename == ""
                    assert settings.documents_dir == "/my/documents"
                def download(service, update):
                    update("embeddings", "Search model", 50, 100)
                    update("llm", "Answer model", 100, 100)
                    if not st.session_state.get("retried"):
                        st.session_state.retried = True
                        raise RuntimeError("Connection interrupted")
                    st.session_state.installed = True
                    return "embeddings", "model.gguf"
                application.save_settings = save
                application._cached_service = lambda *args: Service()
                application._download_models_with_updates = download
                application._empty_state_dialog(
                    AppSettings(documents_dir="/my/documents", llm_repo="custom/model",
                                llm_filename="custom.gguf"), Service()
                )
                """
            ),
            default_timeout=15,
        ).run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == "Proceed with recommended settings").click().run()
        self.assertFalse(app.exception)
        self.assertIn("Connection interrupted", app.error[0].value)
        self.assertTrue(app.session_state.retried)
        next(b for b in app.button if b.label == "Proceed with recommended settings").click().run()
        self.assertFalse(app.exception)
        self.assertIn("Your models are ready", app.success[0].value)
        next(b for b in app.button if b.label == "Add documents").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.active_view, "Library")
        self.assertFalse(app.session_state.show_empty_state_dialog)

    def test_chat_toolbar_preserves_scope_and_streams_saved_answers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = str(Path(directory) / "conversations.json")
            app = AppTest.from_string(dedent(f"""
                import streamlit as st
                from pathlib import Path
                from document_analyst.app import _chat_tab, _initialize_conversation_state
                from document_analyst.ui.state import initialize_session_state
                from document_analyst.services.conversation_store import ConversationStore
                from document_analyst.services.rag import AnswerStream

                initialize_session_state()
                store = ConversationStore(Path({store_path!r}))
                _initialize_conversation_state(store)
                class Service:
                    def indexed_documents(self):
                        return [{{"source_path": "/docs/book.pdf", "document_name": "book.pdf"}}]
                    def answer_question_stream(self, prompt, history, **kwargs):
                        st.session_state.request_scope = kwargs["source_paths"]
                        return AnswerStream(iter(["A clear ", "answer."]), [])
                _chat_tab(Service(), store)
            """), default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("popover")), 3)
            app.multiselect[0].set_value(["/docs/book.pdf"]).run()
            app.chat_input[0].set_value("Explain the book").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.request_scope, ["/docs/book.pdf"])
            self.assertEqual(app.session_state.messages[-1]["content"], "A clear answer.")
            self.assertEqual(len(ConversationStore(Path(store_path)).list()[0]["messages"]), 2)

    def test_empty_chat_guides_user_to_library(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = AppTest.from_string(dedent(f"""
                from pathlib import Path
                from types import SimpleNamespace
                from document_analyst.app import _chat_tab, _initialize_conversation_state
                from document_analyst.ui.state import initialize_session_state
                from document_analyst.services.conversation_store import ConversationStore
                initialize_session_state()
                store = ConversationStore(Path({str(Path(directory) / "conversations.json")!r}))
                _initialize_conversation_state(store)
                _chat_tab(SimpleNamespace(indexed_documents=lambda: []), store)
            """), default_timeout=15).run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)
            next(b for b in app.button if b.label == "Go to Library").click().run()
            self.assertEqual(app.session_state.active_view, "Library")

    def _library_app(self):
        return AppTest.from_string(dedent("""
            import streamlit as st
            from document_analyst.app import _library_documents_panel
            docs = [{"document_name": f"book{i}.pdf", "source_path": f"/docs/book{i}.pdf",
                     "chunks": 23, "status": "failed" if i == 1 else "indexed"} for i in range(12)]
            docs.append({"document_name": "notes.txt", "source_path": "/docs/notes.txt",
                         "chunks": 0, "status": "failed"})
            class Service:
                def document_chunks(self, path, offset=0, limit=10):
                    st.session_state.fetch = (path, offset, limit)
                    return [{"sequence": i, "approx_page": i + 1, "text": f"Stored passage {i}"}
                            for i in range(offset, min(offset + limit, 23))]
            _library_documents_panel(Service(), docs)
        """), default_timeout=15).run()

    def test_library_search_filters_combine_and_clear(self) -> None:
        app = self._library_app()
        self.assertFalse(app.exception)
        app.text_input[0].set_value("BOOK1").run()
        self.assertIn("Showing 3 of 13 documents", [c.value for c in app.caption])
        next(m for m in app.multiselect if m.label == "Indexing status").set_value(["indexed"]).run()
        self.assertIn("Showing 2 of 13 documents", [c.value for c in app.caption])
        next(m for m in app.multiselect if m.label == "File type").set_value([".txt"]).run()
        self.assertIn("No matching documents", "\n".join(m.value for m in app.markdown))
        next(b for b in app.button if b.label == "Clear filters").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_input[0].value, "")
        self.assertTrue(all(m.value == [] for m in app.multiselect))
        self.assertIn("Showing 13 of 13 documents", [c.value for c in app.caption])
        self.assertNotIn("fetch", app.session_state)

    def test_library_paginates_documents_and_fetches_passages_only_on_request(self) -> None:
        app = self._library_app()
        self.assertEqual(len([b for b in app.button if b.label == "View passages"]), 10)
        next(n for n in app.number_input if n.label == "Library page").set_value(2).run()
        self.assertEqual(len([b for b in app.button if b.label == "View passages"]), 2)
        next(b for b in app.button if b.key == "view-/docs/book10.pdf").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.fetch, ("/docs/book10.pdf", 0, 10))
        self.assertTrue(any("Extracted passages: book10.pdf" in h.value for h in app.subheader))

    def test_passage_viewer_paginates_without_fetching_all_chunks(self) -> None:
        app = AppTest.from_string(dedent("""
            import streamlit as st
            from document_analyst.app import _document_passages_panel
            class Service:
                def document_chunks(self, path, offset=0, limit=10):
                    st.session_state.fetch = (path, offset, limit)
                    return [{"sequence": i, "approx_page": i + 1, "text": f"Stored passage {i}"}
                            for i in range(offset, min(offset + limit, 23))]
            _document_passages_panel(Service(), {"document_name": "book10.pdf",
                "source_path": "/docs/book10.pdf", "chunks": 23})
        """), default_timeout=15).run()
        next(n for n in app.number_input if n.label == "Passage page").set_value(3).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.fetch, ("/docs/book10.pdf", 20, 10))
        self.assertEqual(len(app.text), 3)
        self.assertEqual(app.text[0].value, "Stored passage 20")
        self.assertIn("Passages 21–23 of 23", [c.value for c in app.caption])

    def test_performance_controls_apply_preset_and_batch_size(self) -> None:
        app = AppTest.from_string(dedent("""
            import streamlit as st
            from types import SimpleNamespace
            import document_analyst.app as application
            from document_analyst.config import AppSettings

            if "settings" not in st.session_state:
                st.session_state.settings = AppSettings()
            def save(settings):
                st.session_state.saved_batch = settings.embedding_batch_size
                st.session_state.saved_mode = settings.chunking_mode
                st.session_state.saved_size = settings.chunk_size
            application.save_settings = save
            service = SimpleNamespace(models=SimpleNamespace(local_embedding_model_exists=lambda: False))
            application._indexing_performance_controls(st.session_state.settings, service, False)
        """), default_timeout=15).run()
        self.assertFalse(app.exception)
        app.selectbox[0].select("Fast")
        app.number_input[0].set_value(64)
        next(b for b in app.button if b.label == "Apply performance settings").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.saved_batch, 64)
        self.assertEqual(app.session_state.saved_mode, "fast")
        self.assertEqual(app.session_state.saved_size, 1000)
        self.assertTrue(next(b for b in app.button if b.label == "Benchmark batch sizes").disabled)

    def test_batch_benchmark_displays_results_without_changing_settings(self) -> None:
        app = AppTest.from_string(dedent("""
            import streamlit as st
            from types import SimpleNamespace
            import document_analyst.app as application
            from document_analyst.config import AppSettings
            from document_analyst.services.indexing_performance import BatchBenchmark
            if "settings" not in st.session_state:
                st.session_state.settings = AppSettings()
            def benchmark(model, update):
                update(1, 2, 32)
                return [BatchBenchmark(32, 2.0, 128.0), BatchBenchmark(64, 1.0, 256.0)]
            application.benchmark_embedding_batches = benchmark
            service = SimpleNamespace(
                models=SimpleNamespace(local_embedding_model_exists=lambda: True, embedding_device="cpu"),
                ensure_embedder=lambda: object(),
            )
            application._indexing_performance_controls(st.session_state.settings, service, False)
        """), default_timeout=15).run()
        next(b for b in app.button if b.label == "Benchmark batch sizes").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 1)
        self.assertIn("Fastest measured batch size: 64", "\n".join(item.value for item in app.caption))
        self.assertEqual(app.session_state.settings.embedding_batch_size, 32)

    def test_performance_controls_disable_changes_while_indexing(self) -> None:
        app = AppTest.from_string(dedent("""
            from types import SimpleNamespace
            from document_analyst.app import _indexing_performance_controls
            from document_analyst.config import AppSettings
            service = SimpleNamespace(models=SimpleNamespace(local_embedding_model_exists=lambda: True))
            _indexing_performance_controls(AppSettings(), service, True)
        """), default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertTrue(all(b.disabled for b in app.button))
        self.assertTrue(app.number_input[0].disabled)
        self.assertTrue(app.selectbox[0].disabled)

    def test_live_index_progress_shows_device_rate_stage_and_cancel(self) -> None:
        app = AppTest.from_string(dedent("""
            import time
            import streamlit as st
            from types import SimpleNamespace
            from document_analyst.app import _IndexJob, _index_job_panel
            from document_analyst.services.index_progress import IndexProgressEvent

            job = st.session_state.get("index_job")
            if not job:
                now = time.monotonic()
                job = _IndexJob("/docs", False, started_at=now - 20)
                job.thread = SimpleNamespace(is_alive=lambda: True)
                for done, timestamp in ((32, now - 4), (64, now - 2), (96, now)):
                    job.events.put((IndexProgressEvent("embedding", "book.pdf", 1, 3,
                                                      False, done, 160, "passages"), timestamp))
                st.session_state.index_job = job
            _index_job_panel(SimpleNamespace(models=SimpleNamespace(embedding_device="mps")))
        """), default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertEqual([m.label for m in app.metric], ["Elapsed time", "Embedding device"])
        self.assertEqual(app.metric[1].value, "Apple MPS")
        captions = "\n".join(item.value for item in app.caption)
        self.assertIn("File 1 of 3 · book.pdf", captions)
        self.assertIn("passages / second", captions)
        self.assertIn("Estimated time left in this stage", captions)
        self.assertEqual(len(app.get("progress")), 2)
        next(b for b in app.button if b.label == "Cancel indexing").click().run()
        self.assertTrue(app.session_state.index_job.cancel_event.is_set())

    def test_completed_index_shows_stable_elapsed_time(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                import streamlit as st
                from document_analyst.app import _IndexJob, _index_job_panel
                from document_analyst.services.rag import IndexResult

                if "index_job" not in st.session_state:
                    job = _IndexJob("/docs", False, started_at=100.0, finished_at=225.0)
                    job.result = IndexResult(document_count=3, chunk_count=100, warnings=[])
                    st.session_state.index_job = job
                _index_job_panel(None)
                """
            ),
            default_timeout=15,
        ).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].label, "Total elapsed time")
        self.assertEqual(app.metric[0].value, "2 min 5 s")
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "2 min 5 s")

    def test_completed_index_automatically_refreshes_library_without_dismiss(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                import streamlit as st
                from document_analyst.app import _IndexJob, _docs_tab
                from document_analyst.config import AppSettings
                from document_analyst.services.rag import IndexResult

                st.session_state.renders = st.session_state.get("renders", 0) + 1
                if "index_job" not in st.session_state:
                    job = _IndexJob("/docs", False, started_at=10.0, finished_at=20.0)
                    job.result = IndexResult(document_count=1, chunk_count=42, warnings=[])
                    st.session_state.index_job = job
                class Service:
                    def document_statuses(self):
                        return [{"document_name": "book.pdf", "source_path": "/docs/book.pdf",
                                 "chunks": 42, "status": "indexed", "indexed_at": "Today"}]
                _docs_tab(AppSettings(), Service())
                """
            ),
            default_timeout=15,
        ).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.renders, 2)
        self.assertEqual([e.label for e in app.expander], ["Add a document folder", "Performance options"])
        markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("book.pdf", markup)
        self.assertTrue(any("42 chunks" in c.value for c in app.caption))
        self.assertEqual(app.metric[0].value, "10.0 s")
        self.assertFalse(any(b.label in ("Dismiss", "Cancel indexing") for b in app.button))
        self.assertFalse(next(b for b in app.button if b.label == "Build Index").disabled)
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.renders, 3)

    def test_index_worker_records_completion_time_even_on_failure(self) -> None:
        from unittest.mock import Mock, patch
        import document_analyst.app as application

        for fails in (False, True):
            with self.subTest(fails=fails):
                service = Mock()
                if fails:
                    service.index_documents.side_effect = RuntimeError("Index failed")
                from types import SimpleNamespace
                state = SimpleNamespace(get=lambda key: None)
                with patch.object(application.st, "session_state", state):
                    application._start_index_job(service, "/docs")
                    job = state.index_job
                    job.thread.join(timeout=5)
                self.assertFalse(job.thread.is_alive())
                self.assertIsNotNone(job.finished_at)
                self.assertGreaterEqual(job.finished_at, job.started_at)
                self.assertEqual(job.error is not None, fails)

    def test_help_collects_guidance_away_from_chat_controls(self) -> None:
        app = AppTest.from_string(
            "from document_analyst.app import _help_tab\n_help_tab()",
            default_timeout=15,
        ).run()
        self.assertFalse(app.exception)
        self.assertEqual([item.label for item in app.expander], [
            "Getting started", "Conversations and downloads",
            "Answers, sources, and search controls", "Indexing speed and troubleshooting",
            "Models and privacy",
        ])
        markup = "\n".join(item.value for item in app.markdown)
        for text in ("Download conversation", "Download Markdown (.md)",
                     "Download PDF (.pdf)", "BM25", "Fast — sentence boundaries"):
            self.assertIn(text, markup)

    def test_model_repository_has_a_readable_display_name(self) -> None:
        self.assertEqual(
            model_display_name("lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF"),
            "Qwen2.5 1.5B Instruct",
        )

    def test_advanced_model_settings_are_opt_in(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                from document_analyst.config import AppSettings
                import document_analyst.app as application

                application._cached_model_catalog = lambda: (
                    ["sentence-transformers/all-MiniLM-L6-v2"],
                    ["lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF"],
                )

                class Models:
                    embedding_device = "mps"
                    def local_llm_model_path(self):
                        return None
                    def local_embedding_model_exists(self):
                        return False

                class Service:
                    models = Models()

                application._models_tab(AppSettings(), Service())
                """
            ),
            default_timeout=15,
        ).run()

        self.assertFalse(app.exception)
        markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("Model library", markup)
        self.assertIn("Not installed", markup)
        self.assertIn("Embedding device: Apple GPU (MPS)", "\n".join(item.value for item in app.caption))
        self.assertEqual(len(app.slider), 0)

        advanced = next(toggle for toggle in app.toggle if toggle.label == "Enable advanced settings")
        advanced.set_value(True).run()

        self.assertFalse(app.exception)
        self.assertGreaterEqual(len(app.slider), 6)
        self.assertEqual([tab.label for tab in app.tabs], ["Retrieval", "Indexing", "OCR"])

    def test_model_cards_refresh_after_download(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                from pathlib import Path
                import streamlit as st
                from document_analyst.config import AppSettings
                import document_analyst.app as application

                application._cached_model_catalog = lambda: (
                    ["sentence-transformers/all-MiniLM-L6-v2"],
                    ["lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF"],
                )

                class Models:
                    def local_llm_model_path(self):
                        return Path("/models/model.gguf") if st.session_state.get("installed") else None
                    def local_embedding_model_exists(self):
                        return bool(st.session_state.get("installed"))

                class Service:
                    models = Models()

                def download(service, update):
                    st.session_state.installed = True
                    return "/models/embeddings", "/models/model.gguf"

                application._download_models_with_updates = download
                application._models_tab(AppSettings(), Service())
                """
            ),
            default_timeout=15,
        ).run()

        self.assertFalse(app.exception)
        download = next(
            button for button in app.button if button.label == "Download missing models"
        )
        download.click().run()

        self.assertFalse(app.exception)
        markup = "\n".join(item.value for item in app.markdown)
        self.assertEqual(markup.count("Installed"), 2)
        self.assertTrue(
            any(button.label == "Verify model installation" for button in app.button)
        )

    def test_theme_defines_shared_tokens_and_responsive_layout(self) -> None:
        self.assertIn("--da-accent", THEME_CSS)
        self.assertIn("--da-radius", THEME_CSS)
        self.assertIn("@media (max-width: 900px)", THEME_CSS)
        self.assertNotIn("hero-card", THEME_CSS)

    def test_shared_headers_and_empty_state_render(self) -> None:
        app = AppTest.from_string(
            dedent(
                """
                from document_analyst.ui.components import sidebar_stats, empty_state, page_heading
                from document_analyst.ui.theme import inject_theme

                inject_theme()
                sidebar_stats(12, 480)
                page_heading("Library")
                empty_state("Your library is empty", "Choose a folder to begin.", "▤")
                """
            ),
            default_timeout=15,
        ).run()

        self.assertFalse(app.exception)
        markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("Document library", markup)
        self.assertIn("<strong>12</strong>Documents", markup)
        self.assertIn("<strong>480</strong>Chunks", markup)
        self.assertNotIn("da-app-header", markup)
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
            self.assertIn("#### Download conversation", [item.value for item in app.markdown])
            self.assertEqual(
                [item.proto.label for item in app.get("download_button")],
                ["Download Markdown (.md)", "Download PDF (.pdf)"],
            )
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
