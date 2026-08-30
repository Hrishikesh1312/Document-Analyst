from __future__ import annotations

import html
import hashlib
import json
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import streamlit as st

from document_analyst.config import (
    AppSettings,
    EMBEDDING_REPO_OPTIONS,
    LLM_REPO_OPTIONS,
    load_settings,
    save_settings,
)
from document_analyst.models import SourceRecord
from document_analyst.services.conversation_export import conversation_markdown, conversation_pdf
from document_analyst.services.conversation_store import ConversationStore
from document_analyst.services.rag import RagService
from document_analyst.services.model_manager import discover_model_repositories
from document_analyst.ui.components import (
    app_header,
    empty_state,
    model_display_name,
    model_status_card,
    page_heading,
    section_intro,
    sidebar_brand,
    sidebar_stats,
)
from document_analyst.ui.state import initialize_session_state
from document_analyst.ui.theme import inject_theme

SERVICE_CACHE_VERSION = "document-lifecycle-v3"


@dataclass
class _IndexJob:
    directory: str
    replace_existing: bool
    retry_failed_only: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue = field(default_factory=queue.Queue)
    result: object | None = None
    error: Exception | None = None
    thread: threading.Thread | None = None
    last_progress: tuple[str, str, int, int, bool] | None = None


def run() -> None:
    st.set_page_config(
        page_title="Document Analyst",
        page_icon=":material/search:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_theme()
    initialize_session_state()
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()

    conversation_store = _cached_conversation_store()
    _initialize_conversation_state(conversation_store)

    reset_notice = st.session_state.pop("reset_notice", None)
    if reset_notice:
        st.toast(reset_notice, icon=":material/check_circle:")

    settings: AppSettings = st.session_state.settings
    service = _cached_service(
        json.dumps(asdict(settings), sort_keys=True),
        SERVICE_CACHE_VERSION,
    )
    stats = service.stats()

    if stats["documents"] > 0 or stats["chunks"] > 0:
        st.session_state.show_empty_state_dialog = False

    active_view = _sidebar(settings, service, conversation_store)
    app_header(
        active_view,
        stats["documents"],
        stats["chunks"],
        bool(service.models.local_llm_model_path()),
    )

    if (
        stats["documents"] == 0
        and stats["chunks"] == 0
        and st.session_state.show_empty_state_dialog
    ):
        _empty_state_dialog()

    if active_view == "Ask":
        _chat_tab(service, conversation_store)
    elif active_view == "Library":
        _docs_tab(settings, service)
    else:
        _models_tab(settings, service)


def _sidebar(
    settings: AppSettings, service: RagService, conversation_store: ConversationStore
) -> str:
    with st.sidebar:
        sidebar_brand()
        views = ["Ask", "Library", "Settings"]
        view_labels = {
            "Ask": "⌕  Ask",
            "Library": "▤  Library",
            "Settings": "⚙  Settings",
        }
        active_view = st.radio(
            "Workspace navigation",
            views,
            index=views.index(st.session_state.active_view),
            format_func=lambda view: view_labels[view],
            label_visibility="collapsed",
        )
        st.session_state.active_view = active_view
        stats = service.stats()
        sidebar_stats(stats["documents"], stats["chunks"])
        st.caption("Files and inference remain on this device.")
        with st.container(key="sidebar_reset"):
            if st.button(
                "Reset",
                icon=":material/restart_alt:",
                use_container_width=True,
                type="secondary",
            ):
                _reset_dialog(service, conversation_store)
    return active_view


@st.dialog("Reset application data")
def _reset_dialog(service: RagService, conversation_store: ConversationStore) -> None:
    st.warning("Choose what to clear. Deleted vector data and model files cannot be recovered.")
    clear_vector_db = st.checkbox("Vector database", value=False)
    clear_chat = st.checkbox("Chat history", value=False)
    clear_models = st.checkbox("Downloaded local models", value=False)

    cancel, confirm = st.columns(2)
    with cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with confirm:
        selected = clear_vector_db or clear_chat or clear_models
        if st.button(
            "Clear selected",
            type="primary",
            use_container_width=True,
            disabled=not selected,
        ):
            cleared: list[str] = []
            try:
                if clear_vector_db:
                    service.store.reset()
                    service.manifest.reset()
                    cleared.append("vector database")
                if clear_chat:
                    st.session_state.messages = []
                    st.session_state.sources_by_turn = {}
                    st.session_state.diagnostics_by_turn = {}
                    _persist_current_conversation(conversation_store)
                    cleared.append("chat history")
                if clear_models:
                    service.unload_models()
                    service.models.delete_downloaded_models()
                    cleared.append("local models")
            except OSError as exc:
                st.error(f"Reset failed: {exc}")
                return

            _cached_service.clear()
            if clear_vector_db:
                st.session_state.show_empty_state_dialog = False
            st.session_state.reset_notice = f"Cleared {', '.join(cleared)}."
            st.rerun()


@st.dialog("No Documents Yet", width="large")
def _empty_state_dialog() -> None:
    st.markdown(
        """
        Your local index is empty right now.

        To get started:
        1. Open `Settings` and configure or download your embedding / GGUF models.
        2. Open `Library` and add the local document folder you want to index.
        3. Build the index, then come back to `Ask` to start asking questions.
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Open Settings", use_container_width=True):
            st.session_state.active_view = "Settings"
            st.session_state.show_empty_state_dialog = False
            st.rerun()
    with col2:
        if st.button("Open Library", use_container_width=True):
            st.session_state.active_view = "Library"
            st.session_state.show_empty_state_dialog = False
            st.rerun()
    with col3:
        if st.button("Dismiss", use_container_width=True):
            st.session_state.show_empty_state_dialog = False
            st.rerun()


def _chat_tab(service: RagService, conversation_store: ConversationStore) -> None:
    page_heading("Ask")
    indexed_documents = service.indexed_documents()
    document_labels = {
        str(item["source_path"]): str(item["document_name"]) for item in indexed_documents
    }
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        if not st.session_state.messages:
            empty_state(
                "Start with a question",
                "Ask for an explanation, comparison, summary, or supporting passage from your indexed documents.",
                "⌕",
            )
        for index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.markdown(_citation_links(message["content"], index))
                    with st.expander("Copy answer with citations", expanded=False):
                        st.code(message["content"], language=None)
                else:
                    st.markdown(message["content"])
                if message["role"] == "assistant":
                    sources = st.session_state.sources_by_turn.get(index, [])
                    if sources:
                        with st.expander("Sources", expanded=False):
                            _sources_panel(sources, f"turn-{index}")
                    diagnostics = st.session_state.diagnostics_by_turn.get(index)
                    if diagnostics and st.session_state.show_retrieval_diagnostics:
                        with st.expander("Retrieval diagnostics", expanded=False):
                            _diagnostics_panel(diagnostics)

        prompt = st.chat_input("Ask about your indexed documents")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching your local index and generating a response..."):
                        selected_paths = st.session_state.get("chat_document_filter", [])
                        source_paths = selected_paths or None
                        result = service.answer_question_stream(
                            prompt,
                            st.session_state.messages[:-1],
                            source_paths=source_paths,
                            pinned_source_paths=st.session_state.pinned_source_paths,
                            excluded_source_paths=st.session_state.excluded_source_paths,
                        )
                        answer = st.write_stream(result.chunks)
                except (OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Could not answer the question: {exc}")
                    st.session_state.messages.pop()
                    return
                turn_index = len(st.session_state.messages)
                with st.expander("Sources", expanded=True):
                    _sources_panel(result.sources, f"turn-{turn_index}")
                if result.diagnostics and st.session_state.show_retrieval_diagnostics:
                    with st.expander("Retrieval diagnostics", expanded=True):
                        _diagnostics_panel(result.diagnostics)
                st.session_state.messages.append({"role": "assistant", "content": str(answer)})
                st.session_state.sources_by_turn[turn_index] = result.sources
                if result.diagnostics:
                    st.session_state.diagnostics_by_turn[turn_index] = result.diagnostics
                if st.session_state.conversation_name == "New conversation":
                    st.session_state.conversation_name = prompt.strip()[:60] or "New conversation"
                _persist_current_conversation(conversation_store)
                st.session_state.pinned_source_paths = []
                st.session_state.excluded_source_paths = []
            st.rerun()

    with right:
        _conversation_panel(conversation_store)
        st.markdown("#### Retrieval tools")
        st.toggle(
            "Show retrieval diagnostics",
            key="show_retrieval_diagnostics",
            help="Inspect hybrid-search candidates, score components, document coverage, and timing.",
        )
        st.text_input(
            "Search retrieved evidence",
            key="evidence_search",
            placeholder="Highlight or filter source passages",
        )
        if st.session_state.pinned_source_paths or st.session_state.excluded_source_paths:
            st.caption(
                f"Next response: {len(st.session_state.pinned_source_paths)} pinned, "
                f"{len(st.session_state.excluded_source_paths)} excluded."
            )
            if st.button("Clear next-response source rules", use_container_width=True):
                st.session_state.pinned_source_paths = []
                st.session_state.excluded_source_paths = []
                st.rerun()
        st.markdown("#### Search scope")
        st.multiselect(
            "Documents",
            options=list(document_labels),
            format_func=lambda path: document_labels.get(path, Path(path).name),
            placeholder="All indexed documents",
            key="chat_document_filter",
            help="Leave empty to search the entire index.",
        )
        st.caption(
            f"Semantic floor {service.settings.retrieval_min_score:.2f}; strong BM25 matches can still qualify."
        )
        st.markdown("#### How answers are built")
        st.markdown(
            """
            1. Your question is embedded locally.
            2. Semantic and BM25 searches build a shared candidate pool.
            3. Candidates are reranked and diversified across documents.
            4. The local GGUF model answers with inline citations like `[S1]`.
            """
        )
        st.markdown("#### Chat behavior")
        st.caption("Only the latest conversation window is included in each prompt to keep CPU latency reasonable.")


def _docs_tab(settings: AppSettings, service: RagService) -> None:
    page_heading("Library")
    active_job = st.session_state.get("index_job")
    indexing = bool(active_job and active_job.thread and active_job.thread.is_alive())
    col1, col2 = st.columns([1.3, 1], gap="large")
    with col1:
        st.subheader("Index a Folder")
        with st.form("index-form", clear_on_submit=False):
            directory = st.text_input(
                "Local documents folder",
                value=settings.documents_dir,
                placeholder=str(Path.home()),
            )
            replace_existing = st.checkbox("Replace existing index contents", value=False)
            submitted = st.form_submit_button(
                "Build Index", use_container_width=True, disabled=indexing
            )

        if submitted:
            settings.documents_dir = directory.strip()
            save_settings(settings)
            st.session_state.settings = settings
            try:
                legacy_presentations = service.ingestor.discover_legacy_presentations(
                    settings.documents_dir
                )
            except (OSError, ValueError) as exc:
                st.error(f"Indexing failed: {exc}")
                return
            if legacy_presentations:
                _legacy_ppt_dialog(legacy_presentations)
                return
            _start_index_job(service, settings.documents_dir, replace_existing)
            st.rerun()

        if st.session_state.index_job:
            _index_job_panel(service)

    with col2:
        st.subheader("Index Settings")
        st.markdown(
            f"""
            - Max file size: `{settings.max_file_size_mb}MB`
            - Chunk size: `{settings.chunk_size}` chars
            - Overlap: `{settings.chunk_overlap}` chars
            - Semantic threshold: `{settings.semantic_threshold}`
            - Top-k retrieval: `{settings.top_k}`
            - Minimum relevance: `{settings.retrieval_min_score:.2f}`
            - Supported formats: `{", ".join(settings.supported_extensions)}`
            - OCR fallback: `{"On" if settings.enable_ocr else "Off"}`
            """
        )

    st.write("")
    st.subheader("Indexed Documents")
    docs = service.document_statuses()
    if not docs:
        empty_state(
            "Your library is empty",
            "Choose a local folder above to build your private, searchable knowledge base.",
            "▤",
        )
        return

    for item in docs:
        document_name = html.escape(str(item["document_name"]))
        source_path = html.escape(str(item["source_path"]))
        with st.container():
            st.markdown(
                f"""
                <div class="source-card">
                    <strong>{document_name}</strong><br/>
                    <span class="muted">{source_path}</span><br/>
                    <span class="accent">{item.get("chunks", 0)} chunks • {item.get("status", "indexed")}</span><br/>
                    <span class="muted">Indexed: {item.get("indexed_at") or "Not yet"}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if item.get("error"):
                st.caption(f"Status detail: {item['error']}")
            if item.get("duplicate_of"):
                st.caption(f"Duplicate of: {item['duplicate_of']}")
            if item.get("status") != "removed" and st.button(
                f"Remove from index: {item['document_name']}", key=f"del-{item['source_path']}"
            ):
                service.delete_document(str(item["source_path"]))
                st.success(f"Removed {item['document_name']} from the index.")
                st.rerun()


def _start_index_job(
    service: RagService,
    directory: str,
    replace_existing: bool = False,
    retry_failed_only: bool = False,
) -> None:
    current = st.session_state.get("index_job")
    if current and current.thread and current.thread.is_alive():
        raise RuntimeError("An indexing job is already running.")
    job = _IndexJob(directory, replace_existing, retry_failed_only)

    def progress(phase: str, item: str, current: int, total: int, complete: bool) -> None:
        job.events.put((phase, item, current, total, complete))

    def run_job() -> None:
        try:
            job.result = service.index_documents(
                directory,
                replace_existing=replace_existing,
                progress=progress,
                should_cancel=job.cancel_event.is_set,
                retry_failed_only=retry_failed_only,
            )
        except Exception as exc:
            job.error = exc

    job.thread = threading.Thread(target=run_job, name="document-index", daemon=True)
    st.session_state.index_job = job
    job.thread.start()


@st.fragment(run_every=0.5)
def _index_job_panel(service: RagService) -> None:
    job: _IndexJob | None = st.session_state.get("index_job")
    if not job:
        return
    while not job.events.empty():
        try:
            job.last_progress = job.events.get_nowait()
        except queue.Empty:
            break
    progress_value = 0.0
    progress_text = "Preparing incremental index…"
    if job.last_progress:
        phase, item, current, total, complete = job.last_progress
        ranges = {
            "hashing": (0.0, 0.15),
            "reading": (0.15, 0.30),
            "chunking": (0.30, 0.50),
            "embedding": (0.50, 0.85),
            "writing": (0.85, 1.0),
        }
        start, end = ranges.get(phase, (0.0, 1.0))
        completed = current if complete else max(0, current - 1)
        progress_value = start + ((end - start) * (completed / total if total else 0.0))
        progress_text = f"{phase.title()} {current}/{total}: {item}"
    alive = bool(job.thread and job.thread.is_alive())
    st.progress(min(progress_value, 1.0), text=progress_text)
    if alive:
        if st.button("Cancel indexing", type="secondary", use_container_width=True):
            job.cancel_event.set()
            st.warning("Cancellation requested. The current safe checkpoint will finish first.")
        return
    if job.error:
        st.error(f"Indexing failed: {job.error}")
    elif job.result:
        result = job.result
        if result.cancelled:
            st.warning("Indexing was cancelled. Completed files remain safely indexed.")
        else:
            st.success("Incremental indexing complete.")
        st.markdown(
            f"Indexed `{result.document_count}` changed/new files into `{result.chunk_count}` chunks · "
            f"unchanged `{result.unchanged_count}` · duplicates `{result.duplicate_count}` · "
            f"failed `{result.failed_count}` · removed `{result.removed_count}`"
        )
        for warning in result.warnings:
            st.warning(warning)
        retry, dismiss = st.columns(2)
        if result.failed_count and retry.button("Retry failed files", use_container_width=True):
            _start_index_job(service, job.directory, retry_failed_only=True)
            st.rerun(scope="app")
        if dismiss.button("Dismiss", use_container_width=True):
            st.session_state.index_job = None
            st.rerun(scope="app")


def _models_tab(settings: AppSettings, service: RagService) -> None:
    page_heading("Settings")
    download_notice = st.session_state.pop("model_download_notice", None)
    if download_notice:
        st.toast(download_notice, icon=":material/check_circle:")
    local_llm = service.models.local_llm_model_path()
    embedding_ready = service.models.local_embedding_model_exists()
    llm_ready = local_llm is not None
    catalog_warning = ""
    try:
        live_embeddings, live_llms = _cached_model_catalog()
    except Exception as exc:  # The settings screen must remain usable offline.
        live_embeddings, live_llms = [], []
        catalog_warning = str(exc)
    embedding_options = _repo_options(
        live_embeddings or EMBEDDING_REPO_OPTIONS, settings.embeddings_repo
    )
    llm_options = _repo_options(live_llms or LLM_REPO_OPTIONS, settings.llm_repo)
    if (
        settings.enable_ocr
        and not _tesseract_available(settings.tesseract_cmd)
        and not st.session_state.get("ocr_setup_shown", False)
    ):
        st.session_state.ocr_setup_shown = True
        _ocr_setup_dialog(settings.tesseract_cmd)
    section_intro(
        "Model library",
        "Select one model for document search and one model for answer generation.",
    )
    status_left, status_right = st.columns(2, gap="medium")
    with status_left:
        model_status_card(
            "Embedding model",
            settings.embeddings_repo,
            settings.embeddings_dir if embedding_ready else "Required to index and search documents",
            embedding_ready,
        )
    with status_right:
        model_status_card(
            "Answer model",
            settings.llm_repo,
            str(local_llm) if local_llm else "Required to generate complete answers",
            llm_ready,
        )

    if catalog_warning:
        st.info("The online catalog is unavailable. Built-in model options are shown instead.")
    else:
        st.caption("Model options are refreshed from public Hugging Face repositories every hour.")

    with st.form("model-selection-form"):
        model_left, model_right = st.columns(2, gap="medium")
        with model_left:
            embeddings_repo = st.selectbox(
                "Embedding model",
                options=embedding_options,
                index=embedding_options.index(settings.embeddings_repo),
                format_func=model_display_name,
                help="Creates vector representations used to locate relevant passages.",
            )
            st.caption(f"Repository: `{embeddings_repo}`")
        with model_right:
            llm_repo = st.selectbox(
                "Answer model",
                options=llm_options,
                index=llm_options.index(settings.llm_repo),
                format_func=model_display_name,
                help="Generates cited answers from retrieved passages using llama.cpp.",
            )
            st.caption(f"Repository: `{llm_repo}`")
        apply_selection = st.form_submit_button(
            "Apply model selection",
            type="primary",
            use_container_width=True,
        )
    if apply_selection:
        llm_changed = llm_repo.strip() != settings.llm_repo
        settings.embeddings_repo = embeddings_repo.strip()
        settings.llm_repo = llm_repo.strip()
        if llm_changed:
            settings.llm_filename = ""
        save_settings(settings)
        st.session_state.settings = settings
        st.toast("Model selection updated.", icon=":material/check_circle:")
        st.rerun()

    models_ready = embedding_ready and llm_ready
    if st.button(
        "Verify model installation" if models_ready else "Download missing models",
        type="secondary" if models_ready else "primary",
        use_container_width=True,
        help="Existing model files are reused. Only missing files are downloaded.",
    ):
        overall_progress = st.progress(0, text="Preparing model downloads…")
        file_progress = st.progress(0, text="Waiting for the first file…")

        def update_download_progress(
            phase: str, description: str, downloaded: int, total: int | None
        ) -> None:
            phase_number = 1 if phase == "embeddings" else 2
            phase_name = "Embedding model" if phase == "embeddings" else "Language model"
            file_fraction = min(downloaded / total, 1.0) if total and total > 0 else 0.0
            overall_fraction = ((phase_number - 1) + file_fraction) / 2
            overall_progress.progress(
                overall_fraction,
                text=f"Step {phase_number} of 2: {phase_name}",
            )
            size_text = _format_download_size(downloaded, total)
            file_progress.progress(
                file_fraction,
                text=f"{description} · {size_text}",
            )

        try:
            embedding_dir, llm_path = _download_models_with_updates(
                service, update_download_progress
            )
        except Exception as exc:  # Hugging Face exposes several transport-specific errors.
            st.error(f"Model download failed: {exc}")
            return
        overall_progress.progress(1.0, text="Both models are ready")
        file_progress.progress(1.0, text="Download complete")
        st.session_state.model_download_notice = (
            f"Models ready. Embedding model: {embedding_dir}. Answer model: {llm_path}."
        )
        st.rerun()

    st.divider()
    section_intro(
        "Advanced settings",
        "Change retrieval, indexing, and OCR behavior only when the defaults are unsuitable.",
    )
    show_advanced = st.toggle(
        "Enable advanced settings",
        key="show_advanced_settings",
        help="These controls affect retrieval quality, index construction, and processing time.",
    )
    if not show_advanced:
        st.caption("Recommended defaults are active. Enable advanced settings to modify them.")
        return

    with st.form("advanced-settings-form"):
        retrieval_tab, indexing_tab, ocr_tab = st.tabs(["Retrieval", "Indexing", "OCR"])
        with retrieval_tab:
            top_k = st.slider(
                "Retrieved passages",
                min_value=2,
                max_value=8,
                value=settings.top_k,
                help="Maximum number of source passages supplied to the answer model.",
            )
            retrieval_min_score = st.slider(
                "Minimum retrieval relevance",
                min_value=-1.0,
                max_value=1.0,
                value=float(settings.retrieval_min_score),
                step=0.05,
                help="Strong exact-term BM25 matches may still qualify below this semantic score.",
            )
            max_history_turns = st.slider(
                "Conversation turns in prompt",
                min_value=2,
                max_value=10,
                value=settings.max_history_turns,
            )
        with indexing_tab:
            st.caption(
                "Changes affect newly indexed files. Rebuild the index to apply them "
                "to existing documents."
            )
            semantic_threshold = st.slider(
                "Semantic chunking threshold",
                min_value=0.3,
                max_value=0.8,
                value=float(settings.semantic_threshold),
                step=0.05,
            )
            llm_filename = st.text_input(
                "Preferred GGUF filename",
                value=settings.llm_filename,
                help="Leave empty to select a compatible quantized file automatically.",
            )
        with ocr_tab:
            enable_ocr = st.checkbox(
                "Enable OCR fallback for scanned PDFs",
                value=settings.enable_ocr,
                help="Run Tesseract when a PDF page contains insufficient extractable text.",
            )
            ocr_min_text_chars = st.slider(
                "OCR trigger threshold",
                min_value=20,
                max_value=200,
                value=int(settings.ocr_min_text_chars),
                help="OCR pages with fewer extracted characters than this value.",
            )
            ocr_zoom = st.slider(
                "OCR render scale",
                min_value=1.0,
                max_value=3.0,
                value=float(settings.ocr_zoom),
                step=0.25,
            )
            tesseract_cmd = st.text_input(
                "Tesseract executable path",
                value=settings.tesseract_cmd,
                help="Leave empty when Tesseract is available on PATH.",
            )
        saved = st.form_submit_button(
            "Save advanced settings", type="primary", use_container_width=True
        )
    if saved:
        settings.llm_filename = llm_filename.strip()
        settings.top_k = int(top_k)
        settings.retrieval_min_score = float(retrieval_min_score)
        settings.max_history_turns = int(max_history_turns)
        settings.semantic_threshold = float(semantic_threshold)
        settings.enable_ocr = bool(enable_ocr)
        settings.ocr_min_text_chars = int(ocr_min_text_chars)
        settings.ocr_zoom = float(ocr_zoom)
        settings.tesseract_cmd = tesseract_cmd.strip()
        save_settings(settings)
        st.session_state.settings = settings
        if settings.enable_ocr and not _tesseract_available(settings.tesseract_cmd):
            st.session_state.ocr_setup_shown = True
            _ocr_setup_dialog(settings.tesseract_cmd)
            return
        st.toast("Advanced settings saved.", icon=":material/check_circle:")
        st.rerun()


def _tesseract_available(configured_command: str = "") -> bool:
    command = configured_command.strip()
    if not command:
        return shutil.which("tesseract") is not None
    candidate = Path(command).expanduser()
    return (candidate.is_file() and os.access(candidate, os.X_OK)) or shutil.which(command) is not None


@st.dialog("Legacy PowerPoint files are not supported", width="large")
def _legacy_ppt_dialog(paths: list[Path]) -> None:
    st.warning(
        "The legacy `.ppt` binary format cannot be indexed. Open each file in "
        "Microsoft PowerPoint, LibreOffice Impress, or another compatible editor and "
        "save it as a `.pptx` file, then build the index again."
    )
    st.markdown("Files that need conversion:")
    for path in paths:
        st.code(str(path), language=None)
    if st.button("Close", type="primary", use_container_width=True):
        st.rerun()


@st.dialog("Tesseract OCR must be installed", width="large")
def _ocr_setup_dialog(configured_command: str = "") -> None:
    st.warning(
        "OCR is enabled, but the Tesseract system executable could not be found. "
        "The Python package alone is not enough; install Tesseract manually for your operating system."
    )

    windows, macos, linux = st.tabs(["Windows", "macOS", "Linux"])
    with windows:
        st.markdown(
            """
            1. Download and run a Tesseract installer from the
               [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).
            2. Add `C:\\Program Files\\Tesseract-OCR` to the Windows `Path`, then restart this app.
            3. Alternatively, enter the full path to `tesseract.exe` in **Tesseract executable path**.
            """
        )
    with macos:
        st.markdown("Install with Homebrew, then restart this app:")
        st.code("brew install tesseract", language="bash")
    with linux:
        st.markdown("For Ubuntu or Debian, install from the system package manager:")
        st.code("sudo apt update\nsudo apt install tesseract-ocr", language="bash")
        st.caption("For Fedora, Arch, or another distribution, install its `tesseract` package.")

    st.markdown(
        "See the [official Tesseract installation guide]"
        "(https://tesseract-ocr.github.io/tessdoc/Installation.html) for other platforms and languages."
    )
    if configured_command.strip():
        st.caption(f"Configured executable: `{configured_command.strip()}`")

    recheck, close = st.columns(2)
    with recheck:
        if st.button("Check installation again", type="primary", use_container_width=True):
            if _tesseract_available(configured_command):
                st.session_state.ocr_setup_shown = False
                st.toast("Tesseract is available. OCR is ready.", icon=":material/check_circle:")
                st.rerun()
            st.error("Tesseract still could not be found. Restart the app after installing it.")
    with close:
        if st.button("Close", use_container_width=True):
            st.rerun()


def _sources_panel(sources, anchor_prefix: str) -> None:
    if not sources:
        st.caption("No evidence passed the relevance threshold.")
        return
    search = st.session_state.get("evidence_search", "").strip()
    visible_sources = [
        source for source in sources
        if not search or search.casefold() in source.text.casefold()
    ]
    if not visible_sources:
        st.info(f"No retrieved evidence contains `{search}`.")
        return
    links = " · ".join(
        f"[{source.source_id}](#{anchor_prefix}-{source.source_id.lower()})"
        for source in visible_sources
    )
    st.markdown(f"Jump to evidence: {links}")
    for card_index, source in enumerate(visible_sources):
        _source_card(source, anchor_prefix, card_index, search)


def _diagnostics_panel(diagnostics) -> None:
    st.caption(f"Query: {diagnostics.query}")
    scope = ", ".join(Path(path).name for path in diagnostics.scope) or "All indexed documents"
    st.caption(f"Scope: {scope}")
    first, second, third = st.columns(3)
    first.metric("Candidates", diagnostics.candidate_count)
    second.metric("Selected", diagnostics.selected_count)
    third.metric("Documents", diagnostics.documents_covered)
    st.markdown(
        f"Embedding `{diagnostics.embedding_ms:.1f} ms` · "
        f"semantic `{diagnostics.semantic_ms:.1f} ms` · "
        f"BM25 `{diagnostics.lexical_ms:.1f} ms` · "
        f"rerank `{diagnostics.rerank_ms:.1f} ms` · "
        f"total `{diagnostics.total_ms:.1f} ms`"
    )
    rows = [
        {
            "Rank": item.rank,
            "Decision": item.decision,
            "Document": item.document_name,
            "Page": item.approx_page,
            "Semantic": round(item.semantic_score, 3),
            "BM25": round(item.lexical_score, 3),
            "Combined": round(item.combined_score, 3),
            "Excerpt": item.excerpt,
        }
        for item in diagnostics.candidates
    ]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info("No retrieval candidates matched the current scope and threshold.")


def _initialize_conversation_state(store: ConversationStore) -> None:
    if st.session_state.get("conversation_loaded"):
        return
    conversations = store.list()
    conversation = conversations[0] if conversations else store.create()
    _load_conversation(conversation)


def _load_conversation(conversation: dict) -> None:
    st.session_state.active_conversation_id = conversation["id"]
    st.session_state.conversation_name = conversation.get("name", "New conversation")
    st.session_state.messages = list(conversation.get("messages", []))
    sources_by_turn: dict[int, list[SourceRecord]] = {}
    for turn, sources in conversation.get("sources_by_turn", {}).items():
        restored: list[SourceRecord] = []
        for source in sources:
            if isinstance(source, dict):
                restored.append(SourceRecord(**source))
        sources_by_turn[int(turn)] = restored
    st.session_state.sources_by_turn = sources_by_turn
    st.session_state.diagnostics_by_turn = {}
    st.session_state.conversation_loaded = True


def _persist_current_conversation(store: ConversationStore) -> None:
    conversation_id = st.session_state.get("active_conversation_id")
    if not conversation_id:
        return
    serialized_sources = {
        str(turn): [asdict(source) for source in sources]
        for turn, sources in st.session_state.sources_by_turn.items()
    }
    store.save_conversation(
        conversation_id,
        st.session_state.get("conversation_name", "New conversation"),
        list(st.session_state.messages),
        serialized_sources,
    )


def _conversation_panel(store: ConversationStore) -> None:
    st.markdown("#### Conversations")
    conversations = store.list()
    names = {item["id"]: item["name"] for item in conversations}
    ids = list(names)
    active_id = st.session_state.active_conversation_id
    if active_id not in ids:
        conversation = store.create()
        _load_conversation(conversation)
        st.session_state.pending_conversation_id = conversation["id"]
        st.rerun()
    pending_id = st.session_state.pop("pending_conversation_id", None)
    if pending_id in ids:
        # This runs before the selectbox is instantiated, which is the only safe
        # point at which Streamlit permits programmatic widget-state changes.
        st.session_state.conversation_selector = pending_id
    selected = st.selectbox(
        "Conversation",
        ids,
        index=ids.index(active_id),
        format_func=lambda item: names[item],
        key="conversation_selector",
    )
    if selected != active_id:
        conversation = store.get(selected)
        if conversation:
            _load_conversation(conversation)
            st.rerun()

    new_column, delete_column = st.columns(2)
    if new_column.button("New", use_container_width=True):
        _persist_current_conversation(store)
        conversation = store.create()
        _load_conversation(conversation)
        st.session_state.pending_conversation_id = conversation["id"]
        st.rerun()
    if delete_column.button("Delete", use_container_width=True):
        store.delete(active_id)
        remaining = store.list()
        conversation = remaining[0] if remaining else store.create()
        _load_conversation(conversation)
        st.session_state.pending_conversation_id = conversation["id"]
        st.rerun()

    rename = st.text_input(
        "Conversation name",
        value=st.session_state.conversation_name,
        key=f"conversation-name-{active_id}",
    )
    if st.button("Rename", use_container_width=True):
        store.rename(active_id, rename)
        st.session_state.conversation_name = rename.strip() or "Untitled conversation"
        st.rerun()

    markdown = conversation_markdown(
        st.session_state.conversation_name,
        st.session_state.messages,
        st.session_state.sources_by_turn,
    )
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", st.session_state.conversation_name).strip("-")
    safe_name = safe_name or "conversation"
    export_markdown, export_pdf = st.columns(2)
    export_markdown.download_button(
        "Markdown",
        markdown,
        file_name=f"{safe_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    export_pdf.download_button(
        "PDF",
        conversation_pdf(
            st.session_state.conversation_name,
            st.session_state.messages,
            st.session_state.sources_by_turn,
        ),
        file_name=f"{safe_name}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _source_card(
    source, anchor_prefix: str = "source", card_index: int = 0, search: str = ""
) -> None:
    source_id = html.escape(str(source.source_id))
    document_name = html.escape(str(source.document_name))
    source_path = html.escape(str(source.source_path))
    source_text = _highlight_text(
        str(source.text), search or getattr(source, "matched_passage", "")
    )
    anchor = html.escape(f"{anchor_prefix}-{str(source.source_id).lower()}", quote=True)
    st.markdown(
        f"""
        <div class="source-card" id="{anchor}">
            <strong>[{source_id}] {document_name}</strong><br/>
            <span class="muted">{source_path}</span><br/>
            <span class="accent">Page {source.approx_page} • hybrid {source.score:.3f} • semantic {source.semantic_score:.3f} • BM25 {source.lexical_score:.3f}</span>
            <p style="margin:0.6rem 0 0 0;">{source_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_key = hashlib.sha256(
        f"{anchor_prefix}:{card_index}:{source.source_path}".encode("utf-8")
    ).hexdigest()[:16]
    open_column, pin_column, exclude_column = st.columns(3)
    if Path(source.source_path).suffix.lower() == ".pdf":
        target = Path(source.source_path).resolve().as_uri() + f"#page={source.approx_page}"
        open_column.markdown(f"[Open PDF at page {source.approx_page}]({target})")
    else:
        open_column.markdown(f"[Open source file]({Path(source.source_path).resolve().as_uri()})")
    if pin_column.button("Pin next", key=f"pin-{action_key}", use_container_width=True):
        pinned = set(st.session_state.pinned_source_paths)
        excluded = set(st.session_state.excluded_source_paths)
        pinned.add(source.source_path)
        excluded.discard(source.source_path)
        st.session_state.pinned_source_paths = sorted(pinned)
        st.session_state.excluded_source_paths = sorted(excluded)
        st.rerun()
    if exclude_column.button(
        "Exclude next", key=f"exclude-{action_key}", use_container_width=True
    ):
        pinned = set(st.session_state.pinned_source_paths)
        excluded = set(st.session_state.excluded_source_paths)
        excluded.add(source.source_path)
        pinned.discard(source.source_path)
        st.session_state.pinned_source_paths = sorted(pinned)
        st.session_state.excluded_source_paths = sorted(excluded)
        st.rerun()


def _highlight_text(text: str, search: str) -> str:
    if not search:
        return html.escape(text)
    parts = re.split(f"({re.escape(search)})", text, flags=re.IGNORECASE)
    return "".join(
        f"<mark>{html.escape(part)}</mark>" if part.casefold() == search.casefold()
        else html.escape(part)
        for part in parts
    )


def _citation_links(answer: str, turn_index: int) -> str:
    return re.sub(
        r"\[(S\d+)\]",
        lambda match: f"[[{match.group(1)}]](#turn-{turn_index}-{match.group(1).lower()})",
        answer,
    )


def _repo_options(defaults: list[str], current_value: str) -> list[str]:
    if current_value in defaults:
        return defaults
    return [current_value, *defaults]


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_model_catalog() -> tuple[list[str], list[str]]:
    return discover_model_repositories(limit=5)


def _format_download_size(downloaded: int, total: int | None) -> str:
    def readable(value: int) -> str:
        size = float(value)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    if total and total > 0:
        return f"{readable(downloaded)} / {readable(total)} ({downloaded / total:.0%})"
    return f"{readable(downloaded)} downloaded"


def _download_models_with_updates(service: RagService, update) -> tuple[str, str]:
    """Download off-thread while applying Streamlit updates on the script thread."""
    events: queue.Queue[tuple[str, str, int, int | None]] = queue.Queue()
    result: list[tuple[str, str]] = []
    failure: list[Exception] = []

    def collect(phase: str, description: str, downloaded: int, total: int | None) -> None:
        events.put((phase, description, downloaded, total))

    def download() -> None:
        try:
            result.append(service.download_models(progress=collect))
        except Exception as exc:  # passed back to the main Streamlit thread
            failure.append(exc)

    worker = threading.Thread(target=download, name="model-download", daemon=True)
    worker.start()
    while worker.is_alive() or not events.empty():
        try:
            update(*events.get(timeout=0.1))
        except queue.Empty:
            time.sleep(0.05)
    worker.join()
    if failure:
        raise failure[0]
    if not result:
        raise RuntimeError("Model download ended without a result.")
    return result[0]


@st.cache_resource(show_spinner=False)
def _cached_service(settings_json: str, cache_version: str) -> RagService:
    """Keep native handles stable while invalidating incompatible old services."""
    _ = cache_version
    return RagService(AppSettings(**json.loads(settings_json)))


@st.cache_resource(show_spinner=False)
def _cached_conversation_store() -> ConversationStore:
    return ConversationStore()


if __name__ == "__main__":
    run()
