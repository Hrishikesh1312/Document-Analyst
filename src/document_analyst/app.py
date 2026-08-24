from __future__ import annotations

import html
import json
import os
import queue
import shutil
import threading
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from document_analyst.config import (
    AppSettings,
    EMBEDDING_REPO_OPTIONS,
    LLM_REPO_OPTIONS,
    load_settings,
    save_settings,
)
from document_analyst.services.rag import RagService
from document_analyst.services.model_manager import discover_model_repositories

SERVICE_CACHE_VERSION = "hybrid-retrieval-v2"


def run() -> None:
    st.set_page_config(
        page_title="Document Analyst",
        page_icon=":material/search:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources_by_turn" not in st.session_state:
        st.session_state.sources_by_turn = {}
    if "diagnostics_by_turn" not in st.session_state:
        st.session_state.diagnostics_by_turn = {}
    if "show_retrieval_diagnostics" not in st.session_state:
        st.session_state.show_retrieval_diagnostics = False
    if "active_view" not in st.session_state:
        st.session_state.active_view = "Chat"
    if "show_empty_state_dialog" not in st.session_state:
        st.session_state.show_empty_state_dialog = True

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

    active_view = _sidebar(settings, service)
    _inject_theme()
    _hero(service)

    if (
        stats["documents"] == 0
        and stats["chunks"] == 0
        and st.session_state.show_empty_state_dialog
    ):
        _empty_state_dialog()

    if active_view == "Chat":
        _chat_tab(service)
    elif active_view == "Manage Documents":
        _docs_tab(settings, service)
    else:
        _models_tab(settings, service)


def _sidebar(settings: AppSettings, service: RagService) -> str:
    with st.sidebar:
        st.title("Document Analyst")
        active_view = st.radio(
            "Navigate",
            ["Chat", "Manage Documents", "Models & Settings"],
            index=["Chat", "Manage Documents", "Models & Settings"].index(st.session_state.active_view),
            label_visibility="visible",
        )
        st.session_state.active_view = active_view
        stats = service.stats()
        col1, col2 = st.columns(2)
        col1.metric("Docs", stats["documents"])
        col2.metric("Chunks", stats["chunks"])
        st.caption("Ask questions, explore documents, and turn files into grounded answers.")
        with st.container(key="sidebar_reset"):
            if st.button(
                "Reset",
                icon=":material/restart_alt:",
                use_container_width=True,
                type="secondary",
            ):
                _reset_dialog(service)
    return active_view


@st.dialog("Reset application data")
def _reset_dialog(service: RagService) -> None:
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
                    cleared.append("vector database")
                if clear_chat:
                    st.session_state.messages = []
                    st.session_state.sources_by_turn = {}
                    st.session_state.diagnostics_by_turn = {}
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


def _inject_theme() -> None:
    palette = {
        "bg": "#02060d",
        "panel": "#08111d",
        "card": "#0d1828",
        "border": "#1d3147",
        "text": "#edf4fb",
        "muted": "#89a0b9",
        "accent": "#81f0bc",
    }
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(129, 240, 188, 0.08) 0%, transparent 26%),
                radial-gradient(circle at 10% 20%, rgba(53, 112, 214, 0.10) 0%, transparent 22%),
                linear-gradient(180deg, {palette["bg"]} 0%, {palette["bg"]} 100%);
            color: {palette["text"]};
        }}
        .hero-card, .source-card, .metric-card {{
            background: {palette["panel"]};
            border: 1px solid {palette["border"]};
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 18px 40px rgba(0,0,0,0.22);
        }}
        .source-card {{
            background: {palette["card"]};
            margin-bottom: 0.8rem;
        }}
        [data-testid="stSidebar"] {{
            background: #050b14;
            border-right: 1px solid {palette["border"]};
        }}
        [data-testid="stSidebarUserContent"] {{
            min-height: 100%;
            display: flex;
            flex-direction: column;
        }}
        [data-testid="stSidebarUserContent"] > div {{
            display: flex;
            flex-direction: column;
            flex: 1;
        }}
        [data-testid="stSidebar"] .st-key-sidebar_reset {{
            margin-top: auto;
            padding-top: 1rem;
        }}
        [data-testid="stChatMessage"] {{
            background: rgba(8, 17, 29, 0.72);
            border: 1px solid rgba(29, 49, 71, 0.85);
            border-radius: 18px;
            padding: 0.4rem 0.8rem;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] > label {{
            background: #07101b;
            border: 1px solid {palette["border"]};
            border-radius: 14px;
            padding: 0.35rem 0.65rem;
            margin-bottom: 0.45rem;
            width: 100%;
            min-height: 48px;
            box-sizing: border-box;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] > label[data-baseweb="radio"] {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }}
        [data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
            border-color: {palette["accent"]};
            box-shadow: 0 0 0 1px rgba(129, 240, 188, 0.15);
            background: rgba(16, 34, 53, 0.96);
        }}
        [data-testid="stSidebar"] [role="radiogroup"] p {{
            color: {palette["text"]};
            font-weight: 600;
        }}
        .muted {{
            color: {palette["muted"]};
        }}
        .accent {{
            color: {palette["accent"]};
        }}
        code {{
            color: {palette["accent"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero(service: RagService) -> None:
    stats = service.stats()
    st.markdown(
        f"""
        <div class="hero-card">
            <h1 style="margin:0 0 0.4rem 0;">Document Analyst</h1>
            <p class="muted" style="margin:0 0 1rem 0;">
                Privacy-first semantic search and local Q&A for PDF, DOCX, PPTX, Markdown, and text files.
                Multi-turn chat, source grounding, Chroma persistence, and on-device GGUF inference.
            </p>
            <p style="margin:0;">
                <span class="accent"><strong>{stats["documents"]}</strong> documents</span>
                indexed across
                <span class="accent"><strong>{stats["chunks"]}</strong> chunks</span>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


@st.dialog("No Documents Yet", width="large")
def _empty_state_dialog() -> None:
    st.markdown(
        """
        Your local index is empty right now.

        To get started:
        1. Open `Models & Settings` and configure or download your embedding / GGUF models.
        2. Open `Manage Documents` and add the local document folder you want to index.
        3. Build the index, then come back to `Chat` to start asking questions.
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Open Models & Settings", use_container_width=True):
            st.session_state.active_view = "Models & Settings"
            st.session_state.show_empty_state_dialog = False
            st.rerun()
    with col2:
        if st.button("Open Manage Documents", use_container_width=True):
            st.session_state.active_view = "Manage Documents"
            st.session_state.show_empty_state_dialog = False
            st.rerun()
    with col3:
        if st.button("Dismiss", use_container_width=True):
            st.session_state.show_empty_state_dialog = False
            st.rerun()


def _chat_tab(service: RagService) -> None:
    indexed_documents = service.indexed_documents()
    document_labels = {
        str(item["source_path"]): str(item["document_name"]) for item in indexed_documents
    }
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        for index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
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
            st.rerun()

    with right:
        st.markdown("#### Retrieval tools")
        st.toggle(
            "Show retrieval diagnostics",
            key="show_retrieval_diagnostics",
            help="Inspect hybrid-search candidates, score components, document coverage, and timing.",
        )
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
            submitted = st.form_submit_button("Build Index", use_container_width=True)

        if submitted:
            settings.documents_dir = directory.strip()
            save_settings(settings)
            st.session_state.settings = settings
            index_progress = st.progress(0.0, text="Preparing documents for indexing…")

            def update_index_progress(
                phase: str, item: str, current: int, total: int, complete: bool
            ) -> None:
                completed = current if complete else current - 1
                item_fraction = completed / total if total else 0.0
                ranges = {
                    "reading": (0.0, 0.2),
                    "chunking": (0.2, 0.55),
                    "embedding": (0.55, 0.9),
                    "writing": (0.9, 1.0),
                }
                start, end = ranges[phase]
                fraction = start + ((end - start) * item_fraction)
                labels = {
                    "reading": "Reading",
                    "chunking": "Chunking",
                    "embedding": "Embedding",
                    "writing": "Writing",
                }
                position = f" {current}/{total}" if total > 1 else ""
                index_progress.progress(
                    fraction,
                    text=f"{labels[phase]}{position}: {item}",
                )

            try:
                with st.spinner("Parsing files, chunking semantically, embedding, and writing to Chroma..."):
                    result = service.index_documents(
                        settings.documents_dir,
                        replace_existing=replace_existing,
                        progress=update_index_progress,
                    )
            except (OSError, RuntimeError, ValueError) as exc:
                st.error(f"Indexing failed: {exc}")
                return
            if result.document_count:
                index_progress.progress(1.0, text="Indexing complete")
            else:
                index_progress.progress(0.0, text="No readable documents found to index")
            st.success(f"Indexed {result.document_count} documents into {result.chunk_count} chunks.")
            for warning in result.warnings:
                st.warning(warning)
            st.rerun()

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
    docs = service.indexed_documents()
    if not docs:
        st.info("No indexed documents yet.")
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
                    <span class="accent">{item["chunks"]} chunks</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Remove from index: {item['document_name']}", key=f"del-{item['source_path']}"):
                service.delete_document(str(item["source_path"]))
                st.success(f"Removed {item['document_name']} from the index.")
                st.rerun()


def _models_tab(settings: AppSettings, service: RagService) -> None:
    st.subheader("Model Downloads")
    local_llm = service.models.local_llm_model_path()
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
    st.markdown(
        f"""
        - Embedding model directory: `{settings.embeddings_dir}`
        - Embedding repo: `{settings.embeddings_repo}`
        - LLM repo: `{settings.llm_repo}`
        - Local GGUF: `{local_llm or 'Not downloaded yet'}`
        """
    )
    if st.button("Download Selected Models", use_container_width=True):
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
        st.success(f"Models ready.\nEmbedding model: {embedding_dir}\nLLM: {llm_path}")

    st.write("")
    st.subheader("Model Selection")
    if catalog_warning:
        st.info("Hugging Face is unavailable, so the built-in model catalog is being shown.")
    else:
        st.caption("This catalog is refreshed from public Hugging Face repositories every hour.")
    with st.form("settings-form"):
        embeddings_repo = st.selectbox(
            "Embedding model",
            options=embedding_options,
            index=embedding_options.index(settings.embeddings_repo),
            help="Choose which embedding model repo to download and use locally.",
        )
        llm_repo = st.selectbox(
            "LLM GGUF model",
            options=llm_options,
            index=llm_options.index(settings.llm_repo),
            help="Choose which local instruction model repo to download and run with llama.cpp.",
        )
        llm_filename = st.text_input("Preferred GGUF filename", value=settings.llm_filename)
        top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=settings.top_k)
        retrieval_min_score = st.slider(
            "Minimum retrieval relevance",
            min_value=-1.0,
            max_value=1.0,
            value=float(settings.retrieval_min_score),
            step=0.05,
            help="Minimum semantic similarity. Strong exact-term BM25 matches may still qualify.",
        )
        max_history_turns = st.slider("Turns kept in prompt", min_value=2, max_value=10, value=settings.max_history_turns)
        semantic_threshold = st.slider(
            "Semantic chunking threshold",
            min_value=0.3,
            max_value=0.8,
            value=float(settings.semantic_threshold),
            step=0.05,
        )
        enable_ocr = st.checkbox(
            "Enable OCR fallback for scanned PDFs",
            value=settings.enable_ocr,
            help="If a PDF page has very little extractable text, render it and run Tesseract OCR.",
        )
        ocr_min_text_chars = st.slider(
            "OCR trigger threshold (characters)",
            min_value=20,
            max_value=200,
            value=int(settings.ocr_min_text_chars),
            help="Run OCR on a PDF page when extracted text is shorter than this threshold.",
        )
        ocr_zoom = st.slider(
            "OCR render scale",
            min_value=1.0,
            max_value=3.0,
            value=float(settings.ocr_zoom),
            step=0.25,
            help="Higher render scales can improve OCR quality, but are slower.",
        )
        tesseract_cmd = st.text_input(
            "Tesseract executable path (optional)",
            value=settings.tesseract_cmd,
            help="Set this only if Tesseract is installed outside your normal PATH.",
        )
        saved = st.form_submit_button("Save Settings", use_container_width=True)
    if saved:
        repo_changed = embeddings_repo.strip() != settings.embeddings_repo or llm_repo.strip() != settings.llm_repo
        settings.embeddings_repo = embeddings_repo.strip()
        settings.llm_repo = llm_repo.strip()
        settings.llm_filename = "" if repo_changed and not llm_filename.strip() else llm_filename.strip()
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
        st.success("Settings saved.")
        st.rerun()


def _tesseract_available(configured_command: str = "") -> bool:
    command = configured_command.strip()
    if not command:
        return shutil.which("tesseract") is not None
    candidate = Path(command).expanduser()
    return (candidate.is_file() and os.access(candidate, os.X_OK)) or shutil.which(command) is not None


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
    links = " · ".join(
        f"[{source.source_id}](#{anchor_prefix}-{source.source_id.lower()})" for source in sources
    )
    st.markdown(f"Jump to evidence: {links}")
    for source in sources:
        _source_card(source, anchor_prefix)


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


def _source_card(source, anchor_prefix: str = "source") -> None:
    source_id = html.escape(str(source.source_id))
    document_name = html.escape(str(source.document_name))
    source_path = html.escape(str(source.source_path))
    source_text = html.escape(str(source.text))
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


if __name__ == "__main__":
    run()
