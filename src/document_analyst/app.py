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
from dataclasses import asdict, dataclass, field, replace
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
from document_analyst.services.index_progress import IndexProgressEvent, IndexProgressTracker
from document_analyst.services.indexing_performance import apply_indexing_preset, benchmark_embedding_batches
from document_analyst.services.library_search import filter_library_documents
from document_analyst.services.model_manager import discover_model_repositories
from document_analyst.ui.components import (
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

SERVICE_CACHE_VERSION = "automatic-embedding-device-v4"


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
    progress: IndexProgressTracker = field(default_factory=IndexProgressTracker)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    completion_refreshed: bool = False


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
    page_heading(active_view)
    _workspace_status(service, stats)

    if (
        stats["documents"] == 0
        and stats["chunks"] == 0
        and st.session_state.show_empty_state_dialog
    ):
        _empty_state_dialog(settings, service)

    if active_view == "Ask":
        _chat_tab(service, conversation_store)
    elif active_view == "Library":
        _docs_tab(settings, service)
    elif active_view == "Settings":
        _models_tab(settings, service)
    elif active_view == "Help":
        _help_tab()


def _select_workspace_view() -> None:
    st.session_state.active_view = st.session_state.workspace_navigation


def _workspace_status(service: RagService, stats: dict) -> None:
    embedding_ready = service.models.local_embedding_model_exists()
    llm_ready = bool(service.models.local_llm_model_path())
    models_ready = embedding_ready and llm_ready
    document_count = int(stats.get("documents", 0))
    chunk_count = int(stats.get("chunks", 0))

    if models_ready and document_count:
        st.markdown(
            f"""
            <div class="da-workspace-status ready compact">
                <span class="da-status-dot"></span>
                <strong>Ready</strong>
                <span>{document_count:,} document{'s' if document_count != 1 else ''} · {chunk_count:,} passages</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    elif not models_ready:
        missing = []
        if not embedding_ready:
            missing.append("search model")
        if not llm_ready:
            missing.append("answer model")
        title = "Setup needs attention"
        detail = f"Install the {' and '.join(missing)} to use your local workspace."
        state_class = "attention"
        action_label = "Open settings"
        target_view = "Settings"
    else:
        title = "Models ready"
        detail = "Add a document folder to create your searchable library."
        state_class = "attention"
        action_label = "Add documents"
        target_view = "Library"

    status_markup = f"""
        <div class="da-workspace-status {state_class} compact">
            <span class="da-status-dot"></span>
            <strong>{html.escape(title)}</strong>
            <span>{html.escape(detail)}</span>
        </div>
    """
    if st.session_state.get("active_view") == target_view:
        st.markdown(status_markup, unsafe_allow_html=True)
        return

    status_column, action_column = st.columns([6, 1], vertical_alignment="center")
    with status_column:
        st.markdown(status_markup, unsafe_allow_html=True)
    with action_column:
        if st.button(
            action_label,
            key=f"workspace-status-{target_view.lower()}",
            use_container_width=True,
        ):
            st.session_state.active_view = target_view
            st.rerun()


def _sidebar(
    settings: AppSettings, service: RagService, conversation_store: ConversationStore
) -> str:
    with st.sidebar:
        sidebar_brand()
        views = ["Ask", "Library", "Settings", "Help"]
        view_labels = {
            "Ask": "⌕  Ask",
            "Library": "▤  Library",
            "Settings": "⚙  Settings",
            "Help": "?  Help",
        }
        # Keep programmatic navigation in sync without changing widget identity.
        # User selections update active_view in the callback before the next run.
        st.session_state.workspace_navigation = st.session_state.active_view
        active_view = st.radio(
            "Workspace navigation",
            views,
            key="workspace_navigation",
            on_change=_select_workspace_view,
            format_func=lambda view: view_labels[view],
            label_visibility="collapsed",
        )
        stats = service.stats()
        sidebar_stats(stats["documents"], stats["chunks"])
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


@st.dialog("Welcome to Document Analyst", width="large")
def _empty_state_dialog(settings: AppSettings, service: RagService) -> None:
    st.markdown("Set up your local assistant, then add documents to start asking questions.")
    if settings.chunking_mode == "semantic":
        st.info(
            "Semantic chunking is currently enabled. It groups passages by topic and can take "
            "longer for large documents or many files. For faster indexing, open **Settings**, "
            "enable **Advanced settings**, then choose **Indexing → Chunking method → "
            "Fast — sentence boundaries** and save."
        )
    else:
        st.caption("Fast chunking is enabled for quicker document indexing.")
    ready = service.models.local_embedding_model_exists() and bool(service.models.local_llm_model_path())
    if not ready:
        st.markdown(
            "**Get started with recommended settings**\n\n"
            "We’ll download two models: one to search your documents and one to write answers. "
            "This one-time setup needs internet and may take a few minutes. "
            "Your documents and answers stay on your computer."
        )
        defaults = AppSettings()
        st.caption(
            f"Search: {model_display_name(defaults.embeddings_repo)} · "
            f"Answers: {model_display_name(defaults.llm_repo)}"
        )
        if st.button("Proceed with recommended settings", type="primary", use_container_width=True):
            recommended = replace(settings, embeddings_repo=defaults.embeddings_repo,
                                  llm_repo=defaults.llm_repo, llm_filename="")
            try:
                save_settings(recommended)
                st.session_state.settings = recommended
                recommended_service = _cached_service(
                    json.dumps(asdict(recommended), sort_keys=True), SERVICE_CACHE_VERSION
                )
            except Exception as exc:
                st.error(f"Could not prepare setup: {exc}")
                return
            if not _download_models_in_ui(recommended_service):
                st.info("Click Proceed again to retry. Downloaded files will be reused.")
                return
            st.rerun()
        if st.button("Customize settings", use_container_width=True):
            st.session_state.active_view = "Settings"
            st.session_state.show_empty_state_dialog = False
            st.rerun()
    else:
        st.success("Your models are ready. Next, add your documents.")
        st.markdown(
            "Choose a folder in **Library**, then click **Build index**. "
            "Once indexing finishes, open **Ask** to chat with your documents."
        )
        if st.button("Add documents", type="primary", use_container_width=True):
            st.session_state.active_view = "Library"
            st.session_state.show_empty_state_dialog = False
            st.rerun()
    if st.button("Set up later"):
        st.session_state.show_empty_state_dialog = False
        st.rerun()


def _chat_tab(service: RagService, conversation_store: ConversationStore) -> None:
    indexed_documents = service.indexed_documents()
    document_labels = {
        str(item["source_path"]): str(item["document_name"]) for item in indexed_documents
    }
    conversation_column, scope_column, tools_column, _ = st.columns(
        [1.15, 1, .8, 3.2], gap="small"
    )
    with conversation_column.popover(
        "Chats", icon=":material/chat_bubble:", use_container_width=True
    ):
        _conversation_panel(conversation_store)
    selected_scope = st.session_state.get("chat_document_filter", [])
    scope_button_label = f"{len(selected_scope)} documents" if selected_scope else "All documents"
    with scope_column.popover(
        scope_button_label, icon=":material/filter_alt:", use_container_width=True
    ):
        st.markdown("#### Search in")
        selected = st.session_state.get("chat_document_filter", [])
        if any(path not in document_labels for path in selected):
            st.session_state.chat_document_filter = [path for path in selected if path in document_labels]
        st.multiselect(
            "Documents", options=list(document_labels),
            format_func=lambda path: document_labels.get(path, Path(path).name),
            placeholder="All indexed documents", key="chat_document_filter",
            help="Leave empty to search the entire library.",
        )
    with tools_column.popover(
        "Tools", icon=":material/tune:", use_container_width=True
    ):
        st.text_input("Search retrieved evidence", key="evidence_search", placeholder="Find text in source passages")
        st.toggle("Show retrieval diagnostics", key="show_retrieval_diagnostics")

    queued_prompt = st.session_state.pop("pending_chat_prompt", None)
    regenerate_prompt = st.session_state.pop("pending_regenerate_prompt", None)
    if not st.session_state.messages:
        if document_labels:
            empty_state("What would you like to understand?",
                        "Ask a question, compare ideas, or find a passage in your documents.", "⌕")
            starter_prompts = _starter_prompts(
                document_labels,
                st.session_state.get("chat_document_filter", []),
            )
            first_row = st.columns(2, gap="small")
            second_row = st.columns(2, gap="small")
            for index, (column, (label, prompt_text)) in enumerate(
                zip(first_row + second_row, starter_prompts)
            ):
                if column.button(label, key=f"starter-{index}", use_container_width=True):
                    queued_prompt = prompt_text
        else:
            empty_state("Add your first documents", "Build a library, then ask questions with sources you can check.", "▤")
            if st.button("Go to Library", type="primary"):
                st.session_state.active_view = "Library"
                st.rerun()
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(_citation_links(message["content"], index))
            else:
                st.markdown(message["content"])
            if message["role"] == "assistant":
                sources = st.session_state.sources_by_turn.get(index, [])
                _answer_actions(
                    message["content"],
                    index,
                    sources,
                    conversation_store,
                    is_latest=index == len(st.session_state.messages) - 1,
                )
                if sources:
                    with st.expander(f"Sources ({len(sources)})", expanded=False):
                        _sources_panel(sources, f"turn-{index}")
                diagnostics = st.session_state.diagnostics_by_turn.get(index)
                if diagnostics and st.session_state.show_retrieval_diagnostics:
                    with st.expander("Retrieval diagnostics", expanded=False):
                        _diagnostics_panel(diagnostics)

    prompt = queued_prompt or regenerate_prompt or st.chat_input(
        "Ask a question about your documents…", disabled=not bool(document_labels)
    )
    if prompt:
        if not regenerate_prompt:
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
                if not regenerate_prompt:
                    st.session_state.messages.pop()
                return
            turn_index = len(st.session_state.messages)
            with st.expander(f"Sources ({len(result.sources)})", expanded=True):
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
        st.rerun()


def _starter_prompts(
    document_labels: dict[str, str], selected_paths: list[str]
) -> list[tuple[str, str]]:
    paths = [path for path in selected_paths if path in document_labels]
    if not paths:
        paths = list(document_labels)
    names = [document_labels[path] for path in paths]
    if not names:
        return []

    first = names[0]
    first_label = _short_document_label(first)
    if len(names) == 1:
        return [
            (
                f"Summarise {first_label}",
                f"Summarise the main themes from {first}.",
            ),
            (
                f"Key findings in {first_label}",
                f"What are the key findings and conclusions in {first}?",
            ),
            (
                f"Important passages in {first_label}",
                f"Find the most important passages in {first} and explain why they matter.",
            ),
            (
                f"Brief me on {first_label}",
                f"Create a concise briefing on {first} with supporting sources.",
            ),
        ]

    second = names[1]
    second_label = _short_document_label(second)
    collection_label = (
        "both documents" if len(names) == 2 else f"all {len(names)} documents"
    )
    return [
        (
            f"Summarise {first_label}",
            f"Summarise the main themes from {first}.",
        ),
        (
            f"Summarise {second_label}",
            f"Summarise the main themes from {second}.",
        ),
        (
            f"Compare {first_label} and {second_label}",
            f"Compare the key ideas, findings, and points of disagreement in {first} and {second}.",
        ),
        (
            f"Brief me on {collection_label}",
            f"Create a concise briefing across {collection_label} with the main findings and supporting sources.",
        ),
    ]


def _short_document_label(name: str, limit: int = 30) -> str:
    if len(name) <= limit:
        return f"“{name}”"
    suffix = Path(name).suffix
    available = max(8, limit - len(suffix) - 1)
    return f"“{name[:available]}…{suffix}”"

def _help_tab() -> None:
    with st.expander("Getting started", expanded=True):
        st.markdown(
            "1. In the welcome popup, choose **Proceed with recommended settings** to download "
            "the search and answer models. You can also download or change models in **Settings**.\n"
            "2. In **Library**, open **Add a document folder**, choose your folder, and click **Build Index**.\n"
            "3. Open **Ask** and ask a question about your indexed documents.\n\n"
            "In **Library**, search by document name or folder path and filter by indexing status "
            "or file type. Use **Clear filters** to see all documents. Results are paginated. "
            "Click **View passages** on a document to inspect its extracted text and page references."
        )
    with st.expander("Conversations and downloads"):
        st.markdown(
            "Open the **Conversations** menu in Ask to reopen a saved chat. "
            "You can create, rename, or delete conversations there.\n\n"
            "Under **Download conversation**, choose **Download Markdown (.md)** for an editable "
            "text file or **Download PDF (.pdf)** for a document you can print or share. "
            "Downloads include the conversation and its source references."
        )
    with st.expander("Answers, sources, and search controls"):
        st.markdown(
            "Your question is embedded locally. Semantic search and keyword search (BM25) find "
            "candidate passages, which are reranked and selected across documents. The local "
            "answer model uses these passages and cites them as **[S1]**, **[S2]**, and so on.\n\n"
            "Open **Sources** below an answer to check its evidence. Use **Search retrieved evidence** "
            "to find text within those passages.\n\n"
            "Under **Search scope**, select documents to limit your search; leave it empty to search "
            "the full index. Open **Search tools** and enable **Show retrieval diagnostics** to inspect search scores and timings. "
            "The relevance threshold is in Settings; strong keyword matches can still qualify below it.\n\n"
            "Only the latest conversation window is sent to the answer model to limit processing time. "
            "Change its length in advanced settings."
        )
    with st.expander("Indexing speed and troubleshooting"):
        st.markdown(
            "Embedding batches automatically use CUDA when available, otherwise Apple MPS, "
            "and CPU when neither is available. Settings shows the selected device after first use. "
            "This accelerates sentence chunking, document embeddings, and search-query embeddings.\n\n"
            "Large documents can take time to process. Progress shows PDF pages and embedding batches; "
            "the first embedding batch also loads the search model. Live elapsed time and device are shown "
            "during indexing. After several batches, the panel shows throughput and an estimate for the "
            "current stage only. Overall progress is approximate because stages and file sizes vary. "
            "Cancellation takes effect at a safe checkpoint.\n\n"
            "In **Library → Performance options**, choose **Fast**, **Balanced**, or **Quality** "
            "and click **Apply performance settings**. Fast uses larger sentence-based passages; "
            "Balanced uses smaller sentence-based passages; Quality uses semantic chunking. "
            "Batch size controls how many passages the model processes together. "
            "**Benchmark batch sizes** compares sample-text speed without changing your settings. "
            "Choose the fastest measured size above to use it. Larger batches need more memory; "
            "if indexing runs out of memory, reduce the batch size and retry failed files.\n\n"
            "For faster chunking, open **Settings → Advanced settings → Indexing** and choose "
            "**Fast — sentence boundaries**. It skips sentence embeddings during chunking; final chunks "
            "still receive embeddings for semantic search. Semantic mode uses topic-aware boundaries.\n\n"
            "Changes apply to newly indexed files. To apply them to existing files, select "
            "**Replace existing index contents** in Library and rebuild. This replaces the entire "
            "current index with the selected folder. Unchanged files are otherwise skipped.\n\n"
            "If a file fails, check its status in Library and use **Retry failed files**. "
            "Scanned PDFs may need OCR enabled in Settings and Tesseract installed. "
            "Convert legacy PowerPoint (.ppt) files to .pptx before indexing."
        )
    with st.expander("Models and privacy"):
        st.markdown(
            "The search model finds relevant passages; the answer model writes responses. "
            "Model downloads and the public model catalog need internet access. "
            "Existing downloaded files are reused. Catalog options refresh hourly, with built-in "
            "options available when the catalog cannot be reached.\n\n"
            "Your documents, index, conversations, and inference stay on this device. "
            "Use **Reset** in the sidebar to choose which local app data to delete."
        )


def _docs_tab(settings: AppSettings, service: RagService) -> None:
    docs = service.document_statuses()
    active_job = st.session_state.get("index_job")
    indexing = bool(active_job and active_job.thread and active_job.thread.is_alive())
    add_column, performance_column = st.columns(2, gap="medium")
    with add_column.expander("Add a document folder", expanded=not bool(docs)):
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

    with performance_column.expander("Performance options", expanded=False):
        _indexing_performance_controls(settings, service, indexing)
    if st.session_state.index_job:
        _index_job_panel(service)

    st.subheader("Your documents")
    if not docs:
        empty_state(
            "Your library is empty",
            "Choose a local folder above to build your private, searchable knowledge base.",
            "▤",
        )
        return

    _library_documents_panel(service, docs, indexing)


def _clear_library_filters() -> None:
    st.session_state.library_query = ""
    st.session_state.library_status_filter = []
    st.session_state.library_type_filter = []


def _library_documents_panel(service: RagService, documents: list[dict], indexing: bool = False) -> None:
    query = st.text_input("Search library", key="library_query", placeholder="Document name or folder path")
    status_options = sorted({str(item.get("status", "indexed")) for item in documents})
    type_options = sorted({Path(str(item["document_name"])).suffix.lower() for item in documents} - {""})
    for key, options in (("library_status_filter", status_options), ("library_type_filter", type_options)):
        if key in st.session_state and any(value not in options for value in st.session_state[key]):
            st.session_state[key] = [value for value in st.session_state[key] if value in options]
    status_column, type_column = st.columns(2)
    statuses = status_column.multiselect("Indexing status", status_options, key="library_status_filter")
    file_types = type_column.multiselect("File type", type_options, key="library_type_filter")
    docs = filter_library_documents(documents, query, statuses, file_types)
    st.caption(f"Showing {len(docs)} of {len(documents)} documents")
    st.button("Clear filters", on_click=_clear_library_filters)
    if not docs:
        empty_state("No matching documents", "Try another name or clear the filters.")
        return
    page_size = 10
    pages = (len(docs) + page_size - 1) // page_size
    filter_id = hashlib.sha256(repr((query, statuses, file_types, len(docs))).encode()).hexdigest()[:12]
    page = int(st.number_input("Library page", 1, pages, 1, key=f"library-page-{filter_id}")) if pages > 1 else 1
    for item in docs[(page - 1) * page_size:page * page_size]:
        document_name = html.escape(str(item["document_name"]))
        status = str(item.get("status", "indexed"))
        status_class = "ready" if status == "indexed" else "attention" if status in {"failed", "cancelled"} else ""
        with st.container(border=True):
            name_column, view_column, manage_column = st.columns([4, 1.5, 1], vertical_alignment="center")
            with name_column:
                st.markdown(
                    f'<div class="da-document-title">{document_name} '
                    f'<span class="da-badge {status_class}">{html.escape(status.title())}</span></div>',
                    unsafe_allow_html=True,
                )
                indexed = str(item.get("indexed_at") or "")[:10]
                detail = f"{int(item.get('chunks', 0)):,} chunks"
                if indexed:
                    detail += f" · Indexed {indexed}"
                st.caption(detail)
                if item.get("error"):
                    st.caption(f"Needs attention: {item['error']}")
            with view_column:
                if status != "removed" and int(item.get("chunks", 0)) > 0:
                    if st.button("View passages", key=f"view-{item['source_path']}", use_container_width=True):
                        _document_passages_dialog(service, item)
            with manage_column.popover("Details", use_container_width=True):
                st.caption("Source file")
                st.text(str(item["source_path"]))
                if item.get("duplicate_of"):
                    st.caption(f"Duplicate of: {item['duplicate_of']}")
                if status != "removed" and st.button(
                    f"Remove from index: {item['document_name']}", key=f"del-{item['source_path']}", disabled=indexing
                ):
                    service.delete_document(str(item["source_path"]))
                    st.toast(f"Removed {item['document_name']} from the index.")
                    st.rerun()



@st.dialog("Document passages", width="large")
def _document_passages_dialog(service: RagService, document: dict) -> None:
    _document_passages_panel(service, document)


def _document_passages_panel(service: RagService, document: dict) -> None:
    st.subheader(f"Extracted passages: {document['document_name']}")
    count = int(document.get("chunks", 0))
    pages = max(1, (count + 9) // 10)
    identifier = hashlib.sha256(repr((document["source_path"], document.get("file_hash"), count)).encode()).hexdigest()[:12]
    page = int(st.number_input("Passage page", 1, pages, 1, key=f"passages-{identifier}")) if pages > 1 else 1
    offset = (page - 1) * 10
    try:
        chunks = service.document_chunks(str(document["source_path"]), offset=offset, limit=10)
    except Exception as exc:
        st.error(f"Could not load passages: {exc}")
        return
    if not chunks:
        st.info("No stored passages on this page. The index may have changed; refresh the library.")
        return
    st.caption(f"Passages {offset + 1}–{offset + len(chunks)} of {count}")
    for chunk in chunks:
        st.markdown(f"**Passage {int(chunk['sequence']) + 1} · Page {chunk.get('approx_page', 1)}**")
        st.text(str(chunk["text"]))


def _indexing_performance_controls(settings: AppSettings, service: RagService, indexing: bool) -> None:
    st.subheader("Indexing performance")
    st.caption(
        f"Current: {settings.chunking_mode.title()} chunking · {settings.chunk_size} characters · "
        f"batch size {settings.embedding_batch_size}"
    )
    with st.form("indexing-performance"):
        preset = st.selectbox("Indexing preset", ["Keep current", "Fast", "Balanced", "Quality"],
                              disabled=indexing)
        st.caption("Fast: larger sentence-based passages. Balanced: smaller sentence-based passages. "
                   "Quality: topic-aware semantic chunking.")
        batch_size = st.number_input(
            "Embedding batch size", min_value=1, max_value=256,
            value=settings.embedding_batch_size, disabled=indexing,
            help="Larger batches may improve throughput but use more memory. Reduce this if your GPU runs out of memory.",
        )
        if st.form_submit_button("Apply performance settings", disabled=indexing, use_container_width=True):
            updated = apply_indexing_preset(settings, preset, int(batch_size))
            save_settings(updated)
            st.session_state.settings = updated
            st.toast("Performance settings saved. Rebuild existing documents to apply chunking changes.")
            st.rerun(scope="app")

    embedding_ready = service.models.local_embedding_model_exists() if hasattr(service, "models") else False
    if st.button("Benchmark batch sizes", disabled=indexing or not embedding_ready, use_container_width=True,
                 help="Compare local embedding speed on sample text. No documents or settings are changed."):
        progress = st.progress(0, text="Loading search model and warming up…")
        try:
            def update(current: int, total: int, size: int) -> None:
                progress.progress((current - 1) / total, text=f"Testing batch size {size} ({current}/{total})")
            results = benchmark_embedding_batches(service.ensure_embedder(), update)
        except Exception as exc:
            st.error(f"Benchmark failed: {exc}")
        else:
            progress.progress(1.0, text="Benchmark complete")
            st.session_state.batch_benchmark = {
                "repo": settings.embeddings_repo,
                "device": service.models.embedding_device,
                "results": results,
            }
    benchmark = st.session_state.get("batch_benchmark")
    if benchmark and benchmark["repo"] == settings.embeddings_repo:
        st.caption(f"Sample-text benchmark · {str(benchmark['device']).upper()} · real documents may differ")
        st.dataframe([
            {"Batch size": r.batch_size,
             "Passages / second": round(r.passages_per_second, 1) if r.passages_per_second is not None else None,
             "Result": r.error or "Completed"}
            for r in benchmark["results"]
        ], hide_index=True, use_container_width=True)
        successful = [r for r in benchmark["results"] if r.passages_per_second is not None]
        if successful:
            best = max(successful, key=lambda r: r.passages_per_second)
            st.caption(f"Fastest measured batch size: {best.batch_size}. Set it above to use it.")


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

    def progress(event: IndexProgressEvent) -> None:
        job.events.put((event, time.monotonic()))

    def run_job() -> None:
        try:
            job.result = service.index_documents(
                directory,
                replace_existing=replace_existing,
                detail_progress=progress,
                should_cancel=job.cancel_event.is_set,
                retry_failed_only=retry_failed_only,
            )
        except Exception as exc:
            job.error = exc
        finally:
            job.finished_at = time.monotonic()

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
            job.progress.observe(*job.events.get_nowait())
        except queue.Empty:
            break
    alive = bool(job.thread and job.thread.is_alive())
    if alive:
        now = time.monotonic()
        event = job.progress.latest
        labels = {"hashing": "Checking files", "reading": "Reading document",
                  "chunking": "Creating passages", "embedding": "Embedding passages",
                  "writing": "Saving to library", "skipped": "File processed"}
        title = labels.get(event.phase, event.phase.title()) if event else "Preparing index"
        st.progress(job.progress.overall_fraction, text=title)
        if event:
            st.caption(f"File {event.document_index} of {event.documents_total} · {event.document_name}")
            if event.total_units:
                st.progress(event.stage_fraction,
                            text=f"{event.completed_units:,} / {event.total_units:,} {event.unit}")
            if event.detail:
                st.caption(event.detail)
        elapsed_column, device_column = st.columns(2)
        elapsed_column.metric("Elapsed time", _format_elapsed_time(now - job.started_at))
        device = getattr(getattr(service, "models", None), "embedding_device", None)
        device_column.metric("Embedding device", {"cuda": "CUDA GPU", "mps": "Apple MPS", "cpu": "CPU"}.get(
            device, "Not loaded"
        ))
        rate = job.progress.throughput(now)
        remaining = job.progress.remaining_seconds(now)
        if rate is not None:
            st.caption(f"{rate:.1f} passages / second")
        if remaining is not None:
            st.caption(f"Estimated time left in this stage: {_format_elapsed_time(remaining)}")
        elif event and event.unit == "passages" and event.completed_units < event.total_units:
            st.caption("Estimating stage time after a few batches…")
        if st.button("Cancel indexing", type="secondary", use_container_width=True):
            job.cancel_event.set()
            st.warning("Cancellation requested. The current safe checkpoint will finish first.")
        return
    if not job.completion_refreshed:
        job.completion_refreshed = True
        # The fragment refreshes independently while indexing. Refresh the whole
        # app once so document cards, sidebar counts, and controls catch up too.
        st.rerun(scope="app")
    if job.error:
        st.error(f"Indexing failed: {job.error}")
    elif job.result:
        result = job.result
        if result.cancelled:
            st.warning("Indexing was cancelled. Completed files remain safely indexed.")
        else:
            st.success("Incremental indexing complete.")
        if job.finished_at is not None:
            st.metric("Total elapsed time", _format_elapsed_time(job.finished_at - job.started_at))
        st.markdown(
            f"Indexed `{result.document_count}` changed/new files into `{result.chunk_count}` chunks · "
            f"unchanged `{result.unchanged_count}` · duplicates `{result.duplicate_count}` · "
            f"failed `{result.failed_count}` · removed `{result.removed_count}`"
        )
        for warning in result.warnings:
            st.warning(warning)
        if result.failed_count and st.button("Retry failed files", use_container_width=True):
            _start_index_job(service, job.directory, retry_failed_only=True)
            st.rerun(scope="app")


def _format_elapsed_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes} min {seconds} s"
    return f"{minutes} min {seconds} s"


def _models_tab(settings: AppSettings, service: RagService) -> None:
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
        "Download models once for private search and answers on your device.",
    )
    status_left, status_right = st.columns(2, gap="medium")
    with status_left:
        model_status_card(
            "Embedding model",
            settings.embeddings_repo,
            "Ready to search your documents" if embedding_ready else "Download to index and search documents",
            embedding_ready,
        )
    with status_right:
        model_status_card(
            "Answer model",
            settings.llm_repo,
            "Ready to answer questions" if local_llm else "Download to generate answers",
            llm_ready,
        )

    device = getattr(service.models, "embedding_device", None)
    device_labels = {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple GPU (MPS)", "cpu": "CPU"}
    if device:
        st.caption(f"Embedding device: {device_labels.get(device, device)}")
    else:
        st.caption("Embedding device: automatic (CUDA/MPS → CPU), selected on first use.")

    if catalog_warning:
        st.info("The online catalog is unavailable. Built-in model options are shown instead.")

    models_ready = embedding_ready and llm_ready
    if st.button(
        "Verify model installation" if models_ready else "Download missing models",
        type="secondary" if models_ready else "primary",
        use_container_width=True,
        help="Existing model files are reused. Only missing files are downloaded.",
    ):
        if not _download_models_in_ui(service):
            return
        st.session_state.model_download_notice = (
            "Models are ready to use."
        )
        st.rerun()

    with st.expander("Change models", expanded=False):
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
            with model_right:
                llm_repo = st.selectbox(
                    "Answer model",
                    options=llm_options,
                    index=llm_options.index(settings.llm_repo),
                    format_func=model_display_name,
                    help="Generates cited answers from retrieved passages using llama.cpp.",
                )
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

    st.divider()
    section_intro(
        "Advanced settings",
        "Fine-tune search, indexing, and scanned-document support.",
    )
    show_advanced = st.toggle(
        "Enable advanced settings",
        key="show_advanced_settings",
        help="These controls affect retrieval quality, index construction, and processing time.",
    )
    if not show_advanced:
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
            chunking_mode = st.selectbox(
                "Chunking method",
                options=["fast", "semantic"],
                index=["fast", "semantic"].index(settings.chunking_mode),
                format_func=lambda mode: "Fast — sentence boundaries" if mode == "fast" else "Semantic — topic-aware boundaries",
                help="Fast skips sentence embeddings during chunking. Both methods use semantic search. "
                     "Rebuild the index to apply this to documents already indexed.",
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
        settings.chunking_mode = chunking_mode
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


def _answer_actions(
    answer: str,
    turn_index: int,
    sources: list[SourceRecord],
    conversation_store: ConversationStore,
    *,
    is_latest: bool,
) -> None:
    plain_answer = re.sub(r"\s*\[S\d+\]", "", answer).strip()
    cited_answer = _answer_markdown(answer, sources)
    copy_column, export_column, follow_up_column, regenerate_column = st.columns(
        [1, 1, 1.25, 1.25], gap="small"
    )
    with copy_column.popover(
        "Copy", icon=":material/content_copy:", use_container_width=True
    ):
        answer_tab, citations_tab = st.tabs(["Answer", "With citations"])
        with answer_tab:
            st.caption("Use the copy icon in the code block.")
            st.code(plain_answer, language=None)
        with citations_tab:
            st.caption("Includes a source list for sharing.")
            st.code(cited_answer, language=None)
    export_column.download_button(
        "Export",
        cited_answer,
        file_name=f"answer-{turn_index}.md",
        mime="text/markdown",
        icon=":material/download:",
        key=f"export-answer-{turn_index}",
        use_container_width=True,
    )
    if is_latest:
        with follow_up_column.popover(
            "Ask a follow-up", icon=":material/add_comment:", use_container_width=True
        ):
            st.caption("Continue from this answer")
            follow_ups = [
                "Explain that more simply.",
                "What evidence best supports that answer?",
                "What are the most important implications?",
            ]
            for follow_up in follow_ups:
                if st.button(
                    follow_up,
                    key=f"follow-up-{turn_index}-{follow_up}",
                    use_container_width=True,
                ):
                    st.session_state.pending_chat_prompt = follow_up
                    st.rerun()
        if regenerate_column.button(
            "Regenerate",
            icon=":material/refresh:",
            key=f"regenerate-{turn_index}",
            use_container_width=True,
        ):
            previous_index = turn_index - 1
            if previous_index >= 0 and st.session_state.messages[previous_index]["role"] == "user":
                prompt = st.session_state.messages[previous_index]["content"]
                st.session_state.messages.pop()
                st.session_state.sources_by_turn.pop(turn_index, None)
                st.session_state.diagnostics_by_turn.pop(turn_index, None)
                st.session_state.pending_regenerate_prompt = prompt
                _persist_current_conversation(conversation_store)
                st.rerun()
    else:
        follow_up_column.empty()
        regenerate_column.empty()


def _answer_markdown(answer: str, sources: list[SourceRecord]) -> str:
    if not sources:
        return answer.strip()
    source_lines = [
        f"- [{source.source_id}] {source.document_name}, page {source.approx_page} — "
        f"{source.source_path}"
        for source in sources
    ]
    return f"{answer.strip()}\n\nSources\n\n" + "\n".join(source_lines)


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
    for source in visible_sources:
        _source_card(source, anchor_prefix, search=search)


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
    conversation = store.create()
    _load_conversation(conversation)
    st.session_state.pending_conversation_id = conversation["id"]


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
    st.markdown("#### Download conversation")
    export_markdown, export_pdf = st.columns(2)
    export_markdown.download_button(
        "Download Markdown (.md)",
        markdown,
        file_name=f"{safe_name}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    export_pdf.download_button(
        "Download PDF (.pdf)",
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
    source, anchor_prefix: str = "source", search: str = ""
) -> None:
    source_id = html.escape(str(source.source_id))
    document_name = html.escape(str(source.document_name))
    source_text = _highlight_text(
        str(source.text), search or getattr(source, "matched_passage", "")
    )
    anchor = html.escape(f"{anchor_prefix}-{str(source.source_id).lower()}", quote=True)
    relevance, relevance_class = _relevance_label(float(source.score))
    st.markdown(f'<span id="{anchor}"></span>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="da-source-heading">
                <div><strong>[{source_id}] {document_name}</strong><span>Page {source.approx_page}</span></div>
                <span class="da-relevance {relevance_class}">{relevance}</span>
            </div>
            <div class="da-source-passage">{source_text}</div>
            """,
            unsafe_allow_html=True,
        )
        open_column, citation_column, details_column = st.columns([1.2, 1.2, 2], gap="small")
        source_url = _source_url(str(source.source_path), int(source.approx_page))
        open_column.link_button(
            "Open document",
            source_url,
            icon=":material/open_in_new:",
            use_container_width=True,
            help="Open the local source file, at the cited page when supported.",
        )
        citation = f"[{source.source_id}] {source.document_name}, page {source.approx_page}"
        with citation_column.popover(
            "Copy citation", icon=":material/content_copy:", use_container_width=True
        ):
            st.caption("Use the copy icon in the code block.")
            st.code(citation, language=None)
        with details_column.popover(
            "Technical details", icon=":material/tune:", use_container_width=True
        ):
            st.caption("Source file")
            st.text(str(source.source_path))
            st.caption(
                f"Hybrid score {source.score:.3f} · Semantic {source.semantic_score:.3f} · "
                f"BM25 {source.lexical_score:.3f}"
            )


def _source_url(source_path: str, page: int) -> str:
    path = Path(source_path).expanduser()
    try:
        url = path.resolve().as_uri()
    except ValueError:
        return "#"
    if path.suffix.casefold() == ".pdf":
        return f"{url}#page={max(1, page)}"
    return url


def _relevance_label(score: float) -> tuple[str, str]:
    if score >= 0.65:
        return "High relevance", "high"
    if score >= 0.35:
        return "Relevant", "medium"
    return "Supporting", "supporting"


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



def _download_models_in_ui(service: RagService) -> bool:
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
        _download_models_with_updates(
            service, update_download_progress
        )
    except Exception as exc:  # Hugging Face exposes several transport-specific errors.
        st.error(f"Model download failed: {exc}")
        return False
    overall_progress.progress(1.0, text="Both models are ready")
    file_progress.progress(1.0, text="Download complete")
    return True


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
