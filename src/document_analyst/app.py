from __future__ import annotations

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
    if "active_view" not in st.session_state:
        st.session_state.active_view = "Chat"

    settings: AppSettings = st.session_state.settings
    service = RagService(settings)

    active_view = _sidebar(settings, service)
    _inject_theme()
    _hero(service)

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
        st.caption("Fully local RAG with a darker single-theme interface and first-launch model downloads.")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources_by_turn = {}
            st.rerun()
    return active_view


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
                Privacy-first semantic search and local Q&A for PDFs, Markdown, and text files.
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


def _chat_tab(service: RagService) -> None:
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        for index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    sources = st.session_state.sources_by_turn.get(index, [])
                    if sources:
                        with st.expander("Sources", expanded=False):
                            for source in sources:
                                _source_card(source)

        prompt = st.chat_input("Ask about your indexed documents")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching your local index and generating a response..."):
                    result = service.answer_question(prompt, st.session_state.messages[:-1])
                st.markdown(result.answer)
                with st.expander("Sources", expanded=True):
                    for source in result.sources:
                        _source_card(source)
                turn_index = len(st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
                st.session_state.sources_by_turn[turn_index] = result.sources
            st.rerun()

    with right:
        st.markdown("#### How answers are built")
        st.markdown(
            """
            1. Your question is embedded locally.
            2. Chroma retrieves the most similar document chunks.
            3. The local GGUF model answers with inline citations like `[S1]`.
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
            with st.spinner("Parsing files, chunking semantically, embedding, and writing to Chroma..."):
                result = service.index_documents(settings.documents_dir, replace_existing=replace_existing)
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
            """
        )

    st.write("")
    st.subheader("Indexed Documents")
    docs = service.indexed_documents()
    if not docs:
        st.info("No indexed documents yet.")
        return

    for item in docs:
        with st.container():
            st.markdown(
                f"""
                <div class="source-card">
                    <strong>{item["document_name"]}</strong><br/>
                    <span class="muted">{item["source_path"]}</span><br/>
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
    embedding_options = _repo_options(EMBEDDING_REPO_OPTIONS, settings.embeddings_repo)
    llm_options = _repo_options(LLM_REPO_OPTIONS, settings.llm_repo)
    st.markdown(
        f"""
        - Embedding model directory: `{settings.embeddings_dir}`
        - Embedding repo: `{settings.embeddings_repo}`
        - LLM repo: `{settings.llm_repo}`
        - Local GGUF: `{local_llm or 'Not downloaded yet'}`
        """
    )
    if st.button("Download Selected Models", use_container_width=True):
        with st.spinner("Downloading models into the local models folder..."):
            embedding_dir, llm_path = service.download_models()
        st.success(f"Models ready.\nEmbedding model: {embedding_dir}\nLLM: {llm_path}")

    st.write("")
    st.subheader("Model Selection")
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
        max_history_turns = st.slider("Turns kept in prompt", min_value=2, max_value=10, value=settings.max_history_turns)
        semantic_threshold = st.slider(
            "Semantic chunking threshold",
            min_value=0.3,
            max_value=0.8,
            value=float(settings.semantic_threshold),
            step=0.05,
        )
        saved = st.form_submit_button("Save Settings", use_container_width=True)
    if saved:
        repo_changed = embeddings_repo.strip() != settings.embeddings_repo or llm_repo.strip() != settings.llm_repo
        settings.embeddings_repo = embeddings_repo.strip()
        settings.llm_repo = llm_repo.strip()
        settings.llm_filename = "" if repo_changed and not llm_filename.strip() else llm_filename.strip()
        settings.top_k = int(top_k)
        settings.max_history_turns = int(max_history_turns)
        settings.semantic_threshold = float(semantic_threshold)
        save_settings(settings)
        st.session_state.settings = settings
        st.success("Settings saved.")
        st.rerun()


def _source_card(source) -> None:
    st.markdown(
        f"""
        <div class="source-card">
            <strong>[{source.source_id}] {source.document_name}</strong><br/>
            <span class="muted">{source.source_path}</span><br/>
            <span class="accent">Page {source.approx_page} • score {source.score:.3f}</span>
            <p style="margin:0.6rem 0 0 0;">{source.text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _repo_options(defaults: list[str], current_value: str) -> list[str]:
    if current_value in defaults:
        return defaults
    return [current_value, *defaults]
