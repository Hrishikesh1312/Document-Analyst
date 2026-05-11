from __future__ import annotations

from pathlib import Path

import streamlit as st

from document_analyst.config import AppSettings, load_settings, save_settings
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

    settings: AppSettings = st.session_state.settings
    service = RagService(settings)

    theme_mode = _sidebar(settings, service)
    _inject_theme(theme_mode)
    _hero(service)

    chat_tab, docs_tab, models_tab = st.tabs(["Chat", "Manage Documents", "Models & Settings"])
    with chat_tab:
        _chat_tab(service)
    with docs_tab:
        _docs_tab(settings, service)
    with models_tab:
        _models_tab(settings, service)


def _sidebar(settings: AppSettings, service: RagService) -> str:
    with st.sidebar:
        st.title("Document Analyst")
        theme_mode = st.radio("Appearance", ["Dark", "Light"], horizontal=True)
        stats = service.stats()
        col1, col2 = st.columns(2)
        col1.metric("Docs", stats["documents"])
        col2.metric("Chunks", stats["chunks"])
        st.caption("Fully local RAG. First launch can download models into `models/`.")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources_by_turn = {}
            st.rerun()
    return theme_mode


def _inject_theme(theme_mode: str) -> None:
    palette = {
        "Dark": {
            "bg": "#08121f",
            "panel": "#101d31",
            "card": "#13243b",
            "border": "#264563",
            "text": "#ecf3fb",
            "muted": "#98aeca",
            "accent": "#7ee6b6",
        },
        "Light": {
            "bg": "#f4f7fb",
            "panel": "#ffffff",
            "card": "#eef4fb",
            "border": "#d0deef",
            "text": "#102033",
            "muted": "#58708a",
            "accent": "#0f8f63",
        },
    }[theme_mode]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top right, {palette["card"]} 0%, transparent 30%),
                linear-gradient(180deg, {palette["bg"]} 0%, {palette["bg"]} 100%);
            color: {palette["text"]};
        }}
        .hero-card, .source-card, .metric-card {{
            background: {palette["panel"]};
            border: 1px solid {palette["border"]};
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 30px rgba(0,0,0,0.08);
        }}
        .source-card {{
            background: {palette["card"]};
            margin-bottom: 0.8rem;
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
    st.markdown(
        f"""
        - Embedding model directory: `{settings.embeddings_dir}`
        - Embedding repo: `{settings.embeddings_repo}`
        - LLM repo: `{settings.llm_repo}`
        - Local GGUF: `{local_llm or 'Not downloaded yet'}`
        """
    )
    if st.button("Download Recommended Models", use_container_width=True):
        with st.spinner("Downloading models into the local models folder..."):
            embedding_dir, llm_path = service.download_models()
        st.success(f"Models ready.\nEmbedding model: {embedding_dir}\nLLM: {llm_path}")

    st.write("")
    st.subheader("Advanced Settings")
    with st.form("settings-form"):
        embeddings_repo = st.text_input("Embedding repo", value=settings.embeddings_repo)
        llm_repo = st.text_input("LLM GGUF repo", value=settings.llm_repo)
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
        settings.embeddings_repo = embeddings_repo.strip()
        settings.llm_repo = llm_repo.strip()
        settings.llm_filename = llm_filename.strip()
        settings.top_k = int(top_k)
        settings.max_history_turns = int(max_history_turns)
        settings.semantic_threshold = float(semantic_threshold)
        save_settings(settings)
        st.session_state.settings = settings
        st.success("Settings saved.")


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
