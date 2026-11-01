from __future__ import annotations

import html

import streamlit as st


VIEW_DETAILS = {
    "Ask": ("Ask your documents", "Find answers, explore ideas, and check the sources."),
    "Library": ("Document library", "Your documents, ready to explore."),
    "Settings": ("Settings", "Manage your models and tailor your workspace."),
    "Help": ("Help", "Get started, understand answers, and find troubleshooting guidance."),
}


def sidebar_brand() -> None:
    st.markdown(
        """
        <div class="da-brand">
            <div class="da-brand-mark">D</div>
            <div>
                <div class="da-brand-name">Document Analyst</div>
                <div class="da-brand-copy">Private, local research</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_stats(documents: int, chunks: int) -> None:
    st.markdown(
        f"""
        <div class="da-sidebar-stats">
            <div class="da-sidebar-stat"><strong>{documents}</strong>Documents</div>
            <div class="da-sidebar-stat"><strong>{chunks}</strong>Chunks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_heading(view: str) -> None:
    title, description = VIEW_DETAILS[view]
    st.markdown(
        f"""
        <div class="da-page-heading">
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(title: str, description: str, icon: str = "◇") -> None:
    st.markdown(
        f"""
        <div class="da-empty-state">
            <div class="da-empty-icon">{html.escape(icon)}</div>
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_intro(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="da-section-intro">
            <h3>{html.escape(title)}</h3>
            <p>{html.escape(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_status_card(
    role: str,
    repository: str,
    detail: str,
    ready: bool,
) -> None:
    badge = "Installed" if ready else "Not installed"
    badge_class = "ready" if ready else ""
    st.markdown(
        f"""
        <div class="da-model-card">
            <div class="da-model-card-top">
                <span class="da-model-role">{html.escape(role)}</span>
                <span class="da-badge {badge_class}">{badge}</span>
            </div>
            <div class="da-model-name">{html.escape(model_display_name(repository))}</div>
            <div class="da-model-detail">{html.escape(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_display_name(repository: str) -> str:
    return repository.rsplit("/", 1)[-1].replace("-GGUF", "").replace("-", " ")
