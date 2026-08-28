from __future__ import annotations

import streamlit as st


SESSION_DEFAULTS = {
    "messages": [],
    "sources_by_turn": {},
    "diagnostics_by_turn": {},
    "show_retrieval_diagnostics": False,
    "pinned_source_paths": [],
    "excluded_source_paths": [],
    "active_view": "Ask",
    "show_empty_state_dialog": True,
    "index_job": None,
}

LEGACY_VIEW_NAMES = {
    "Chat": "Ask",
    "Manage Documents": "Library",
    "Models & Settings": "Settings",
}


def initialize_session_state() -> None:
    """Install independent defaults once and migrate pre-redesign navigation state."""
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value
    st.session_state.active_view = LEGACY_VIEW_NAMES.get(
        st.session_state.active_view, st.session_state.active_view
    )

