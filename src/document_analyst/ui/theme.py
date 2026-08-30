from __future__ import annotations

import streamlit as st


THEME_CSS = """
<style>
:root {
    --da-bg: #0b0f14;
    --da-sidebar: #10161d;
    --da-surface: #151d26;
    --da-surface-raised: #1a242f;
    --da-border: #26313d;
    --da-border-strong: #354455;
    --da-text: #f2f5f7;
    --da-muted: #94a3b3;
    --da-accent: #42d6a4;
    --da-accent-soft: rgba(66, 214, 164, 0.12);
    --da-warning: #f6b94a;
    --da-error: #f36c75;
    --da-radius: 12px;
}

.stApp {
    background: var(--da-bg);
    color: var(--da-text);
}

.block-container {
    max-width: 1440px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: var(--da-sidebar);
    border-right: 1px solid var(--da-border);
}
[data-testid="stSidebarUserContent"] {
    min-height: 100%;
    display: flex;
    flex-direction: column;
}
[data-testid="stSidebarUserContent"] > div {
    display: flex;
    flex: 1;
    flex-direction: column;
}
[data-testid="stSidebar"] .st-key-sidebar_reset {
    margin-top: auto;
    padding-top: 1rem;
}

.da-brand {
    display: flex;
    align-items: center;
    gap: .7rem;
    margin: .15rem 0 1.35rem;
}
.da-brand-mark {
    align-items: center;
    background: var(--da-accent);
    border-radius: 10px;
    color: #07120e;
    display: flex;
    font-size: 1.15rem;
    font-weight: 800;
    height: 38px;
    justify-content: center;
    width: 38px;
}
.da-brand-name { font-size: 1rem; font-weight: 700; letter-spacing: -.01em; }
.da-brand-copy { color: var(--da-muted); font-size: .73rem; margin-top: .08rem; }

[data-testid="stSidebar"] [role="radiogroup"] > label {
    border: 1px solid transparent;
    border-radius: 10px;
    box-sizing: border-box;
    margin-bottom: .3rem;
    min-height: 44px;
    padding: .25rem .65rem;
    width: 100%;
}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background: rgba(255,255,255,.035);
}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background: var(--da-accent-soft);
    border-color: rgba(66,214,164,.3);
}
[data-testid="stSidebar"] [role="radiogroup"] p {
    color: var(--da-text);
    font-weight: 600;
}

.da-sidebar-stats {
    border-top: 1px solid var(--da-border);
    display: grid;
    gap: .5rem;
    grid-template-columns: 1fr 1fr;
    margin-top: 1.1rem;
    padding-top: 1rem;
}
.da-sidebar-stat { color: var(--da-muted); font-size: .72rem; }
.da-sidebar-stat strong { color: var(--da-text); display: block; font-size: 1rem; }

.da-app-header {
    align-items: center;
    border-bottom: 1px solid var(--da-border);
    display: flex;
    justify-content: space-between;
    margin-bottom: 1.75rem;
    padding: .1rem 0 1.1rem;
}
.da-app-title { font-size: 1.1rem; font-weight: 700; letter-spacing: -.02em; }
.da-app-meta { color: var(--da-muted); font-size: .8rem; margin-top: .2rem; }
.da-status {
    align-items: center;
    background: var(--da-surface);
    border: 1px solid var(--da-border);
    border-radius: 999px;
    color: var(--da-muted);
    display: inline-flex;
    font-size: .75rem;
    gap: .45rem;
    padding: .38rem .7rem;
}
.da-status-dot { background: var(--da-muted); border-radius: 50%; height: 7px; width: 7px; }
.da-status.ready .da-status-dot { background: var(--da-accent); box-shadow: 0 0 0 3px var(--da-accent-soft); }

.da-page-heading { margin-bottom: 1.4rem; }
.da-eyebrow {
    color: var(--da-accent);
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .1em;
    margin-bottom: .35rem;
    text-transform: uppercase;
}
.da-page-heading h1 { font-size: 2rem; letter-spacing: -.04em; margin: 0; }
.da-page-heading p { color: var(--da-muted); margin: .4rem 0 0; max-width: 700px; }

.da-panel, .source-card, .metric-card {
    background: var(--da-surface);
    border: 1px solid var(--da-border);
    border-radius: var(--da-radius);
    padding: 1rem 1.1rem;
}
.source-card { margin-bottom: .65rem; }
.da-model-card {
    background: var(--da-surface);
    border: 1px solid var(--da-border);
    border-radius: var(--da-radius);
    margin-bottom: .75rem;
    min-height: 118px;
    padding: 1rem 1.1rem;
}
.da-model-card-top {
    align-items: center;
    display: flex;
    justify-content: space-between;
    margin-bottom: .45rem;
}
.da-model-role { font-size: .78rem; font-weight: 700; letter-spacing: .03em; }
.da-model-name { font-size: 1rem; font-weight: 650; margin-bottom: .3rem; }
.da-model-detail { color: var(--da-muted); font-size: .78rem; overflow-wrap: anywhere; }
.da-badge {
    border: 1px solid var(--da-border-strong);
    border-radius: 999px;
    color: var(--da-muted);
    font-size: .68rem;
    font-weight: 700;
    padding: .2rem .5rem;
    text-transform: uppercase;
}
.da-badge.ready {
    background: var(--da-accent-soft);
    border-color: rgba(66,214,164,.3);
    color: var(--da-accent);
}
.da-section-intro { margin: .25rem 0 1rem; }
.da-section-intro h3 { font-size: 1.1rem; margin: 0; }
.da-section-intro p { color: var(--da-muted); font-size: .86rem; margin: .3rem 0 0; }
.da-empty-state {
    background: var(--da-surface);
    border: 1px dashed var(--da-border-strong);
    border-radius: var(--da-radius);
    margin: 1rem 0;
    padding: 2.5rem 1.5rem;
    text-align: center;
}
.da-empty-icon {
    align-items: center;
    background: var(--da-accent-soft);
    border-radius: 12px;
    color: var(--da-accent);
    display: inline-flex;
    font-size: 1.35rem;
    height: 46px;
    justify-content: center;
    margin-bottom: .8rem;
    width: 46px;
}
.da-empty-state h3 { margin: 0; }
.da-empty-state p { color: var(--da-muted); margin: .45rem auto 0; max-width: 520px; }

[data-testid="stChatMessage"] {
    background: transparent;
    border: 0;
    border-bottom: 1px solid var(--da-border);
    border-radius: 0;
    padding: .8rem .25rem 1.1rem;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(255,255,255,.018);
}
[data-testid="stChatInput"] { border-color: var(--da-border-strong); }

.muted { color: var(--da-muted); }
.accent { color: var(--da-accent); }
mark { background: rgba(246,185,74,.25); color: var(--da-text); padding: 0 .12em; }
code { color: var(--da-accent); }

div.stButton > button, div.stDownloadButton > button { border-radius: 9px; }
[data-testid="stMetric"] {
    background: var(--da-surface);
    border: 1px solid var(--da-border);
    border-radius: var(--da-radius);
    padding: .75rem .9rem;
}
[data-testid="stExpander"] { border-color: var(--da-border); border-radius: var(--da-radius); }

@media (max-width: 900px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .da-app-header { align-items: flex-start; gap: .8rem; }
    .da-page-heading h1 { font-size: 1.65rem; }
}
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
