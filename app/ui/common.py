"""Shared UI helpers: styling, plot rendering, cached data access, small components."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

TAGLINE = "From biological hypotheses to clinically relevant evidence."

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
code, pre { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
.block-container { padding-top: 3.6rem; max-width: 1280px; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
.nb-eyebrow { text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; font-weight: 600; color: #256abf; margin-bottom: .2rem; }
.nb-muted { color: #6b6a65; font-size: .9rem; }
.nb-card { border: 1px solid rgba(128,128,128,.22); border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; background: rgba(128,128,128,.03); }
.nb-card h4 { margin: 0 0 4px 0; font-size: 1rem; font-weight: 600; }
.nb-pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .72rem; font-weight: 600; margin-right: 4px; border: 1px solid rgba(128,128,128,.35); }
.nb-assoc { background: #e6eef9; color: #184f95; border-color: #b7d3f6; }
.nb-mech { background: #efe9fb; color: #4a3aa7; border-color: #d2c9f3; }
.nb-plaus { background: #f3f2ee; color: #52514e; }
.nb-insuf { background: #fdecea; color: #a12c2c; border-color: #f4c2c0; }
.nb-ok { background: #e7f5e7; color: #136b13; border-color: #bfe3bf; }
.nb-warn { background: #fff4dc; color: #8a5a00; border-color: #f6d999; }
.nb-stage { display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-bottom:1px solid rgba(128,128,128,.15); }
.nb-kpi { font-size: 1.6rem; font-weight: 700; line-height: 1.1; }
.nb-kpi-label { font-size: .78rem; color: #6b6a65; text-transform: uppercase; letter-spacing: .06em; }
.nb-banner { border-left: 3px solid #256abf; padding: 8px 12px; background: rgba(37,106,191,.06); border-radius: 4px; font-size: .9rem; }
@media (prefers-color-scheme: dark) { .nb-muted, .nb-kpi-label { color: #a9a8a0; } }
</style>
"""

LEVEL_PILL = {
    "statistical association": ("ASSOCIATION", "nb-assoc"),
    "mechanistic evidence": ("MECHANISTIC (literature)", "nb-mech"),
    "biological plausibility": ("PLAUSIBILITY", "nb-plaus"),
    "causal evidence": ("CAUSAL", "nb-mech"),
    "insufficient evidence": ("INSUFFICIENT", "nb-insuf"),
}
STATUS_PILL = {"supported": "nb-ok", "partially supported": "nb-warn", "not supported": "nb-insuf"}


def setup(title: str | None = None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001
        return False


def show_fig(fig, key: str | None = None, caption: bool = True) -> None:
    from neurobridge.viz.theme import for_dark_mode

    f = fig
    if is_dark():
        import plotly.graph_objects as go

        f = for_dark_mode(go.Figure(fig))
    st.plotly_chart(f, theme=None, key=key, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
    cap = (fig.layout.meta or {}).get("caption") if fig.layout.meta else None
    if caption and cap:
        st.caption(cap)


def pill(text: str, cls: str) -> str:
    return f"<span class='nb-pill {cls}'>{text}</span>"


def claim_card(c: dict, citations: dict | None = None) -> None:
    lvl, cls = LEVEL_PILL.get(c["level"], (c["level"], "nb-plaus"))
    st_cls = STATUS_PILL.get(c["status"], "nb-plaus")
    cites = ""
    if c.get("citations") and citations:
        cites = " · ".join(f"<a href='https://pubmed.ncbi.nlm.nih.gov/{p}/' target='_blank'>{citations[p].short() if p in citations else 'PMID ' + p}</a>" for p in c["citations"])
    analyses = f"<div class='nb-muted'>analyses: <code>{', '.join(c['analyses'])}</code></div>" if c.get("analyses") else ""
    cav = "".join(f"<li>{x}</li>" for x in c.get("caveats", []))
    st.markdown(
        f"<div class='nb-card'><div>{pill(lvl, cls)}{pill(c['kind'], 'nb-plaus')}{pill(c['status'], st_cls)} <span class='nb-muted'>{c['id']}</span></div>"
        f"<div style='margin:6px 0'>{c['text']}</div>{analyses}"
        + (f"<div class='nb-muted'>evidence: {cites}</div>" if cites else "")
        + (f"<details><summary class='nb-muted'>caveats ({len(c.get('caveats', []))})</summary><ul class='nb-muted'>{cav}</ul></details>" if cav else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def kpi(col, value, label) -> None:
    col.markdown(f"<div class='nb-kpi'>{value}</div><div class='nb-kpi-label'>{label}</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading cohort (first load downloads and caches the public files)...")
def cohort(dataset_id: str):
    from neurobridge.data.loaders import load_cohort

    return load_cohort(dataset_id)


@st.cache_resource(show_spinner="Running the single-cell pipeline (QC, normalisation, UMAP, Leiden; ~20 s)...")
def processed_single_cell(dataset_id: str = "gse84465_darmanis"):
    from neurobridge.analysis import singlecell as scm
    from neurobridge.data.loaders import load_single_cell

    a, rec = scm.preprocess(load_single_cell(dataset_id))
    return a, rec


@st.cache_data(show_spinner=False)
def catalog_frame() -> pd.DataFrame:
    from neurobridge.data.catalog import build_catalog

    return pd.DataFrame([d.model_dump(mode="json") for d in build_catalog()])


def safe_call(fn, *args, what: str = "operation", **kwargs):
    """Run fn and show a clear error state instead of a stack trace."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        st.error(f"{what} failed: {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            import traceback

            st.code(traceback.format_exc())
        return None


def disclaimer() -> None:
    st.caption("NeuroBridge is a research exploration platform. It does not make diagnostic or treatment recommendations. All results are computed from public data during the session; associations are not causal claims.")


@st.cache_resource(show_spinner="Loading and processing the Visium section (first load downloads ~30 MB)...")
def processed_spatial(dataset_id: str = "visium_gbm_10x"):
    from neurobridge.analysis import spatial as spa
    from neurobridge.data.loaders import load_spatial
    from neurobridge.data.spatial import lowres_image

    a, rec = spa.preprocess_spatial(load_spatial(dataset_id))
    dom = spa.annotate_domains(a)
    return a, rec, dom, spa.hex_neighbours(a.obs), lowres_image(a)
