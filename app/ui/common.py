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
.block-container { padding-top: 3.4rem; max-width: 1280px; }
h1 { font-weight: 300 !important; letter-spacing: -0.02em; line-height: 1.02 !important; }
h2, h3 { font-weight: 300 !important; letter-spacing: -0.01em; }
h1 em, h2 em { color: #bcb8ae; }
[data-testid="stSidebar"] h3 { font-family: "Hanken Grotesk", sans-serif !important; font-weight: 600 !important; }
.nb-eyebrow { font-family: "Geist Mono", ui-monospace, monospace; font-size: .72rem; letter-spacing: .02em; color: #868791; margin-bottom: .6rem; display: flex; align-items: center; gap: 10px; }
.nb-eyebrow::before { content: ""; width: 22px; height: 1px; background: #868791; }
.nb-muted { color: #868791; font-size: .9rem; }
.nb-card { border: 1px solid #1d2434; background: #080c17; padding: 14px 16px; margin-bottom: 10px; border-radius: 2px; }
.nb-card h4 { margin: 0 0 4px 0; font-size: 1rem; font-weight: 600; }
.nb-pill { display: inline-flex; align-items: center; gap: 7px; padding: 3px 9px; font-family: "Geist Mono", ui-monospace, monospace; font-size: .68rem; letter-spacing: .02em; margin-right: 6px; border: 1px solid rgba(236,231,220,.24); color: #bcb8ae; border-radius: 2px; }
.nb-pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: #868791; }
.nb-assoc::before { background: #8fd3ee; } .nb-mech::before { background: #43c9b4; } .nb-plaus::before { background: #9d8cf2; } .nb-insuf::before { background: #e6a35c; }
.nb-ok::before { background: #43c9b4; } .nb-warn::before { background: #e6a35c; } .nb-no::before { background: #9d8cf2; }
.nb-stage { display:flex; gap:12px; align-items:flex-start; padding:9px 0; border-bottom:1px solid #161c2a; }
.nb-kpi { font-family: "Newsreader", Georgia, serif; font-weight: 300; font-size: 2.3rem; line-height: 1; font-variant-numeric: lining-nums tabular-nums; }
.nb-kpi-label { font-family: "Geist Mono", ui-monospace, monospace; font-size: .68rem; color: #868791; margin-top: 6px; }
.nb-banner { border-left: 2px solid #43c9b4; padding: 10px 14px; background: #080c17; font-size: .92rem; color: #bcb8ae; }
.nb-banner b { color: #ece7dc; }
.nb-brand { display: flex; align-items: center; gap: 10px; margin: 2px 0 6px; }
.nb-brand b { font-weight: 600; font-size: 1.02rem; letter-spacing: .01em; }
[data-testid="stCaptionContainer"] { color: #868791; }
</style>
"""

LEVEL_PILL = {
    "statistical association": ("Association", "nb-assoc"),
    "mechanistic evidence": ("Mechanistic · literature", "nb-mech"),
    "biological plausibility": ("Plausibility", "nb-plaus"),
    "causal evidence": ("Causal", "nb-mech"),
    "insufficient evidence": ("Insufficient", "nb-insuf"),
}
STATUS_PILL = {"supported": "nb-ok", "partially supported": "nb-warn", "not supported": "nb-no"}
BRAND_SVG = ('<svg width="30" height="20" viewBox="0 0 30 20" aria-hidden="true"><circle cx="6" cy="10" r="4.4" fill="none" stroke="#43c9b4" stroke-width="1.5"/>'
             '<path d="M10.4 10h7.2" stroke="#ece7dc" stroke-width="1.2" stroke-dasharray="1.6 1.6"/><circle cx="23.6" cy="10" r="5.4" fill="none" stroke="#9d8cf2" stroke-width="1.5"/>'
             '<circle cx="23.6" cy="10" r="1.8" fill="#9d8cf2"/></svg>')


def setup(title: str | None = None) -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def show_fig(fig, key: str | None = None, caption: bool = True) -> None:
    st.plotly_chart(fig, theme=None, key=key, width="stretch", config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})
    cap = (fig.layout.meta or {}).get("caption") if fig.layout.meta else None
    if caption and cap:
        st.caption(cap)


def eyebrow(text: str) -> None:
    st.markdown(f"<div class='nb-eyebrow'>{text}</div>", unsafe_allow_html=True)


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
