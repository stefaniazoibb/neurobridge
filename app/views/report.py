import tempfile
from pathlib import Path

import streamlit as st

from ui.common import disclaimer, setup, show_fig

setup()
from neurobridge.config import get_settings  # noqa: E402
from neurobridge.report.builder import REPORT_FIGURES  # noqa: E402
from neurobridge.report.export import export_markdown, markdown_to_pdf  # noqa: E402
from neurobridge.runs import list_runs, load_run  # noqa: E402

st.markdown("<div class='nb-eyebrow'>Module E</div>", unsafe_allow_html=True)
st.title("Evidence Report")
st.caption("Generated from executed analyses only. Every claim carries an evidence tag and links to its analysis id or to a verified PubMed record; interpretations are labelled as such.")

runs = list_runs(50)
if not runs:
    st.info("No investigation has been run yet. Start one from the Research Copilot.")
    st.stop()
ids = [r["id"] for r in runs]
default = ids.index(st.session_state["report_run"]) if st.session_state.get("report_run") in ids else 0
rid = st.selectbox("Investigation", ids, index=default, format_func=lambda i: next(f"{r['started_at']} · {r['question'][:90]}" for r in runs if r["id"] == i))

live = st.session_state.get("inv")
if live and live[0].investigation_id == rid:
    res, view = live
else:
    res, view = load_run(rid)

run_dir = get_settings().runs_dir / rid
md_text = (run_dir / "report.md").read_text() if (run_dir / "report.md").exists() else (res.report_markdown or "")

c1, c2, c3 = st.columns([1, 1, 3])
c1.download_button("Markdown", md_text, f"neurobridge_{rid}.md", "text/markdown", icon=":material/download:")
pdf_path = run_dir / "export" / "report.pdf"
if pdf_path.exists():
    c2.download_button("PDF", pdf_path.read_bytes(), f"neurobridge_{rid}.pdf", "application/pdf", icon=":material/picture_as_pdf:")
elif c2.button("Build PDF", icon=":material/picture_as_pdf:"):
    with st.spinner("Rendering figures and typesetting the PDF (about 30 s)…"):
        try:
            md = export_markdown(res, view, run_dir / "export")
            markdown_to_pdf(md, pdf_path)
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF export failed: {exc}. The Markdown export is still available.")
c3.caption(f"Run folder: `{run_dir}` (investigation.json, figures/, tables/, report.md, reproducibility.json)")

left, right = st.columns([1.6, 1])
with left:
    with st.container(height=900, border=True):
        st.markdown(md_text)
with right:
    st.markdown("**Key figures**")
    for fid, name in REPORT_FIGURES:
        if fid in view.figures:
            import plotly.graph_objects as go

            f = go.Figure(view.figures[fid]).update_layout(title=None, margin=dict(t=40), height=360)
            st.markdown(f"**{name}**")
            show_fig(f, key=f"rep_{fid}", caption=False)
disclaimer()
