from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ui.common import catalog_frame, disclaimer, eyebrow, kpi, setup

setup()
from neurobridge.agents.orchestrator import DEMO_QUESTION  # noqa: E402
from neurobridge.runs import list_runs  # noqa: E402

left, right = st.columns([1.05, 1], gap="large", vertical_alignment="center")
with left:
    eyebrow("NeuroBridge · agentic research platform")
    st.markdown("<h1 style='font-size:clamp(2.6rem,5.2vw,4.6rem);margin:0 0 1.2rem'>Biology is complex.<br><em>Discovery shouldn't be.</em></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:1.15rem;line-height:1.55;color:#bcb8ae;max-width:34em'>An agentic AI research platform connecting biological hypotheses, multimodal data and reproducible evidence, built around one question: do neurons shape glioblastoma in patients?</p>", unsafe_allow_html=True)
    q = st.text_area("Research question", value=st.session_state.get("question", DEMO_QUESTION), height=96, placeholder="Ask a research question about neuron–glioma biology…")
    c1, c2 = st.columns([1, 1.3])
    if c1.button("Investigate", type="primary", icon=":material/arrow_forward:", width="stretch"):
        st.session_state["question"] = q
        st.session_state["autorun"] = True
        st.switch_page("views/copilot.py")
    if c2.button("Run the demo investigation", icon=":material/science:", width="stretch"):
        st.session_state["question"] = DEMO_QUESTION
        st.session_state["autorun"] = True
        st.switch_page("views/copilot.py")
with right:
    components.html((Path(__file__).resolve().parents[1] / "ui" / "cell_hero.html").read_text(), height=560)

st.write("")
st.markdown("<div class='nb-banner'><b>The question.</b> Can neuronal activity-related molecular signatures identify distinct glioblastoma subgroups, reveal neuron–tumour interactions and explain differences in patient outcomes?</div>", unsafe_allow_html=True)
st.write("")

eyebrow("Example questions")
examples = [
    DEMO_QUESTION,
    "Is NLGN3 or BDNF expression associated with overall survival in IDH-wildtype glioblastoma after adjusting for age and MGMT status?",
    "Which cell types express glutamatergic receptor genes GRIA2 and GRIN2B in glioblastoma, and are they expressed by neoplastic cells?",
    "Where are synaptic receptor genes expressed in glioblastoma tissue relative to neuron-rich regions?",
]
for i, e in enumerate(examples):
    if st.button(e, key=f"ex{i}", type="tertiary", icon=":material/arrow_forward:"):
        st.session_state["question"] = e
        st.session_state["autorun"] = True
        st.switch_page("views/copilot.py")

st.divider()
a, b = st.columns([1.15, 1], gap="large")
with a:
    eyebrow("Data")
    st.subheader("Five modalities, one reproducible workflow")
    from neurobridge.data.catalog import INGESTIBLE

    cat = catalog_frame()
    ing = cat[cat["dataset_id"].isin(INGESTIBLE)]
    k = st.columns(4)
    kpi(k[0], len(ing), "ingestible datasets")
    kpi(k[1], f"{int(ing['n_samples'].fillna(0).sum()):,}", "bulk samples")
    kpi(k[2], "3,589", "single cells")
    kpi(k[3], "3,468", "tissue spots")
    st.write("")
    show = cat[["dataset_id", "source", "modalities", "role", "ingestion"]].copy()
    show["modalities"] = show["modalities"].map(lambda m: ", ".join(m))
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("All data are public. No demonstration or synthetic patient data are used in any analysis.")
with b:
    eyebrow("Recent investigations")
    st.subheader("Your runs")
    runs = list_runs(6)
    if not runs:
        st.info("No investigations yet. Run the demo: about one minute once the public data are cached (the first run downloads them).")
    for r in runs:
        with st.container(border=True):
            st.markdown(f"**{r['question'][:110]}{'…' if len(r['question']) > 110 else ''}**")
            st.caption(f"{r['started_at']} · {r['n_tools']} tool calls · {r['n_claims']} graded claims · {r['status']}")
            if st.button("Open report", key="open_" + r["id"], type="tertiary", icon=":material/description:"):
                st.session_state["report_run"] = r["id"]
                st.switch_page("views/report.py")
disclaimer()
