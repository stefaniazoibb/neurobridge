import streamlit as st

from ui.common import TAGLINE, catalog_frame, disclaimer, kpi, setup

setup()
from neurobridge.agents.orchestrator import DEMO_QUESTION  # noqa: E402
from neurobridge.runs import list_runs  # noqa: E402

st.markdown("<div class='nb-eyebrow'>Agentic research platform · glioblastoma</div>", unsafe_allow_html=True)
st.title("NeuroBridge")
st.markdown(f"#### {TAGLINE}")
st.markdown(
    "NeuroBridge investigates **neuron–glioma interactions in glioblastoma**. A team of specialised agents turns a research question "
    "into hypotheses, selects public cohorts, runs real statistical and single-cell analyses, tests replication in independent cohorts, "
    "and writes an evidence report in which every claim is graded and linked to the analysis or paper behind it."
)

st.markdown("<div class='nb-banner'><b>Central question.</b> Can neuronal activity-related molecular signatures identify distinct glioblastoma subgroups, "
            "reveal neuron–tumour interactions and explain differences in patient outcomes?</div>", unsafe_allow_html=True)
st.write("")

q = st.text_area("Research question", value=st.session_state.get("question", DEMO_QUESTION), height=90, label_visibility="collapsed", placeholder="Ask a research question about neuron–glioma biology…")
c1, c2, _ = st.columns([1.2, 1.4, 3])
if c1.button("Investigate", type="primary", icon=":material/play_arrow:"):
    st.session_state["question"] = q
    st.session_state["autorun"] = True
    st.switch_page("views/copilot.py")
if c2.button("Run the demo investigation", icon=":material/science:"):
    st.session_state["question"] = DEMO_QUESTION
    st.session_state["autorun"] = True
    st.switch_page("views/copilot.py")

st.markdown("**Example questions**")
examples = [
    DEMO_QUESTION,
    "Is NLGN3 or BDNF expression associated with overall survival in IDH-wildtype glioblastoma after adjusting for age and MGMT status?",
    "Which cell types express glutamatergic receptor genes GRIA2 and GRIN2B in glioblastoma, and are they expressed by neoplastic cells?",
    "Which pathways distinguish glioblastomas with high versus low synaptic gene expression?",
]
for i, e in enumerate(examples):
    if st.button(e, key=f"ex{i}", type="tertiary", icon=":material/arrow_forward:"):
        st.session_state["question"] = e
        st.session_state["autorun"] = True
        st.switch_page("views/copilot.py")

st.divider()
left, right = st.columns([1.15, 1])
with left:
    st.subheader("Data availability")
    cat = catalog_frame()
    from neurobridge.data.catalog import INGESTIBLE

    ing = cat[cat["dataset_id"].isin(INGESTIBLE)]
    k = st.columns(4)
    kpi(k[0], len(ing), "ingestible datasets")
    kpi(k[1], int(ing["n_samples"].fillna(0).sum()), "bulk samples (all grades, ingestible)")
    kpi(k[2], int(ing["n_cells"].fillna(0).sum()), "single cells")
    kpi(k[3], int(ing["survival_available"].sum()), "cohorts with survival")
    st.write("")
    show = cat[["dataset_id", "source", "modalities", "role", "ingestion"]].copy()
    show["modalities"] = show["modalities"].map(lambda m: ", ".join(m))
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("All data are public. No demonstration or synthetic patient data are used in any analysis.")
with right:
    st.subheader("Recent investigations")
    runs = list_runs(8)
    if not runs:
        st.info("No investigations yet. Launch the demo to create one (about one minute on cached data).")
    for r in runs:
        with st.container(border=True):
            st.markdown(f"**{r['question'][:110]}{'…' if len(r['question']) > 110 else ''}**")
            st.caption(f"{r['started_at']} · {r['n_tools']} tool calls · {r['n_claims']} graded claims · {r['status']}")
            if st.button("Open report", key="open_" + r["id"], type="tertiary", icon=":material/description:"):
                st.session_state["report_run"] = r["id"]
                st.switch_page("views/report.py")
disclaimer()
