import pandas as pd
import streamlit as st

from ui.common import claim_card, disclaimer, kpi, setup, show_fig

setup()
from neurobridge.agents.llm import provider_status  # noqa: E402
from neurobridge.agents.orchestrator import DEMO_QUESTION, Investigation  # noqa: E402
from neurobridge.agents.planner import WORKFLOWS  # noqa: E402

st.markdown("<div class='nb-eyebrow'>Module A · Research Copilot</div>", unsafe_allow_html=True)
st.title("Research Copilot")
st.caption("An orchestrator coordinates seven specialised agents. Each agent chooses tools from the registry, and every tool call is logged with its inputs, outputs and limitations. Nothing is simulated: a step either ran, ran in degraded form, or failed and says so.")

ps = provider_status()
with st.sidebar:
    st.markdown("**Run settings**")
    st.caption(f"LLM assistance: {'Claude (' + ps.get('model', '') + ')' if ps['available'] else 'off (deterministic planner and reviewer)'}")
    use_llm = st.toggle("Use LLM for decomposition and review narrative", value=ps["available"], disabled=not ps["available"], help=ps.get("reason") or "LLM output is schema-validated and never produces numbers.")
    override = st.multiselect("Force workflows (optional)", list(WORKFLOWS), help="Leave empty to let the Hypothesis Agent choose from the question.")
    perms = st.select_slider("GSEA permutations", [200, 500, 1000, 2000], value=1000)
    live = st.toggle("Live dataset listings (cBioPortal, CELLxGENE)", value=True)

q = st.text_area("Research question", value=st.session_state.get("question", DEMO_QUESTION), height=80)
run = st.button("Run investigation", type="primary", icon=":material/play_arrow:") or st.session_state.pop("autorun", False)

STAGES = ["Hypothesis Agent", "Literature Agent", "Data Agent", "Bioinformatics Agent", "Statistics Agent", "Validation Agent", "Scientific Reviewer Agent"]

if run:
    st.session_state["question"] = q
    box = st.container(border=True)
    box.markdown("**Workflow progress**")
    stage_slots = {s: box.empty() for s in STAGES}
    for s in STAGES:
        stage_slots[s].markdown(f"<div class='nb-stage'>○ <span class='nb-muted'>{s}</span></div>", unsafe_allow_html=True)
    tool_lines: dict[str, list[str]] = {s: [] for s in STAGES}
    state = {"current": None}

    def render(agent, icon):
        lines = "<br>".join(tool_lines.get(agent, [])[-8:])
        stage_slots[agent].markdown(f"<div class='nb-stage'>{icon} <div><b>{agent}</b><div class='nb-muted' style='font-size:.82rem'>{lines}</div></div></div>", unsafe_allow_html=True)

    def progress(ev):
        k = ev.get("kind")
        ag = ev.get("agent")
        if k == "agent_started" and ag in stage_slots:
            state["current"] = ag
            render(ag, "◐")
        elif k == "agent_finished" and ag in stage_slots:
            render(ag, {"succeeded": "●", "degraded": "◑", "failed": "✕"}.get(ev.get("status"), "●"))
        elif k in ("tool_started", "tool_finished", "tool_failed", "tool_retry") and ag in tool_lines:
            t = ev["tool"]
            if k == "tool_started":
                tool_lines[ag].append(f"▸ {t} running…")
            else:
                msg = {"tool_finished": "✓", "tool_failed": "✕", "tool_retry": "↻"}[k]
                detail = (ev.get("summary") or ev.get("error") or "")[:140]
                if tool_lines[ag] and tool_lines[ag][-1].startswith(f"▸ {t}"):
                    tool_lines[ag][-1] = f"{msg} {t} ({ev.get('duration_s', '')}s): {detail}"
                else:
                    tool_lines[ag].append(f"{msg} {t}: {detail}")
            render(ag, "◐")

    inv = Investigation(q, {"use_llm": use_llm, "workflows": override or None, "gsea_permutations": perms, "live_listing": live}, progress=progress)
    try:
        res = inv.run()
        st.session_state["inv"] = (res, inv.ws)
        st.session_state["report_run"] = res.investigation_id
        st.toast(f"Investigation {res.investigation_id} finished: {res.status}", icon=":material/check_circle:")
    except Exception as exc:  # noqa: BLE001
        st.error(f"The investigation stopped unexpectedly: {type(exc).__name__}: {exc}. Completed steps are listed above; see the execution log for details.")

if "inv" not in st.session_state:
    st.info("Enter a question and run an investigation, or launch the demo from the overview page. A run on cached data takes about one minute; the first run downloads ~80 MB of public data.")
    disclaimer()
    st.stop()

res, ws = st.session_state["inv"]
st.divider()
k = st.columns(5)
kpi(k[0], len(res.datasets_used), "datasets analysed")
kpi(k[1], len(res.log.tool_calls), "tool calls")
kpi(k[2], sum(1 for t in res.log.tool_calls if t.status.value == "failed"), "failed calls")
kpi(k[3], len(res.claims), "graded claims")
kpi(k[4], res.review.get("n_confirmatory_supported", 0), "confirmatory claims supported")
st.caption(f"Investigation `{res.investigation_id}` · status: {res.status} · highest evidence level reached: **{res.review.get('highest_evidence_level_reached')}**")

tabs = st.tabs(["Conclusions", "Plan", "Agents", "Execution log", "Figures", "Literature", "Review"])
with tabs[0]:
    st.markdown("**Which conclusions are supported**")
    order = {"confirmatory": 0, "exploratory": 1, "sensitivity": 2, "interpretation": 3, "literature": 4}
    kind = st.segmented_control("Show", ["all", "confirmatory", "exploratory", "sensitivity", "interpretation", "literature"], default="all")
    for c in sorted(res.claims, key=lambda c: order.get(c["kind"], 9)):
        if kind in (None, "all") or c["kind"] == kind:
            claim_card(c, ws.citations)
    st.markdown("**What remains uncertain**")
    for u in res.review.get("remains_uncertain", []):
        st.markdown(f"- {u}")
with tabs[1]:
    p = res.plan
    st.markdown(f"**Restated question.** {p.restated_question}")
    st.markdown(f"Planner: `{p.planner}` · workflows: " + ", ".join(f"`{w}`" for w in p.requested_workflows))
    st.markdown("**Sub-questions**")
    for s in p.sub_questions:
        st.markdown(f"1. {s}")
    st.markdown("**Hypotheses (pre-specified)**")
    st.dataframe(pd.DataFrame(p.hypotheses), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown("**Candidate genes**: " + ", ".join(p.candidate_genes))
    c1.markdown("**Gene sets**: " + ", ".join(p.gene_set_ids))
    c2.markdown("**Confounders considered**: " + ", ".join(p.confounders))
    c2.markdown(f"**Evidence ceiling of this design**: {p.evidence_ceiling}")
    for n in p.notes:
        st.caption(n)
with tabs[2]:
    for m in res.log.messages:
        icon = {"succeeded": ":material/check_circle:", "degraded": ":material/warning:", "failed": ":material/error:"}[m.status.value] if m.status.value in ("succeeded", "degraded", "failed") else None
        with st.expander(f"{m.agent} · {m.stage} · {m.status.value}", icon=icon):
            c1, c2 = st.columns(2)
            c1.markdown("**Inputs**")
            c1.json(m.inputs, expanded=False)
            c1.markdown("**Actions (tool calls)**")
            for a in m.actions:
                c1.markdown(f"- `{a[:160]}`")
            c2.markdown("**Outputs**")
            c2.json(m.outputs, expanded=False)
            c2.markdown("**Limitations reported**")
            for l in m.limitations:
                c2.markdown(f"- {l}")
with tabs[3]:
    df = pd.DataFrame([{"agent": t.agent, "tool": t.tool, "status": t.status.value, "attempts": t.attempts, "duration_s": t.duration_s, "summary": t.output_summary or t.error, "outputs": t.output_ref, "started": t.started_at} for t in res.log.tool_calls])
    st.dataframe(df, hide_index=True, width="stretch", column_config={"summary": st.column_config.TextColumn(width="large")})
    sel = st.selectbox("Inspect a call", [f"{i}: {t.tool} ({t.agent})" for i, t in enumerate(res.log.tool_calls)])
    if sel:
        t = res.log.tool_calls[int(sel.split(":")[0])]
        st.json(t.model_dump(mode="json"), expanded=False)
with tabs[4]:
    figs = list(ws.figures)
    pick = st.multiselect("Figures", figs, default=[f for f in figs if f.startswith(("validation_", "groups_neuronal", "composition_tcga", "dotplot"))][:5])
    for f in pick:
        show_fig(ws.figures[f], key=f"cop_{f}")
with tabs[5]:
    cits = list(ws.citations.values())
    key_refs = [c for c in cits if c.relevance and not c.relevance.startswith("Retrieved by query")]
    other = [c for c in cits if c not in key_refs]
    st.markdown(f"**{len(key_refs)} curated key references** (PMIDs verified against Europe PMC; metadata retrieved live)")
    st.dataframe(pd.DataFrame([{"PubMed": c.url, "reference": c.short(), "title": c.title, "why it matters": c.relevance, "cited by": c.cited_by} for c in key_refs]), hide_index=True, width="stretch",
                 column_config={"PubMed": st.column_config.LinkColumn(display_text=r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")})
    st.markdown(f"**{len(other)} additional records from live queries** (relevance-ranked leads, not screened)")
    st.dataframe(pd.DataFrame([{"PMID": c.pmid, "reference": c.short(), "title": c.title, "query": (c.relevance or "").replace("Retrieved by query: ", "")} for c in other]), hide_index=True, width="stretch")
with tabs[6]:
    rv = res.review
    st.markdown(f"**Highest evidence level reached:** {rv.get('highest_evidence_level_reached')}  \n**Design ceiling:** {rv.get('evidence_ceiling_of_design')}")
    st.dataframe(pd.DataFrame(rv.get("issues", [])), hide_index=True, width="stretch")
    if rv.get("narrative"):
        st.markdown(f"**Narrative** ({rv['narrative_source']})")
        st.write(rv["narrative"]["overall_assessment"])
disclaimer()
