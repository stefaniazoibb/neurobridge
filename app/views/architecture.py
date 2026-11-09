import json

import pandas as pd
import streamlit as st

from ui.common import disclaimer, setup

setup()
from neurobridge.agents.llm import provider_status  # noqa: E402
from neurobridge.agents.tools import REGISTRY  # noqa: E402

st.markdown("<div class='nb-eyebrow'>System · Architecture</div>", unsafe_allow_html=True)
st.title("Architecture & tools")
st.markdown("""
NeuroBridge is a **transparent orchestration layer**: no agent framework, no hidden prompts. The orchestrator runs specialised agents in dependency order;
each agent calls **typed tools** from a registry through an executor that validates inputs (pydantic), retries transient failures, and logs every call.
Analyses return typed results that carry an **analysis record** (method, parameters, seed, software, timestamps, n used, limitations).
Claims are generated from those results with fixed templates, graded by evidence level, and audited by the reviewer agent.
""")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; fontcolor="#bcb8ae"; node [shape=box, style="rounded,filled", fillcolor="#0d1322", color="#43c9b4", fontname="Helvetica", fontsize=11, fontcolor="#ece7dc"]; edge [color="#868791"];
  Q [label="Research question", fillcolor="#080c17"];
  O [label="Orchestrator\\n(execution log, retries,\\npartial failure)", fillcolor="#123236"];
  H [label="Hypothesis Agent"]; L [label="Literature Agent"]; D [label="Data Agent"]; B [label="Bioinformatics Agent"]; S [label="Statistics Agent"]; V [label="Validation Agent"]; R [label="Scientific Reviewer"];
  T [label="Tool registry\\n(MCP-shaped schemas)", shape=folder, fillcolor="#0d1322"];
  W [label="Workspace\\n(cohorts, results,\\nfigures, citations)", shape=cylinder, fillcolor="#0d1322"];
  C [label="Claims\\n(evidence-graded)", fillcolor="#080c17"]; RP [label="Evidence report\\n(MD / PDF)", fillcolor="#080c17"];
  Q -> O; O -> H -> L -> D -> B -> S -> V -> C -> R -> RP;
  {L D B S V} -> T [style=dashed]; T -> W [style=dashed];
  subgraph cluster_data { label="Public data"; fontname="Helvetica"; fontcolor="#bcb8ae"; color="#3b4356"; X [label="UCSC Xena (TCGA)"]; G [label="CGGA"]; E [label="GEO GSE84465"]; P [label="Europe PMC"]; CB [label="cBioPortal / CELLxGENE"]; }
  X -> D; G -> D; E -> D; CB -> D; P -> L;
}
""")
ps = provider_status()
st.markdown(f"**LLM integration:** {'enabled, ' + ps['model'] if ps['available'] else 'off'} · {ps.get('reason', '')}. The LLM (Anthropic Claude, configurable) is only used for question decomposition and the reviewer narrative; outputs are schema-validated and it never computes or reports numbers.")
st.subheader("Tool registry")
df = pd.DataFrame([{"tool": t.name, "category": t.category, "retryable": t.retryable, "description": t.description} for t in REGISTRY.all()])
st.dataframe(df, hide_index=True, width="stretch", column_config={"description": st.column_config.TextColumn(width="large")})
t = st.selectbox("Input schema", REGISTRY.names())
st.json(REGISTRY.get(t).schema(), expanded=False)
st.download_button("Download MCP tool manifest (JSON)", json.dumps(REGISTRY.mcp_manifest(), indent=2), "neurobridge_tools.mcp.json", "application/json")
st.caption("The manifest follows the Model Context Protocol tool shape (name, description, inputSchema), so the same skills can be served by an MCP server or handed to an LLM tool-use loop without rewriting.")
disclaimer()
