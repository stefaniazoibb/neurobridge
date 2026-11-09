"""Agent layer: registry, executor (validation, retry, failure), planner, reviewer rules, literature behaviour."""

import pytest
from pydantic import BaseModel

from neurobridge.agents.executor import ToolExecutor
from neurobridge.agents.tools import REGISTRY
from neurobridge.agents.tools.registry import Tool, ToolOutput, TransientToolError
from neurobridge.agents.workspace import Workspace
from neurobridge.provenance import utc_now_iso
from neurobridge.schemas.agents import ExecutionLog, ToolStatus

REQUIRED_TOOLS = {"search_literature", "discover_datasets", "load_expression_data", "inspect_clinical_metadata", "perform_differential_expression", "perform_pathway_enrichment",
                  "perform_survival_analysis", "perform_cell_type_analysis", "validate_across_cohorts", "generate_scientific_report"}


def test_registry_contains_required_tools_and_mcp_manifest():
    assert REQUIRED_TOOLS <= set(REGISTRY.names())
    for t in REGISTRY.mcp_manifest():
        assert {"name", "description", "inputSchema"} <= set(t)
        assert t["inputSchema"]["type"] == "object"


class _In(BaseModel):
    x: int


def _executor():
    ws = Workspace("test")
    return ToolExecutor(ws, ExecutionLog(investigation_id="test", started_at=utc_now_iso()), backoff_s=0), ws


def test_executor_retries_transient_then_succeeds():
    calls = {"n": 0}

    def fn(ws, inp):
        calls["n"] += 1
        if calls["n"] < 2:
            raise TransientToolError("flaky network")
        return ToolOutput(summary=f"ok {inp.x}")

    REGISTRY._tools["_flaky"] = Tool("_flaky", "test", _In, fn, "test", retryable=True)
    ex, _ = _executor()
    out = ex.run("_flaky", "tester", x=3)
    assert out.summary == "ok 3" and ex.log.tool_calls[-1].attempts == 2 and ex.log.tool_calls[-1].status == ToolStatus.SUCCEEDED
    del REGISTRY._tools["_flaky"]


def test_executor_records_failure_without_raising():
    def fn(ws, inp):
        raise ValueError("boom")

    REGISTRY._tools["_bad"] = Tool("_bad", "test", _In, fn, "test")
    ex, _ = _executor()
    assert ex.run("_bad", "tester", x=1) is None
    assert ex.log.tool_calls[-1].status == ToolStatus.FAILED and "boom" in ex.log.tool_calls[-1].error
    assert ex.run("_bad", "tester", x="not an int") is None
    assert "validation" in ex.log.tool_calls[-1].error
    del REGISTRY._tools["_bad"]


def test_planner_detects_workflows_and_genes():
    from neurobridge.agents.planner import deterministic_plan

    p = deterministic_plan("Is BDNF associated with overall survival in IDH-wildtype GBM?")
    assert "survival" in p.requested_workflows and "composition" in p.requested_workflows
    p2 = deterministic_plan("Which pathways differ? Look at FAKEGENE1 and GRIA2.")
    assert "differential" in p2.requested_workflows
    assert any("FAKEGENE1" in n for n in p2.notes)
    assert "statistical association" in p.evidence_ceiling


def test_reviewer_flags_causal_language_and_single_cohort():
    from neurobridge.agents.agents import Context
    from neurobridge.agents.planner import deterministic_plan
    from neurobridge.agents.reviewer import ReviewerAgent

    ex, ws = _executor()
    ws.claims = [{"id": "X", "text": "High score drives poor survival.", "level": "statistical association", "kind": "confirmatory", "analyses": ["a"], "citations": [], "caveats": [], "status": "supported"},
                 {"id": "L", "text": "Literature says so.", "level": "mechanistic evidence", "kind": "literature", "analyses": [], "citations": [], "caveats": [], "status": "supported"}]
    ctx = Context(question="q", plan=deterministic_plan("survival"), ws=ws, ex=ex, log=ex.log, options={"use_llm": False})
    ReviewerAgent().run(ctx, {})
    issues = ctx.outputs["review"]["issues"]
    assert any("causal verb" in i["issue"] for i in issues)
    assert any("only one cohort" in i["issue"] for i in issues)
    assert ws.claims[0]["status"] == "not validated (single cohort)"
    assert ws.claims[1]["status"] == "removed"


def test_literature_never_fabricates_when_offline(monkeypatch, tmp_path):
    from neurobridge import config
    from neurobridge.literature import europepmc

    s = config.Settings(cache_dir=tmp_path, allow_network=False)
    monkeypatch.setattr(europepmc, "get_settings", lambda: s)
    with pytest.raises(europepmc.LiteratureUnavailable):
        europepmc.search("an uncached query about glioma")


def test_llm_disabled_by_default():
    from neurobridge.agents.llm import LLMUnavailable, provider_status, structured

    if provider_status()["provider"] == "none":
        with pytest.raises(LLMUnavailable):
            structured("s", "p", _In)


def test_planner_gene_level_and_idh_restriction():
    from neurobridge.agents.planner import deterministic_plan

    p = deterministic_plan("Is NLGN3 or BDNF expression associated with overall survival in IDH-wildtype glioblastoma?")
    assert p.restrictions == ["IDH-wildtype"]
    assert p.gene_level == ["NLGN3", "BDNF"] and "user_genes" in p.gene_set_ids
    p2 = deterministic_plan("Is neuronal signalling associated with survival?")
    assert p2.restrictions == [] and p2.gene_level == []
