"""End-to-end investigation on cached public data with the network disabled."""

import pytest

from neurobridge import config
from tests.conftest import requires_cgga, requires_sc, requires_tcga


@pytest.fixture
def offline(monkeypatch, tmp_path):
    s = config.Settings(allow_network=False, runs_dir=tmp_path / "runs")
    s.ensure_dirs()
    import neurobridge.config as cfg

    cfg.get_settings.cache_clear()
    monkeypatch.setattr(cfg, "get_settings", lambda: s)
    for mod in ["neurobridge.data.cache", "neurobridge.literature.europepmc", "neurobridge.agents.orchestrator", "neurobridge.knowledge.genesets", "neurobridge.data.catalog", "neurobridge.analysis.common", "neurobridge.data.cbioportal", "neurobridge.data.cellxgene", "neurobridge.agents.llm"]:
        m = __import__(mod, fromlist=["x"])
        if hasattr(m, "get_settings"):
            monkeypatch.setattr(m, "get_settings", lambda: s)
    return s


@requires_tcga
@requires_cgga
@requires_sc
@pytest.mark.skipif(not __import__("tests.conftest", fromlist=["cached"]).cached("spatial/visium_gbm_10x/filtered_feature_bc_matrix.h5"), reason="Visium not cached")
@pytest.mark.slow
def test_full_investigation_offline(offline):
    from neurobridge.agents.orchestrator import DEMO_QUESTION, run_investigation

    res, inv = run_investigation(DEMO_QUESTION, use_llm=False, live_listing=False, gsea_permutations=200)
    tools = {t.tool: t.status.value for t in res.log.tool_calls}
    assert tools["perform_survival_analysis"] == "succeeded"
    assert tools["validate_across_cohorts"] == "succeeded"
    assert tools["perform_cell_type_analysis"] == "succeeded"
    ids = {c["id"] for c in res.claims}
    assert {"S1", "S2", "C1", "O2", "SC1", "SP1", "SP3"} <= ids
    assert tools["perform_spatial_analysis"] == "succeeded"
    for c in res.claims:
        if c["kind"] in ("exploratory", "confirmatory", "sensitivity"):
            assert c["analyses"], c["id"]
        if c["kind"] == "literature":
            assert c["citations"]
    md = res.report_markdown
    for i, sec in enumerate(["Executive summary", "Research question", "Biological rationale", "Datasets", "Methods", "Results", "Statistical evidence", "Cross-dataset validation", "Biological interpretation", "Alternative explanations", "Limitations", "Proposed follow-up experiments", "References"], 1):
        assert f"## {i}. {sec}" in md
    assert (offline.runs_dir / res.investigation_id / "reproducibility.json").exists()


@requires_tcga
def test_single_cohort_investigation_is_never_called_validated(offline):
    from neurobridge.agents.orchestrator import run_investigation

    res, _ = run_investigation("Is the neuronal score associated with survival?", use_llm=False, live_listing=False, datasets=["tcga_gbm_rnaseq"], workflows=["survival"])
    o2 = next(c for c in res.claims if c["id"] == "O2")
    assert "not validated" in o2["text"]
    assert any("only one cohort" in i["issue"] for i in res.review["issues"])
    assert res.review["highest_evidence_level_reached"] != "statistical association" or all(c["status"] != "supported" for c in res.claims if c["kind"] == "confirmatory")


@requires_tcga
def test_missing_dataset_is_reported_not_simulated(offline, monkeypatch):
    from neurobridge.agents.orchestrator import run_investigation
    from neurobridge.data import loaders

    real = loaders.load_cohort

    def broken(did):
        if did == "cgga_693":
            raise RuntimeError("source unavailable")
        return real(did)

    monkeypatch.setattr("neurobridge.agents.tools.builtin.load_cohort", broken)
    res, _ = run_investigation("Is the neuronal score associated with survival?", use_llm=False, live_listing=False, datasets=["tcga_gbm_rnaseq", "cgga_693"], workflows=["survival"])
    assert any(u["dataset_id"] == "cgga_693" for u in res.datasets_unavailable)
    assert res.status == "completed with failures"
    assert "cgga_693" not in res.datasets_used
