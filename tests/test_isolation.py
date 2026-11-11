"""Investigation isolation: concurrent investigations in one process must not see each other's state.

The Streamlit app serves every visitor from one Python process, so anything stored at module level
(gene panels) or in an in-process cache (loaded cohorts) is shared by all simultaneous users. These tests pin
down that per-investigation state stays inside the investigation's Workspace.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from neurobridge.agents.agents import Context, HypothesisAgent
from neurobridge.agents.executor import ToolExecutor
from neurobridge.agents.tools import builtin
from neurobridge.agents.workspace import Workspace
from neurobridge.knowledge.genesets import PANELS, all_panels
from neurobridge.schemas.agents import ExecutionLog
from tests.conftest import make_synthetic_cohort, requires_cgga, requires_tcga

# Genes from the candidate / extended panels, so the planner accepts them as measured symbols.
GENE_SETS = [
    ["NLGN3", "BDNF"],
    ["GRIA2"],
    ["GRIN2B", "GRIA1"],
    ["SLC17A7"],
    ["NLGN3", "GRIA2", "GRIN2B"],
    ["BDNF", "GRIA1"],
    ["GRIA1"],
    ["SLC17A7", "BDNF"],
]


def _cohort_with(dataset_id: str = "synthetic_iso"):
    """Synthetic cohort whose first genes are renamed to real symbols so panels resolve against it."""
    c = make_synthetic_cohort(n=80, dataset_id=dataset_id)
    names = sorted({g for gs in GENE_SETS for g in gs})
    mapping = {f"BG{i}": g for i, g in enumerate(names)}
    c.expression = c.expression.rename(index=mapping)
    return c


def _new_context(question: str) -> Context:
    ws = Workspace(f"test-{abs(hash(question)) % 10**8}")
    log = ExecutionLog(investigation_id=ws.investigation_id, started_at="t0")
    return Context(question=question, plan=None, ws=ws, ex=ToolExecutor(ws, log), log=log, options={"use_llm": False, "workflows": ["subtype"]})


def _plan(question: str) -> Context:
    ctx = _new_context(question)
    HypothesisAgent().run(ctx, {"question": question})
    assert ctx.plan is not None
    return ctx


def _scored_genes(ctx: Context, panel_id: str) -> list[str]:
    out = ctx.ex.run("score_signature", "test", dataset_id="synthetic_iso", panel_id=panel_id)
    assert out is not None, ctx.log.tool_calls[-1].error
    return ctx.ws.artefacts[out.refs[0]].genes_requested


def _question(genes: list[str]) -> str:
    return f"Is {' and '.join(genes)} expression associated with glioblastoma subtype?"


def test_planning_does_not_touch_global_panels():
    before = set(PANELS)
    ctx = _plan(_question(["NLGN3", "BDNF"]))
    assert set(PANELS) == before
    assert "user_genes" not in all_panels(include_pathways=False)
    assert ctx.ws.panels["user_genes"].genes == ["NLGN3", "BDNF"]


def test_builtin_panels_are_read_only():
    """Guard against reintroducing the bug: any run-time write to the shared registry must fail loudly."""
    with pytest.raises(TypeError):
        PANELS["user_genes"] = PANELS["neuronal_activity_candidates"]  # type: ignore[index]


def test_interleaved_investigations_keep_their_own_genes():
    """Deterministic version of the race: A plans, B plans, then A scores. A must score A's genes."""
    a = _plan(_question(["NLGN3", "BDNF"]))
    b = _plan(_question(["GRIA2"]))
    for ctx in (a, b):
        ctx.ws.cohorts["synthetic_iso"] = _cohort_with()
    assert _scored_genes(a, "user_genes") == ["NLGN3", "BDNF"]
    assert _scored_genes(b, "user_genes") == ["GRIA2"]
    assert _scored_genes(b, "gene_GRIA2") == ["GRIA2"]
    # A never asked for GRIA2 as a single gene, so that panel must not exist for A
    out = a.ex.run("score_signature", "test", dataset_id="synthetic_iso", panel_id="gene_GRIA2")
    assert out is None and "unknown gene panel" in a.log.tool_calls[-1].error


def test_simultaneous_investigations_in_threads():
    """Eight 'users' plan at the same time, wait for each other, then score: no cross-contamination."""
    barrier = threading.Barrier(len(GENE_SETS))

    def user(genes: list[str]) -> tuple[list[str], list[str]]:
        ctx = _plan(_question(genes))
        ctx.ws.cohorts["synthetic_iso"] = _cohort_with()
        barrier.wait(timeout=30)  # every user has planned before anyone scores
        return genes, _scored_genes(ctx, "user_genes")

    for _ in range(3):
        with ThreadPoolExecutor(max_workers=len(GENE_SETS)) as pool:
            results = list(pool.map(user, GENE_SETS))
        for asked, scored in results:
            assert scored == asked


def test_cached_cohort_is_not_mutated_by_an_investigation(monkeypatch):
    """load_cohort is lru_cached: the object it returns is shared by every investigation and by the app's views."""
    shared_train = make_synthetic_cohort(n=120, dataset_id="tcga_gbm_rnaseq", seed=1)
    shared_target = make_synthetic_cohort(n=90, dataset_id="cgga_325", seed=2)
    shared_target.clinical["subtype"] = np.nan  # validation cohorts have no labels
    cache = {"tcga_gbm_rnaseq": shared_train, "cgga_325": shared_target}
    monkeypatch.setattr(builtin, "load_cohort", lambda did: cache[did])
    clin_before = {k: v.clinical.copy() for k, v in cache.items()}
    expr_before = {k: v.expression.copy() for k, v in cache.items()}

    ctx = _new_context("subtype question")
    for did in cache:
        assert ctx.ex.run("load_expression_data", "test", dataset_id=did) is not None
    assert ctx.ex.run("score_signature", "test", dataset_id="tcga_gbm_rnaseq", panel_id="neuronal_activity_candidates") is not None
    out = ctx.ex.run("project_subtypes", "test", train_id="tcga_gbm_rnaseq", target_ids=["cgga_325"])
    assert out is not None, ctx.log.tool_calls[-1].error

    # the investigation's own copy carries the projection ...
    assert "subtype_projected" in ctx.ws.cohorts["cgga_325"].clinical
    # ... the shared cached objects do not
    for did, c in cache.items():
        pd.testing.assert_frame_equal(c.clinical, clin_before[did])
        pd.testing.assert_frame_equal(c.expression, expr_before[did])  # shared matrix is treated as read-only
    # and a second investigation starts clean
    ctx2 = _new_context("another question")
    ctx2.ex.run("load_expression_data", "test", dataset_id="cgga_325")
    assert "subtype_projected" not in ctx2.ws.cohorts["cgga_325"].clinical


@requires_tcga
@requires_cgga
@pytest.mark.slow
def test_two_real_investigations_concurrently(offline):
    """Two users on the real cached cohorts at the same time, asking about different genes (network disabled)."""
    from neurobridge.agents.orchestrator import Investigation

    opts = {"use_llm": False, "live_listing": False, "persist": False, "workflows": ["subtype"],
            "datasets": ["tcga_gbm_rnaseq", "cgga_325"]}
    questions = {"A": _question(["NLGN3", "BDNF"]), "B": _question(["GRIA2"])}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {k: pool.submit(lambda q: Investigation(q, dict(opts)).run(), q) for k, q in questions.items()}
        res = {k: f.result() for k, f in futs.items()}
    for key, genes in (("A", ["NLGN3", "BDNF"]), ("B", ["GRIA2"])):
        r = res[key]
        assert r.plan.custom_panels["user_genes"] == genes
        scored = [v for k, v in r.artefacts.items() if k.startswith("signature:") and v.get("signature") == "user_genes"]
        assert scored, f"{key}: no user_genes score recorded"
        assert all(v["genes_requested"] == genes for v in scored)
