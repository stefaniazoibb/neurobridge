"""Sensitivity analyses: does the association survive alternative specifications and a composition control?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from neurobridge.analysis.signatures import dichotomise, score_signature
from neurobridge.analysis.survival import cox_model, exposure_row, kaplan_meier
from neurobridge.data.cohort import Cohort


def residualise(y: pd.Series, x: pd.Series) -> pd.Series:
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    X = sm.add_constant(df["x"])
    res = sm.OLS(df["y"], X).fit()
    return pd.Series(res.resid, index=df.index)


def survival_sensitivity(cohort: Cohort, genes: list[str], name: str, control_genes: list[str], covariates: list[str]) -> list[dict]:
    """Re-estimate the survival association under alternative choices. Each row is one specification."""
    c = cohort.primary_tumours()
    did = cohort.dataset.dataset_id
    s = pd.Series(score_signature(c, genes, name).scores)
    s_rank = pd.Series(score_signature(c, genes, name, method="rank").scores)
    ctrl = pd.Series(score_signature(c, control_genes, "neuron_markers").scores)
    cov_here = [cv for cv in covariates if cv in c.clinical.columns and c.clinical[cv].notna().mean() >= 0.5]
    rows: list[dict] = []

    def add(spec: str, cx, extra: dict | None = None):
        r = exposure_row(cx, name if name in [x.covariate for x in cx.rows] else cx.rows[0].covariate)
        rows.append({"cohort": did, "specification": spec, "n": cx.n, "events": cx.events, "hr": r.hr, "ci_low": r.hr_ci_low, "ci_high": r.hr_ci_high, "p": r.p, "ph_ok": cx.ph_assumption_ok, "covariates": ", ".join(cx.covariates_adjusted) or "none", **(extra or {})})

    add("primary: mean-z score, adjusted", cox_model(c.clinical, s, name, did, cov_here))
    add("unadjusted", cox_model(c.clinical, s, name, did, []))
    add("rank-based score, adjusted", cox_model(c.clinical, s_rank, name, did, cov_here))
    clin2 = c.clinical.copy()
    clin2["neuron_marker_score"] = ctrl
    add("adjusted + neuron-marker (composition) score", cox_model(clin2, s, name, did, cov_here + ["neuron_marker_score"]))
    resid = residualise(s, ctrl)
    add("candidate score residualised on neuron markers, adjusted", cox_model(c.clinical, resid, name, did, cov_here))
    if "idh_status" in c.clinical and c.clinical["idh_status"].notna().any():
        wt = c.clinical.index[c.clinical["idh_status"] == "Wildtype"]
        cw = c.subset(wt)
        sw = pd.Series(score_signature(cw, genes, name).scores)
        add("IDH-wildtype only, adjusted", cox_model(cw.clinical, sw, name, did, [v for v in cov_here if v != "idh_status"]))
    if "g_cimp" in c.clinical and c.clinical["g_cimp"].notna().any():
        nc = c.clinical.index[c.clinical["g_cimp"] == "non-G-CIMP"]
        cn = c.subset(nc)
        sn = pd.Series(score_signature(cn, genes, name).scores)
        add("non-G-CIMP only, adjusted", cox_model(cn.clinical, sn, name, did, [v for v in cov_here if v != "g_cimp"]))
    if "subtype" in c.clinical and c.clinical["subtype"].notna().any():
        nn = c.clinical.index[c.clinical["subtype"] != "Neural"]
        cn = c.subset(nn)
        sn = pd.Series(score_signature(cn, genes, name).scores)
        add("Neural subtype excluded, adjusted", cox_model(cn.clinical, sn, name, did, cov_here))
    km = kaplan_meier(c.clinical, dichotomise(s, "tertiles"), "tertiles", did)
    rows.append({"cohort": did, "specification": "top vs bottom tertile (log-rank)", "n": km.n_total, "events": km.n_events, "hr": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": km.logrank_p, "ph_ok": None, "covariates": "none"})
    return rows


def composition_check(cohort: Cohort, genes: list[str], control_genes: list[str], grouping: str = "subtype") -> dict:
    """How much of the candidate score is explained by mature-neuron marker expression, and does the group
    difference survive residualisation on it?"""
    c = cohort.primary_tumours()
    s = pd.Series(score_signature(c, genes, "candidate").scores)
    ctrl = pd.Series(score_signature(c, control_genes, "neuron_markers").scores).reindex(s.index)
    rho, p = stats.spearmanr(s, ctrl)
    r2 = float(np.corrcoef(s, ctrl)[0, 1] ** 2)
    out = {"cohort": cohort.dataset.dataset_id, "spearman_rho": float(rho), "spearman_p": float(p), "r2_linear": r2, "n": int(len(s))}
    if grouping in c.clinical and c.clinical[grouping].notna().sum() > 10:
        g = c.clinical[grouping]
        resid = residualise(s, ctrl)
        groups_raw = [s[g == k].dropna().values for k in g.dropna().unique() if (g == k).sum() >= 5]
        groups_res = [resid.reindex(g.index[g == k]).dropna().values for k in g.dropna().unique() if (g == k).sum() >= 5]
        out["kw_p_raw"] = float(stats.kruskal(*groups_raw).pvalue)
        out["kw_p_residualised"] = float(stats.kruskal(*groups_res).pvalue)
        out["residual_medians"] = {k: float(np.median(resid.reindex(g.index[g == k]).dropna())) for k in g.dropna().unique() if (g == k).sum() >= 5}
    return out
