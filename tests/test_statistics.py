"""Unit tests of the statistical core against known answers."""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from neurobridge.analysis.common import bh_adjust, cliffs_delta, hedges_g
from neurobridge.analysis.correlation import gene_correlation
from neurobridge.analysis.differential import differential_expression
from neurobridge.analysis.groups import compare_groups
from neurobridge.analysis.signatures import dichotomise, score_signature
from neurobridge.analysis.survival import cox_model, exposure_row, kaplan_meier
from neurobridge.analysis.validation import meta_analyse_log_hr, validate_group_difference, validate_signature_survival
from tests.conftest import make_synthetic_cohort


def test_bh_matches_statsmodels():
    p = np.array([0.001, 0.01, 0.02, 0.04, 0.2, 0.5, np.nan])
    q = bh_adjust(p)
    ref = multipletests(p[:-1], method="fdr_bh")[1]
    assert np.allclose(q[:-1], ref) and np.isnan(q[-1])


def test_hedges_g_known_value():
    a = np.array([2.0, 4.0, 6.0, 8.0])
    b = np.array([1.0, 3.0, 5.0, 7.0])
    g, lo, hi = hedges_g(a, b)
    d = 1.0 / np.std(a, ddof=1)
    assert np.isclose(g, d * (1 - 3 / (4 * 8 - 9)))
    assert lo < g < hi


def test_cliffs_delta_extremes():
    assert cliffs_delta(np.array([5, 6]), np.array([1, 2])) == 1.0
    assert cliffs_delta(np.array([1, 2]), np.array([1, 2])) == 0.0


def test_meta_analysis_homogeneous():
    m = meta_analyse_log_hr(np.array([0.2, 0.2, 0.2]), np.array([0.1, 0.1, 0.1]))
    assert np.isclose(m["fixed_log_hr"], 0.2) and np.isclose(m["random_log_hr"], 0.2)
    assert m["I2"] == 0.0 and np.isclose(m["fixed_se"], 0.1 / np.sqrt(3))


def test_meta_analysis_heterogeneous():
    m = meta_analyse_log_hr(np.array([-0.5, 0.5]), np.array([0.1, 0.1]))
    assert m["I2"] > 90 and m["random_se"] > m["fixed_se"]


def test_signature_score_and_missing_genes(synth):
    r = score_signature(synth, ["G0", "G1", "NOT_A_GENE"], "x")
    assert r.genes_missing == ["NOT_A_GENE"] and len(r.scores) == 200
    s = pd.Series(r.scores)
    assert abs(s.mean()) < 1e-6
    assert any("absent" in w for w in r.record.warnings)


def test_dichotomise():
    s = pd.Series(np.arange(9.0))
    assert dichotomise(s, "median").value_counts().to_dict() == {"high": 5, "low": 4}
    t = dichotomise(s, "tertiles")
    assert len(t) == 6 and set(t) == {"high", "low"}


def test_cox_recovers_simulated_hazard_ratio():
    c = make_synthetic_cohort(n=1500, beta=0.5, seed=1)
    s = pd.Series(score_signature(c, [f"G{i}" for i in range(5)], "sig").scores)
    res = cox_model(c.clinical, s, "sig", "synthetic")
    r = exposure_row(res, "sig")
    # score is a noisy proxy of the latent factor (attenuation), so expect HR clearly > 1 and below exp(0.5)
    assert 1.3 < r.hr < np.exp(0.55) and r.p < 1e-10
    assert res.ph_assumption_ok is not None


def test_cox_null_effect_not_significant():
    c = make_synthetic_cohort(n=400, beta=0.0, seed=2)
    s = pd.Series(score_signature(c, [f"G{i}" for i in range(5)], "sig").scores)
    r = exposure_row(cox_model(c.clinical, s, "sig", "synthetic", ["age", "sex"]), "sig")
    assert r.hr_ci_low < 1 < r.hr_ci_high


def test_cox_reference_levels_and_constant_covariates(synth):
    s = pd.Series(score_signature(synth, ["G0"], "sig").scores)
    res = cox_model(synth.clinical, s, "sig", "synthetic", ["sex", "idh_status"])
    names = [r.covariate for r in res.rows]
    assert "sex[Female vs Male]" in names  # Male is the fixed reference level
    assert not any(n.startswith("idh_status") for n in names)  # constant column dropped, not a singular fit


def test_kaplan_meier_matches_lifelines(synth):
    from lifelines import KaplanMeierFitter

    lab = pd.Series(np.where(np.arange(200) % 2 == 0, "a", "b"), index=synth.clinical.index)
    km = kaplan_meier(synth.clinical, lab, "parity", "synthetic")
    sub = synth.clinical[lab == "a"]
    ref = KaplanMeierFitter().fit(sub.os_months, sub.os_event).median_survival_time_
    assert np.isclose(next(g for g in km.groups if g.label == "a").median_survival, ref)
    assert 0 <= km.logrank_p <= 1


def test_group_comparison_detects_shift():
    c = make_synthetic_cohort(n=300, subtype_shift=2.0, seed=3)
    s = pd.Series(score_signature(c, [f"G{i}" for i in range(5)], "sig").scores)
    res = compare_groups(s, c.clinical["subtype"], "sig", "subtype", "synthetic")
    assert res.omnibus_p < 1e-6
    pm = next(p for p in res.pairwise if {p["group_a"], p["group_b"]} == {"Proneural", "Mesenchymal"})
    assert "q" in pm and pm["q"] < 1e-6


def test_differential_expression_finds_signal_genes(synth):
    s = pd.Series(score_signature(synth, ["G0", "G1"], "sig").scores)
    de = differential_expression(synth, dichotomise(s), "high", "low")
    top = [r.gene for r in de.rows[:5]]
    assert set(top) == {f"G{i}" for i in range(5)}
    bg = [r for r in de.rows if r.gene.startswith("BG")]
    assert np.mean([r.q_value < 0.05 for r in bg]) < 0.05  # FDR controlled on null genes


def test_correlation_matrix_symmetric(synth):
    cr = gene_correlation(synth, ["G0", "G1", "BG0"])
    m = np.array(cr.matrix)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 1) and m[0, 1] > 0.5


def test_validation_requires_independent_cohort():
    only = {"disc": make_synthetic_cohort(seed=4, dataset_id="disc")}
    vs, _ = validate_signature_survival(only, "disc", [f"G{i}" for i in range(5)], "sig", ["age"])
    assert vs.verdict.startswith("not validated") and vs.n_cohorts_replicating == 0


def test_validation_replicates_true_effect():
    cohorts = {k: make_synthetic_cohort(n=500, beta=0.6, seed=s, dataset_id=k) for k, s in [("disc", 10), ("v1", 11), ("v2", 12)]}
    vs, _ = validate_signature_survival(cohorts, "disc", [f"G{i}" for i in range(5)], "sig", ["age", "sex"])
    assert vs.n_cohorts_replicating == 2 and vs.verdict.startswith("replicated")


def test_group_validation_direction_rule():
    cs = {}
    for k, sh in [("disc", 2.0), ("v1", 2.0), ("v2", -2.0)]:
        c = make_synthetic_cohort(n=300, subtype_shift=sh, seed=hash(k) % 100, dataset_id=k)
        cs[k] = (pd.Series(score_signature(c, ["G0", "G1"], "s").scores), c.clinical["subtype"], "synthetic")
    vs = validate_group_difference(cs, "disc", "Mesenchymal", "Proneural", "s")
    v2 = next(r for r in vs.per_cohort if r.cohort_id == "v2")
    assert v2.direction_consistent is False and vs.n_cohorts_replicating == 1 and "partially" in vs.verdict
