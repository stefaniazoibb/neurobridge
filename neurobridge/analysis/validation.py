"""Cross-cohort validation of a pre-specified association, with fixed- and random-effects meta-analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from neurobridge.analysis.common import finish, new_record
from neurobridge.analysis.signatures import score_signature
from neurobridge.analysis.survival import cox_model, exposure_row
from neurobridge.data.cohort import Cohort
from neurobridge.schemas.analysis import AnalysisKind, CoxResult, ValidationCohortResult, ValidationSummary


def meta_analyse_log_hr(log_hr: np.ndarray, se: np.ndarray) -> dict:
    """Inverse-variance fixed-effect pooling plus DerSimonian-Laird random effects, with Cochran's Q and I^2."""
    w = 1 / se**2
    fe = float(np.sum(w * log_hr) / np.sum(w))
    fe_se = float(np.sqrt(1 / np.sum(w)))
    Q = float(np.sum(w * (log_hr - fe) ** 2))
    k = len(log_hr)
    q_p = float(1 - stats.chi2.cdf(Q, k - 1)) if k > 1 else np.nan
    I2 = float(max(0.0, (Q - (k - 1)) / Q) * 100) if Q > 0 and k > 1 else 0.0
    C = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    wr = 1 / (se**2 + tau2)
    re = float(np.sum(wr * log_hr) / np.sum(wr))
    re_se = float(np.sqrt(1 / np.sum(wr)))
    z = re / re_se
    return {"fixed_log_hr": fe, "fixed_se": fe_se, "random_log_hr": re, "random_se": re_se, "random_p": float(2 * (1 - stats.norm.cdf(abs(z)))), "fixed_p": float(2 * (1 - stats.norm.cdf(abs(fe / fe_se)))), "Q": Q, "Q_p": q_p, "I2": I2, "tau2": float(tau2), "k": k}


def validate_signature_survival(
    cohorts: dict[str, Cohort],
    discovery_id: str,
    genes: list[str],
    signature_name: str,
    covariates: list[str],
    *,
    primary_only: bool = True,
    alpha: float = 0.05,
) -> tuple[ValidationSummary, dict[str, CoxResult]]:
    """Confirmatory test: the per-SD adjusted hazard ratio of the signature, estimated separately in each cohort.
    Replication requires the same direction as discovery and p < alpha in the validation cohort. Only cohorts
    other than the discovery cohort count as validation."""
    rec = new_record("cross_cohort_validation", AnalysisKind.CONFIRMATORY, list(cohorts), "per-cohort Cox PH (per-SD HR, adjusted), direction + significance replication rule, inverse-variance meta-analysis", {"signature": signature_name, "genes": genes, "covariates": covariates, "primary_only": primary_only, "alpha": alpha})
    per: list[ValidationCohortResult] = []
    models: dict[str, CoxResult] = {}
    disc_sign = None
    for cid, cohort in cohorts.items():
        c = cohort.primary_tumours() if primary_only else cohort
        sc = score_signature(c, genes, signature_name, kind=AnalysisKind.CONFIRMATORY)
        s = pd.Series(sc.scores)
        cov_here = [cv for cv in covariates if cv in c.clinical.columns and c.clinical[cv].notna().mean() >= 0.5]
        notes = []
        if set(cov_here) != set(covariates):
            notes.append(f"adjusted for {cov_here or 'nothing'} (unavailable here: {sorted(set(covariates) - set(cov_here))})")
        try:
            cx = cox_model(c.clinical, s, signature_name, cid, cov_here, kind=AnalysisKind.CONFIRMATORY)
        except Exception as exc:  # noqa: BLE001
            per.append(ValidationCohortResult(cohort_id=cid, role="discovery" if cid == discovery_id else "validation", n=int(s.notna().sum()), estimate_name="log HR per SD", estimate=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, significant=False, notes=[f"model failed: {exc}"]))
            continue
        models[cid] = cx
        row = exposure_row(cx, signature_name)
        role = "discovery" if cid == discovery_id else "validation"
        if cid == discovery_id:
            disc_sign = np.sign(row.coef)
        per.append(ValidationCohortResult(cohort_id=cid, role=role, n=cx.n, events=cx.events, estimate_name="HR per SD", estimate=row.hr, ci_low=row.hr_ci_low, ci_high=row.hr_ci_high, p=row.p, significant=row.p < alpha, notes=notes + cx.record.warnings + [f"covariates: {cx.covariates_adjusted}"]))
    for r in per:
        if r.role == "validation" and disc_sign is not None and not np.isnan(r.estimate):
            r.direction_consistent = bool(np.sign(np.log(r.estimate)) == disc_sign)
    val = [r for r in per if r.role == "validation" and not np.isnan(r.estimate)]
    n_rep = sum(1 for r in val if r.direction_consistent and r.significant)
    ok = [r for r in per if not np.isnan(r.estimate)]
    pooled = meta_analyse_log_hr(np.array([np.log(r.estimate) for r in ok]), np.array([(np.log(r.ci_high) - np.log(r.ci_low)) / (2 * 1.96) for r in ok])) if len(ok) >= 2 else None
    if not val:
        verdict = "not validated (no independent cohort analysed)"
        rationale = ["Only the discovery cohort was available; a single-cohort result cannot be called validated."]
    elif n_rep == len(val):
        verdict = f"replicated in {n_rep}/{len(val)} independent cohorts"
        rationale = ["Direction consistent with discovery and p < alpha in every validation cohort."]
    elif n_rep > 0:
        verdict = f"partially replicated ({n_rep}/{len(val)} cohorts)"
        rationale = ["Replication is inconsistent across cohorts; consider cohort differences (platform, population, treatment era)."]
    else:
        dirs = [r.direction_consistent for r in val]
        verdict = "not replicated" if not any(dirs) else "direction consistent but not significant in validation"
        rationale = ["Validation cohorts do not reproduce a significant association in the discovery direction."]
    if pooled is not None:
        rationale.append(f"Random-effects pooled HR per SD = {np.exp(pooled['random_log_hr']):.2f} (I^2 = {pooled['I2']:.0f}%); pooled estimates include the discovery cohort and are descriptive.")
        if pooled["I2"] > 50:
            rationale.append("Substantial between-cohort heterogeneity (I^2 > 50%).")
    rec.limitations.append("Validation cohorts differ from discovery in population, platform and available covariates; adjustment sets are therefore not identical across cohorts.")
    rec.limitations.append("Replication of an association does not establish mechanism or causation.")
    summary = ValidationSummary(record=finish(rec), hypothesis=f"{signature_name} score is associated with overall survival (adjusted Cox)", discovery_cohort=discovery_id, validation_cohorts=[c for c in cohorts if c != discovery_id], per_cohort=per, pooled_estimate=np.exp(pooled["random_log_hr"]) if pooled else None, pooled_ci_low=np.exp(pooled["random_log_hr"] - 1.96 * pooled["random_se"]) if pooled else None, pooled_ci_high=np.exp(pooled["random_log_hr"] + 1.96 * pooled["random_se"]) if pooled else None, pooled_p=pooled["random_p"] if pooled else None, heterogeneity_i2=pooled["I2"] if pooled else None, heterogeneity_q_p=pooled["Q_p"] if pooled else None, n_cohorts_replicating=n_rep, verdict=verdict, verdict_rationale=rationale)
    return summary, models


def validate_group_difference(
    cohort_scores: dict[str, tuple[pd.Series, pd.Series, str]],
    discovery_id: str,
    group_a: str,
    group_b: str,
    variable: str,
    *,
    alpha: float = 0.05,
) -> ValidationSummary:
    """Replication of a two-group difference (Hedges' g, a - b) across cohorts.

    cohort_scores: cohort_id -> (values, group labels, label_source). Replication requires the same sign as the
    discovery estimate and a two-sided Mann-Whitney p < alpha in an independent cohort.
    """
    from neurobridge.analysis.common import hedges_g

    rec = new_record("cross_cohort_group_validation", AnalysisKind.CONFIRMATORY, list(cohort_scores), "per-cohort Hedges' g with 95% CI, Mann-Whitney U, direction + significance replication rule, inverse-variance meta-analysis of g", {"group_a": group_a, "group_b": group_b, "variable": variable, "alpha": alpha})
    per: list[ValidationCohortResult] = []
    disc_sign = None
    for cid, (vals, labs, source) in cohort_scores.items():
        df = pd.DataFrame({"v": vals, "g": labs}).dropna()
        a = df.loc[df.g == group_a, "v"].values
        b = df.loc[df.g == group_b, "v"].values
        role = "discovery" if cid == discovery_id else "validation"
        if len(a) < 5 or len(b) < 5:
            per.append(ValidationCohortResult(cohort_id=cid, role=role, n=len(a) + len(b), estimate_name="Hedges g", estimate=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, significant=False, notes=[f"insufficient samples (n_a={len(a)}, n_b={len(b)})", f"labels: {source}"]))
            continue
        g, lo, hi = hedges_g(a, b)
        p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
        if cid == discovery_id:
            disc_sign = np.sign(g)
        per.append(ValidationCohortResult(cohort_id=cid, role=role, n=len(a) + len(b), estimate_name="Hedges g", estimate=g, ci_low=lo, ci_high=hi, p=p, significant=p < alpha, notes=[f"n({group_a})={len(a)}, n({group_b})={len(b)}", f"labels: {source}"]))
    for r in per:
        if r.role == "validation" and disc_sign is not None and not np.isnan(r.estimate):
            r.direction_consistent = bool(np.sign(r.estimate) == disc_sign)
    val = [r for r in per if r.role == "validation" and not np.isnan(r.estimate)]
    n_rep = sum(1 for r in val if r.direction_consistent and r.significant)
    ok = [r for r in per if not np.isnan(r.estimate)]
    pooled = meta_analyse_log_hr(np.array([r.estimate for r in ok]), np.array([(r.ci_high - r.ci_low) / (2 * 1.96) for r in ok])) if len(ok) >= 2 else None
    if not val:
        verdict, rationale = "not validated (no independent cohort analysed)", ["A single-cohort result cannot be called validated."]
    elif n_rep == len(val):
        verdict, rationale = f"replicated in {n_rep}/{len(val)} independent cohorts", ["Same direction as discovery and p < alpha in every validation cohort."]
    elif n_rep > 0:
        verdict, rationale = f"partially replicated ({n_rep}/{len(val)} cohorts)", ["Replication is inconsistent across cohorts."]
    else:
        verdict, rationale = "not replicated", ["No validation cohort reproduces the discovery difference."]
    if pooled:
        rationale.append(f"Random-effects pooled Hedges g = {pooled['random_log_hr']:.2f} (I^2 = {pooled['I2']:.0f}%), descriptive (includes discovery).")
    rec.limitations.append("Group labels in validation cohorts may be model projections rather than original annotations; see label source per cohort.")
    rec.limitations.append("A replicated group difference is an association in bulk tissue; it does not localise the signal to tumour cells.")
    return ValidationSummary(record=finish(rec), hypothesis=f"{variable} differs between {group_a} and {group_b}", discovery_cohort=discovery_id, validation_cohorts=[c for c in cohort_scores if c != discovery_id], per_cohort=per,
                             pooled_estimate=pooled["random_log_hr"] if pooled else None, pooled_ci_low=(pooled["random_log_hr"] - 1.96 * pooled["random_se"]) if pooled else None, pooled_ci_high=(pooled["random_log_hr"] + 1.96 * pooled["random_se"]) if pooled else None,
                             pooled_p=pooled["random_p"] if pooled else None, heterogeneity_i2=pooled["I2"] if pooled else None, heterogeneity_q_p=pooled["Q_p"] if pooled else None, n_cohorts_replicating=n_rep, verdict=verdict, verdict_rationale=rationale)
