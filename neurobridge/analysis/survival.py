"""Survival analysis: Kaplan-Meier, log-rank, Cox proportional hazards with assumption checks."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test

from neurobridge.analysis.common import finish, new_record
from neurobridge.schemas.analysis import AnalysisKind, CoxResult, CoxRow, KMGroup, KMResult


def _survival_frame(clinical: pd.DataFrame) -> pd.DataFrame:
    df = clinical[["os_months", "os_event"]].copy()
    df = df.dropna()
    df = df[df["os_months"] > 0]
    return df


def kaplan_meier(clinical: pd.DataFrame, groups: pd.Series, stratifier: str, dataset_id: str, kind: AnalysisKind = AnalysisKind.EXPLORATORY, min_n: int = 5) -> KMResult:
    surv = _survival_frame(clinical)
    g = groups.reindex(surv.index).dropna()
    surv = surv.loc[g.index]
    rec = new_record("kaplan_meier", kind, [dataset_id], "Kaplan-Meier estimator, Greenwood 95% CI, log-rank test", {"stratifier": stratifier, "min_n": min_n})
    counts = g.value_counts()
    keep = [k for k in counts.index if counts[k] >= min_n]
    if len(keep) < len(counts):
        rec.warnings.append(f"groups with n < {min_n} excluded: {[k for k in counts.index if counts[k] < min_n]}")
    out_groups: list[KMGroup] = []
    for label in keep:
        idx = g.index[g == label]
        kmf = KaplanMeierFitter()
        kmf.fit(surv.loc[idx, "os_months"], surv.loc[idx, "os_event"], label=str(label))
        ci = kmf.confidence_interval_survival_function_
        med = kmf.median_survival_time_
        try:
            from lifelines.utils import median_survival_times

            mci = median_survival_times(kmf.confidence_interval_)
            lo, hi = float(mci.iloc[0, 0]), float(mci.iloc[0, 1])
        except Exception:  # noqa: BLE001
            lo, hi = None, None
        out_groups.append(KMGroup(
            label=str(label), n=int(len(idx)), events=int(surv.loc[idx, "os_event"].sum()),
            median_survival=None if np.isinf(med) else float(med),
            median_ci_low=None if lo is None or np.isinf(lo) else lo, median_ci_high=None if hi is None or np.isinf(hi) else hi,
            timeline=kmf.survival_function_.index.tolist(), survival=kmf.survival_function_.iloc[:, 0].tolist(),
            ci_low=ci.iloc[:, 0].tolist(), ci_high=ci.iloc[:, 1].tolist(),
        ))
    stat, p = None, None
    if len(keep) == 2:
        a, b = [g.index[g == k] for k in keep]
        lr = logrank_test(surv.loc[a, "os_months"], surv.loc[b, "os_months"], surv.loc[a, "os_event"], surv.loc[b, "os_event"])
        stat, p = float(lr.test_statistic), float(lr.p_value)
    elif len(keep) > 2:
        sub = surv.loc[g.index[g.isin(keep)]]
        lr = multivariate_logrank_test(sub["os_months"], g.loc[sub.index], sub["os_event"])
        stat, p = float(lr.test_statistic), float(lr.p_value)
    rec.n_used = int(sum(gr.n for gr in out_groups))
    rec.n_excluded = int(len(clinical) - rec.n_used)
    rec.limitations.append("Unadjusted comparison: group differences may be confounded by age, molecular subtype or treatment.")
    return KMResult(record=finish(rec), stratifier=stratifier, time_unit="months", groups=out_groups, logrank_statistic=stat, logrank_p=p, n_total=rec.n_used, n_events=int(sum(gr.events for gr in out_groups)))


# Preferred reference levels for dummy coding so hazard ratios read the same way in every cohort.
REFERENCE_LEVELS = {"sex": "Male", "idh_status": "Wildtype", "mgmt_status": "unmethylated", "g_cimp": "non-G-CIMP", "subtype": "Mesenchymal", "sample_type": "Primary", "codeletion_1p19q": "Non-codel"}


def _design(clinical: pd.DataFrame, exposure: pd.Series, exposure_name: str, covariates: list[str], standardise_exposure: bool) -> tuple[pd.DataFrame, list[str], list[str]]:
    surv = _survival_frame(clinical)
    df = surv.copy()
    x = exposure.reindex(df.index)
    if standardise_exposure and pd.api.types.is_numeric_dtype(x):
        x = (x - x.mean()) / x.std(ddof=1)
    df[exposure_name] = x
    used, dropped = [], []
    for cov in covariates:
        if cov not in clinical.columns or clinical[cov].reindex(df.index).notna().sum() < 0.5 * len(df):
            dropped.append(cov)
            continue
        col = clinical[cov].reindex(df.index)
        if pd.api.types.is_numeric_dtype(col):
            df[cov] = col
            used.append(cov)
        else:
            col = col.astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
            levels = col.dropna().unique().tolist()
            if len(levels) < 2:
                dropped.append(cov)
                continue
            pref = REFERENCE_LEVELS.get(cov)
            ref = pref if pref in levels else sorted(levels, key=lambda v: -(col == v).sum())[0]
            for lv in levels:
                if lv == ref:
                    continue
                name = f"{cov}[{lv} vs {ref}]"
                df[name] = (col == lv).astype(float).where(col.notna())
                used.append(name)
    df = df.dropna()
    # Drop covariate columns that are constant after listwise deletion (would make the design singular).
    constant = [c for c in used if df[c].nunique() < 2]
    if constant:
        df = df.drop(columns=constant)
        used = [c for c in used if c not in constant]
        dropped = dropped + [f"{c} (constant after missing-data exclusion)" for c in constant]
    return df, used, dropped


def cox_model(
    clinical: pd.DataFrame,
    exposure: pd.Series,
    exposure_name: str,
    dataset_id: str,
    covariates: Optional[list[str]] = None,
    *,
    standardise_exposure: bool = True,
    kind: AnalysisKind = AnalysisKind.EXPLORATORY,
    penalizer: float = 0.0,
) -> CoxResult:
    """Cox PH model. Continuous exposures are standardised so the HR is per 1 SD. Categorical covariates are
    dummy-coded against their most frequent level. Proportional-hazards assumption tested with scaled Schoenfeld residuals."""
    covariates = covariates or []
    df, used, dropped = _design(clinical, exposure, exposure_name, covariates, standardise_exposure)
    model_name = "univariable" if not used else "multivariable"
    rec = new_record(f"cox_{model_name}", kind, [dataset_id], "Cox proportional hazards (lifelines), Efron ties, Wald CIs; PH test on scaled Schoenfeld residuals", {"exposure": exposure_name, "covariates": covariates, "standardise_exposure": standardise_exposure, "penalizer": penalizer})
    if dropped:
        rec.warnings.append(f"covariates dropped (absent or >50% missing): {dropped}")
    n, events = int(len(df)), int(df["os_event"].sum())
    if n < 20 or events < 10:
        rec.limitations.append(f"very small sample (n={n}, events={events}); estimates unstable")
    epv = events / max(1, len(used) + 1)
    if epv < 10:
        rec.limitations.append(f"events per variable = {epv:.1f} (< 10): multivariable model may be overfitted")
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="os_months", event_col="os_event")
    s = cph.summary
    rows = [CoxRow(covariate=str(i), coef=float(r["coef"]), hr=float(r["exp(coef)"]), hr_ci_low=float(r["exp(coef) lower 95%"]), hr_ci_high=float(r["exp(coef) upper 95%"]), se=float(r["se(coef)"]), z=float(r["z"]), p=float(r["p"])) for i, r in s.iterrows()]
    ph: dict = {}
    ph_ok: Optional[bool] = None
    try:
        pht = proportional_hazard_test(cph, df, time_transform="rank")
        for cov, row in pht.summary.iterrows():
            ph[str(cov)] = {"statistic": float(row["test_statistic"]), "p": float(row["p"])}
        ph_ok = all(v["p"] > 0.05 for v in ph.values())
        if not ph_ok:
            viol = [k for k, v in ph.items() if v["p"] <= 0.05]
            rec.warnings.append(f"proportional-hazards assumption questionable for: {viol} (Schoenfeld test p <= 0.05); HR should be read as a time-averaged effect")
    except Exception as exc:  # noqa: BLE001
        rec.warnings.append(f"PH test failed: {exc}")
    rec.n_used, rec.n_excluded = n, int(len(clinical) - n)
    if standardise_exposure:
        rec.limitations.append(f"HR for {exposure_name} is per 1 standard deviation of the score within this cohort.")
    rec.limitations.append("Observational cohort: hazard ratios are associations, not causal effects.")
    return CoxResult(record=finish(rec), model=model_name, rows=rows, n=n, events=events, concordance=float(cph.concordance_index_), log_likelihood=float(cph.log_likelihood_), aic=float(cph.AIC_partial_), ph_test=ph, ph_assumption_ok=ph_ok, covariates_adjusted=used)


def exposure_row(res: CoxResult, exposure_name: str) -> Optional[CoxRow]:
    for r in res.rows:
        if r.covariate == exposure_name:
            return r
    return None
