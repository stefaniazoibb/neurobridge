"""Two-group differential expression on log2 expression with effect sizes, CIs and BH correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from neurobridge.analysis.common import bh_adjust, finish, hedges_g, new_record
from neurobridge.analysis.preprocessing import filter_expressed_genes
from neurobridge.data.cohort import Cohort
from neurobridge.schemas.analysis import AnalysisKind, DEResult, EffectSizeRow


def differential_expression(
    cohort: Cohort,
    labels: pd.Series,
    group_a: str,
    group_b: str,
    *,
    test: str = "welch",
    min_mean_log: float = 1.0,
    alpha: float = 0.05,
    max_rows: int | None = None,
    kind: AnalysisKind = AnalysisKind.EXPLORATORY,
) -> DEResult:
    """Compare group_a vs group_b (log2 fold change = mean_a - mean_b on the log2 scale).

    test: 'welch' (unequal-variance t-test) or 'mannwhitney'. Genes are filtered for expression first so the
    multiple-testing burden reflects genes that could plausibly be detected.
    """
    labels = labels.dropna()
    ids_a = [s for s in labels.index[labels == group_a] if s in cohort.expression.columns]
    ids_b = [s for s in labels.index[labels == group_b] if s in cohort.expression.columns]
    rec = new_record("differential_expression", kind, [cohort.dataset.dataset_id], f"{test} test, Benjamini-Hochberg FDR, Hedges' g", {"group_a": group_a, "group_b": group_b, "test": test, "min_mean_log": min_mean_log, "alpha": alpha})
    expr = filter_expressed_genes(cohort.expression, min_mean_log=min_mean_log)
    A = expr[ids_a].values.astype(float)
    B = expr[ids_b].values.astype(float)
    if test == "welch":
        t, p = stats.ttest_ind(A, B, axis=1, equal_var=False, nan_policy="omit")
    elif test == "mannwhitney":
        p = np.array([stats.mannwhitneyu(A[i], B[i], alternative="two-sided").pvalue for i in range(A.shape[0])])
    else:
        raise ValueError(test)
    q = bh_adjust(p)
    rows: list[EffectSizeRow] = []
    mean_a, mean_b = A.mean(axis=1), B.mean(axis=1)
    for i, g in enumerate(expr.index):
        gg, lo, hi = hedges_g(A[i], B[i])
        rows.append(EffectSizeRow(gene=g, log2_fc=float(mean_a[i] - mean_b[i]), hedges_g=gg, hedges_g_ci_low=lo, hedges_g_ci_high=hi, mean_a=float(mean_a[i]), mean_b=float(mean_b[i]), p_value=float(p[i]), q_value=float(q[i]), test=test))
    rows.sort(key=lambda r: (np.nan_to_num(r.p_value, nan=1.0), -abs(r.log2_fc)))
    n_sig = int(np.sum(q < alpha))
    if max_rows:
        rows = rows[:max_rows]
    rec.n_used = len(ids_a) + len(ids_b)
    rec.n_excluded = int(cohort.expression.shape[1] - rec.n_used)
    rec.limitations.append("Bulk tissue: differences may reflect cellular composition (e.g. infiltrated neurons, immune cells) rather than tumour-cell-intrinsic programmes.")
    if min(len(ids_a), len(ids_b)) < 20:
        rec.limitations.append(f"small group size (n_a={len(ids_a)}, n_b={len(ids_b)}): effect-size CIs are wide")
    return DEResult(record=finish(rec), group_a=group_a, group_b=group_b, n_a=len(ids_a), n_b=len(ids_b), rows=rows, n_tested=int(expr.shape[0]), n_significant=n_sig, alpha=alpha)
