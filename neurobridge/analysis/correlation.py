"""Gene-gene correlation matrices with BH-adjusted p-values."""

from __future__ import annotations

import numpy as np
from scipy import stats

from neurobridge.analysis.common import bh_adjust, finish, new_record
from neurobridge.data.cohort import Cohort
from neurobridge.schemas.analysis import AnalysisKind, CorrelationResult


def gene_correlation(cohort: Cohort, genes: list[str], method: str = "spearman") -> CorrelationResult:
    found = [g for g in genes if g in cohort.expression.index]
    rec = new_record("gene_correlation", AnalysisKind.EXPLORATORY, [cohort.dataset.dataset_id], f"{method} correlation, BH-adjusted p (upper triangle)", {"genes": genes, "method": method})
    X = cohort.expression.loc[found].values.astype(float)
    n = X.shape[1]
    k = len(found)
    r = np.eye(k)
    p = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            if method == "spearman":
                res = stats.spearmanr(X[i], X[j])
            else:
                res = stats.pearsonr(X[i], X[j])
            r[i, j] = r[j, i] = res[0]
            p[i, j] = p[j, i] = res[1]
    iu = np.triu_indices(k, 1)
    q = np.zeros_like(p)
    q[iu] = bh_adjust(p[iu])
    q = q + q.T
    missing = [g for g in genes if g not in found]
    if missing:
        rec.warnings.append("genes absent: " + ", ".join(missing))
    rec.n_used = n
    rec.limitations.append("Co-expression in bulk tissue is dominated by cell-type composition; correlated genes may simply be expressed by the same cell type.")
    return CorrelationResult(record=finish(rec), genes=found, method=method, matrix=r.round(4).tolist(), p_values=p.tolist(), q_values=q.tolist(), n=n)
