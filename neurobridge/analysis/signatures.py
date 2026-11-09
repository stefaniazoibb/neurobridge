"""Gene-signature scoring for bulk cohorts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobridge.analysis.common import finish, new_record
from neurobridge.analysis.preprocessing import zscore_genes
from neurobridge.data.cohort import Cohort
from neurobridge.schemas.analysis import AnalysisKind, SignatureScoreResult


def score_signature(cohort: Cohort, genes: list[str], signature_name: str, method: str = "mean_z", kind: AnalysisKind = AnalysisKind.EXPLORATORY) -> SignatureScoreResult:
    """Score each sample.

    mean_z: mean across genes of the per-gene z-score (across samples) of log2 expression. Simple, transparent,
    gives equal weight to each gene. Scores are on a standard-deviation scale and comparable within a cohort only.
    rank: mean percentile rank of the genes within each sample (singscore-like, robust to scale differences).
    """
    found = [g for g in genes if g in cohort.expression.index]
    missing = [g for g in genes if g not in cohort.expression.index]
    rec = new_record("signature_score", kind, [cohort.dataset.dataset_id], f"signature score ({method})", {"signature": signature_name, "genes": genes, "method": method})
    if len(found) == 0:
        rec.warnings.append("none of the signature genes are present in the expression matrix")
        return SignatureScoreResult(record=finish(rec), signature=signature_name, genes_requested=genes, genes_found=[], genes_missing=missing, scores={}, method=method)
    sub = cohort.expression.loc[found]
    if method == "mean_z":
        z = zscore_genes(sub)
        scores = z.mean(axis=0, skipna=True)
    elif method == "rank":
        ranks = cohort.expression.rank(axis=0, pct=True)
        scores = ranks.loc[found].mean(axis=0)
    else:
        raise ValueError(method)
    if missing:
        rec.warnings.append(f"{len(missing)} signature gene(s) absent from matrix: {', '.join(missing)}")
    if len(found) < 3:
        rec.limitations.append("fewer than three genes contribute to this score; interpret as a candidate-gene average, not a pathway score")
    rec.n_used = int(sub.shape[1])
    return SignatureScoreResult(record=finish(rec), signature=signature_name, genes_requested=genes, genes_found=found, genes_missing=missing, scores={k: float(v) for k, v in scores.items()}, method=method)


def dichotomise(scores: pd.Series, how: str = "median") -> pd.Series:
    """Return 'high'/'low' labels. 'median' split, or 'tertiles' (top vs bottom tertile, middle dropped)."""
    s = scores.dropna()
    if how == "median":
        return pd.Series(np.where(s >= s.median(), "high", "low"), index=s.index)
    if how == "tertiles":
        lo, hi = s.quantile([1 / 3, 2 / 3])
        lab = pd.Series(index=s.index, dtype=object)
        lab[s <= lo] = "low"
        lab[s >= hi] = "high"
        return lab.dropna()
    raise ValueError(how)
