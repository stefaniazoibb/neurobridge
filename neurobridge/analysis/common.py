from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from statsmodels.stats.multitest import multipletests

from neurobridge.config import get_settings
from neurobridge.provenance import software_versions, utc_now_iso
from neurobridge.schemas.analysis import AnalysisKind, AnalysisRecord


def new_record(name: str, kind: AnalysisKind, dataset_ids: list[str], method: str, parameters: dict[str, Any] | None = None, seed: int | None = None) -> AnalysisRecord:
    return AnalysisRecord(
        analysis_id=f"{name}-{uuid.uuid4().hex[:8]}",
        name=name,
        kind=kind,
        dataset_ids=list(dataset_ids),
        method=method,
        parameters=parameters or {},
        seed=get_settings().seed if seed is None else seed,
        software=software_versions(),
        started_at=utc_now_iso(),
    )


def finish(record: AnalysisRecord) -> AnalysisRecord:
    record.finished_at = utc_now_iso()
    return record


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() > 0:
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def hedges_g(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Hedges' g (bias-corrected standardised mean difference, a - b) with a 95% CI (large-sample normal approximation)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0, 0.0, 0.0
    d = (a.mean() - b.mean()) / sp
    j = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * j
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    return float(g), float(g - 1.96 * se), float(g + 1.96 * se)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = (a[:, None] > b[None, :]).mean()
    lt = (a[:, None] < b[None, :]).mean()
    return float(gt - lt)
