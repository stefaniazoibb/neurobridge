"""Filtering and scaling of bulk expression matrices (genes x samples, log2 scale)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def filter_expressed_genes(expr: pd.DataFrame, min_mean_log: float = 1.0, min_fraction_detected: float = 0.2) -> pd.DataFrame:
    """Keep genes with mean log2 expression >= min_mean_log and detected (> 0) in >= fraction of samples."""
    detected = (expr > 0).mean(axis=1)
    keep = (expr.mean(axis=1) >= min_mean_log) & (detected >= min_fraction_detected)
    return expr.loc[keep]


def zscore_genes(expr: pd.DataFrame) -> pd.DataFrame:
    mu = expr.mean(axis=1)
    sd = expr.std(axis=1, ddof=1).replace(0, np.nan)
    return expr.sub(mu, axis=0).div(sd, axis=0)


def top_variable_genes(expr: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    v = expr.var(axis=1)
    return expr.loc[v.sort_values(ascending=False).index[:n]]
