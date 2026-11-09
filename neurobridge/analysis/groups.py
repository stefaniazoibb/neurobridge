"""Compare a continuous variable (gene or signature score) across molecular groups."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from neurobridge.analysis.common import bh_adjust, cliffs_delta, finish, hedges_g, new_record
from neurobridge.schemas.analysis import AnalysisKind, GroupComparisonResult


def compare_groups(values: pd.Series, groups: pd.Series, variable: str, grouping: str, dataset_id: str, min_n: int = 5) -> GroupComparisonResult:
    """Kruskal-Wallis omnibus test, epsilon-squared effect size, pairwise Mann-Whitney with BH and Hedges' g / Cliff's delta."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    counts = df["g"].value_counts()
    keep = counts.index[counts >= min_n]
    dropped = counts.index[counts < min_n].tolist()
    df = df[df["g"].isin(keep)]
    rec = new_record("group_comparison", AnalysisKind.EXPLORATORY, [dataset_id], "Kruskal-Wallis + pairwise Mann-Whitney (BH), Hedges' g, Cliff's delta", {"variable": variable, "grouping": grouping, "min_n": min_n})
    if dropped:
        rec.warnings.append(f"groups with n < {min_n} excluded: {dropped}")
    samples = [df.loc[df["g"] == g, "v"].values for g in keep]
    if len(samples) >= 2:
        H, p = stats.kruskal(*samples)
        n = len(df)
        eps2 = float((H - len(samples) + 1) / (n - len(samples))) if n > len(samples) else np.nan
    else:
        H, p, eps2 = np.nan, np.nan, np.nan
        rec.warnings.append("fewer than two groups with sufficient samples")
    group_stats = [{"group": g, "n": int(len(s)), "median": float(np.median(s)), "mean": float(np.mean(s)), "sd": float(np.std(s, ddof=1)) if len(s) > 1 else np.nan, "q1": float(np.percentile(s, 25)), "q3": float(np.percentile(s, 75))} for g, s in zip(keep, samples)]
    pairs = []
    for (ga, sa), (gb, sb) in itertools.combinations(zip(keep, samples), 2):
        u = stats.mannwhitneyu(sa, sb, alternative="two-sided")
        g_, lo, hi = hedges_g(sa, sb)
        pairs.append({"group_a": ga, "group_b": gb, "n_a": len(sa), "n_b": len(sb), "u": float(u.statistic), "p": float(u.pvalue), "hedges_g": g_, "g_ci_low": lo, "g_ci_high": hi, "cliffs_delta": cliffs_delta(sa, sb), "median_diff": float(np.median(sa) - np.median(sb))})
    if pairs:
        qs = bh_adjust([pr["p"] for pr in pairs])
        for pr, q in zip(pairs, qs):
            pr["q"] = float(q)
    rec.n_used = int(len(df))
    return GroupComparisonResult(record=finish(rec), variable=variable, grouping=grouping, group_stats=group_stats, omnibus_test="Kruskal-Wallis H", omnibus_statistic=float(H), omnibus_p=float(p), pairwise=pairs, effect_size_name="epsilon-squared", effect_size=eps2)
