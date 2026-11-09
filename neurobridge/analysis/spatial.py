"""Spatial transcriptomics analyses on a Visium section (one patient: every result is descriptive).

1. Spot programmes: scanpy score_genes for candidate, mature-neuron, tumour/reactive-glial, myeloid and vascular sets.
2. Spatial domains: Leiden clustering of spots on expression, annotated by their mean programme scores.
3. Spatial autocorrelation: Moran's I on the Visium hexagonal neighbour graph, permutation p-values.
4. Neuron-interface analysis: among non-neuronal spots, expression versus hexagonal-ring distance to the nearest
   neuron-rich spot (Spearman) and interface (distance 1-2) versus distal (>= 5) spots.
5. Ligand-receptor co-localisation: mean over spots of ligand(spot) x mean receptor(6 neighbours). Two nulls: a global
   spot permutation, and a domain-stratified permutation (ligand shuffled only within its spatial domain), which
   asks whether the pairing is tighter than shared tissue composition alone would produce.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from neurobridge.analysis.common import bh_adjust, cliffs_delta, finish, hedges_g, new_record
from neurobridge.analysis.communication import LR_PAIRS
from neurobridge.config import get_settings
from neurobridge.schemas.analysis import AnalysisKind

PROGRAMMES = {
    "neuron": ["SNAP25", "SYT1", "RBFOX3", "GABRA1", "NEFL", "STMN2", "SYN1", "GAD1"],
    "tumour_glial": ["CHI3L1", "CD44", "VIM", "PTPRZ1", "SOX2", "EGFR", "MKI67", "TOP2A"],
    "myeloid": ["PTPRC", "CD74", "C1QB", "AIF1", "CD14", "TYROBP"],
    "vascular": ["PECAM1", "VWF", "CLDN5", "COL4A1", "ACTA2"],
}
SPATIAL_PARAMS = {"min_counts": 500, "min_genes": 250, "n_top_hvg": 2000, "n_pcs": 30, "n_neighbors": 15, "leiden_resolution": 0.6, "neuron_rich_quantile": 0.8, "n_perm": 999}


def hex_neighbours(obs: pd.DataFrame) -> list[np.ndarray]:
    """Visium hexagonal grid: neighbours of (r, c) are (r, c+-2) and (r+-1, c+-1)."""
    idx = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(obs["array_row"], obs["array_col"]))}
    out = []
    for r, c in zip(obs["array_row"].astype(int), obs["array_col"].astype(int)):
        nb = [idx[k] for k in [(r, c - 2), (r, c + 2), (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)] if k in idx]
        out.append(np.array(nb, dtype=int))
    return out


def ring_distance(nbrs: list[np.ndarray], sources: np.ndarray) -> np.ndarray:
    """Graph distance (in spot rings) from each spot to the nearest source spot (BFS)."""
    d = np.full(len(nbrs), np.inf)
    q = deque()
    for s in np.where(sources)[0]:
        d[s] = 0
        q.append(s)
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if d[j] == np.inf:
                d[j] = d[i] + 1
                q.append(j)
    return d


def weight_matrix(nbrs: list[np.ndarray]):
    """Row-standardised sparse weight matrix of the neighbour graph."""
    import scipy.sparse as spm

    rows = np.concatenate([np.full(len(n), i) for i, n in enumerate(nbrs)]) if nbrs else np.array([], int)
    cols = np.concatenate(nbrs) if nbrs else np.array([], int)
    deg = np.array([max(len(n), 1) for n in nbrs], float)
    vals = 1.0 / deg[rows]
    return spm.csr_matrix((vals, (rows, cols)), shape=(len(nbrs), len(nbrs)))


def morans_i(x: np.ndarray, nbrs, n_perm: int, rng, W=None) -> tuple[float, float]:
    """Moran's I with row-standardised weights; two-sided permutation p (vectorised)."""
    W = weight_matrix(nbrs) if W is None else W
    z = np.asarray(x, float) - np.mean(x)
    denom = float((z**2).sum())
    if denom == 0:
        return np.nan, np.nan
    I = float(z @ (W @ z) / denom)
    Zp = np.column_stack([rng.permutation(z) for _ in range(n_perm)])
    null = np.einsum("ij,ij->j", Zp, W @ Zp) / denom
    p = (1 + np.sum(np.abs(null) >= abs(I))) / (1 + n_perm)
    return I, float(p)


def _gene(a, g) -> np.ndarray:
    x = a.raw[:, g].X if a.raw is not None else a[:, g].X
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x).ravel()


def preprocess_spatial(a, seed: int | None = None):
    seed = get_settings().seed if seed is None else seed
    p = SPATIAL_PARAMS
    rec = new_record("spatial_preprocess", AnalysisKind.EXPLORATORY, [a.uns.get("dataset", {}).get("dataset_id", "visium")], "scanpy spot QC, normalise_total (1e4), log1p, HVG, PCA, kNN, Leiden domains; programme scores (score_genes)", dict(p), seed=seed)
    a = a.copy()
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None)
    n0 = a.n_obs
    a = a[(a.obs["total_counts"] >= p["min_counts"]) & (a.obs["n_genes_by_counts"] >= p["min_genes"])].copy()
    sc.pp.filter_genes(a, min_cells=10)
    rec.n_used, rec.n_excluded = int(a.n_obs), int(n0 - a.n_obs)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    sc.pp.highly_variable_genes(a, n_top_genes=p["n_top_hvg"], flavor="seurat")
    sc.tl.pca(a, n_comps=p["n_pcs"], mask_var="highly_variable", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=p["n_neighbors"], n_pcs=p["n_pcs"], random_state=seed)
    sc.tl.leiden(a, resolution=p["leiden_resolution"], random_state=seed, flavor="igraph", n_iterations=2, directed=False, key_added="domain")
    for name, genes in PROGRAMMES.items():
        found = [g for g in genes if g in a.raw.var_names]
        sc.tl.score_genes(a, found, score_name=f"prog_{name}", random_state=seed, use_raw=True)
    rec.limitations += ["Visium spots (~55 um) contain several cells; a spot score mixes cell types.", "Single section from a single patient: results describe this tissue, not glioblastoma in general."]
    return a, finish(rec)


def annotate_domains(a) -> pd.DataFrame:
    progs = [c for c in a.obs.columns if c.startswith("prog_")]
    tab = a.obs.groupby("domain", observed=True)[progs].mean()
    z = (tab - tab.mean()) / tab.std(ddof=0).replace(0, 1)
    label = z.idxmax(axis=1).str.replace("prog_", "", regex=False) + "-enriched"
    tab["n_spots"] = a.obs["domain"].value_counts().reindex(tab.index)
    tab["dominant_programme"] = label
    a.obs["domain_label"] = a.obs["domain"].map(lambda d: f"D{d} {label[d]}")
    a.obs["domain_programme"] = a.obs["domain"].map(lambda d: label[d])
    return tab


def spatial_programme_analysis(a, candidate_genes: list[str], n_perm: int | None = None, seed: int | None = None) -> tuple[dict, object]:
    seed = get_settings().seed if seed is None else seed
    n_perm = n_perm or SPATIAL_PARAMS["n_perm"]
    rng = np.random.default_rng(seed)
    found = [g for g in candidate_genes if g in a.raw.var_names]
    sc.tl.score_genes(a, found, score_name="candidate_score", random_state=seed, use_raw=True)
    nbrs = hex_neighbours(a.obs)
    W = weight_matrix(nbrs)
    rec = new_record("spatial_programmes", AnalysisKind.EXPLORATORY, [a.uns.get("dataset", {}).get("dataset_id", "visium")], "Spearman spot correlations; Moran's I on the hexagonal neighbour graph (row-standardised, permutation p); BH across genes", {"genes": candidate_genes, "n_perm": n_perm}, seed=seed)
    neuron = a.obs["prog_neuron"].values
    corr = []
    for g in found + ["candidate_score"]:
        x = a.obs["candidate_score"].values if g == "candidate_score" else _gene(a, g)
        r_n, p_n = stats.spearmanr(x, neuron)
        r_t, p_t = stats.spearmanr(x, a.obs["prog_tumour_glial"].values)
        I, pI = morans_i(x, nbrs, n_perm, rng, W)
        corr.append({"feature": g, "detected_frac": float((x > 0).mean()), "rho_neuron": float(r_n), "p_neuron": float(p_n), "rho_tumour_glial": float(r_t), "p_tumour_glial": float(p_t), "morans_I": I, "morans_p": pI})
    df = pd.DataFrame(corr)
    df["q_neuron"] = bh_adjust(df["p_neuron"].values)
    df["q_morans"] = bh_adjust(df["morans_p"].values)
    rec.n_used = int(a.n_obs)
    rec.limitations += ["Spot-level correlations are inflated by spatial autocorrelation (neighbouring spots are not independent); read rho as an effect size, not as an independent-sample test.",
                        "Moran's I permutation null is spot-exchangeable; significant I indicates spatial structure, not its cause."]
    return {"correlations": df, "neighbours": nbrs}, finish(rec)


def interface_analysis(a, genes: list[str], nbrs: list[np.ndarray], quantile: float | None = None) -> tuple[pd.DataFrame, pd.Series, object]:
    """Neuron-rich spots = top quantile of the neuron programme. Among the remaining spots, relate gene expression to
    ring distance from the nearest neuron-rich spot."""
    quantile = quantile or SPATIAL_PARAMS["neuron_rich_quantile"]
    neuron = a.obs["prog_neuron"].values
    rich = neuron >= np.quantile(neuron, quantile)
    d = ring_distance(nbrs, rich)
    dist = pd.Series(d, index=a.obs_names, name="rings_to_neuron_rich")
    a.obs["rings_to_neuron_rich"] = d
    a.obs["neuron_rich"] = rich
    other = ~rich & np.isfinite(d)
    near = other & (d <= 2)
    far = other & (d >= 5)
    rows = []
    for g in genes + ["candidate_score"]:
        x = a.obs["candidate_score"].values if g == "candidate_score" else _gene(a, g)
        rho, p = stats.spearmanr(x[other], d[other])
        # partial Spearman controlling for the spot's own neuron programme (partial-volume check)
        rx = stats.rankdata(x[other]); rd = stats.rankdata(d[other]); rn = stats.rankdata(neuron[other])
        ex = rx - np.polyval(np.polyfit(rn, rx, 1), rn); ed = rd - np.polyval(np.polyfit(rn, rd, 1), rn)
        rho_adj, p_adj = stats.pearsonr(ex, ed)
        gg, lo, hi = hedges_g(x[near], x[far])
        u = stats.mannwhitneyu(x[near], x[far], alternative="two-sided") if near.sum() > 5 and far.sum() > 5 else None
        rows.append({"feature": g, "rho_vs_distance": float(rho), "p_distance": float(p), "rho_vs_distance_neuron_adjusted": float(rho_adj), "p_distance_neuron_adjusted": float(p_adj), "mean_interface": float(x[near].mean()), "mean_distal": float(x[far].mean()), "hedges_g_interface_vs_distal": gg, "g_ci_low": lo, "g_ci_high": hi, "cliffs_delta": cliffs_delta(x[near], x[far]), "p_interface_vs_distal": float(u.pvalue) if u else np.nan})
    df = pd.DataFrame(rows)
    df["q_distance"] = bh_adjust(df["p_distance"].values)
    df["q_distance_neuron_adjusted"] = bh_adjust(df["p_distance_neuron_adjusted"].values)
    rec = new_record("spatial_interface", AnalysisKind.EXPLORATORY, [a.uns.get("dataset", {}).get("dataset_id", "visium")], "neuron-rich spots = top quantile of neuron programme; BFS ring distance on hexagonal grid; Spearman vs distance and interface (1-2 rings) vs distal (>=5 rings) Hedges' g / Mann-Whitney, non-neuron-rich spots only", {"quantile": quantile, "n_neuron_rich": int(rich.sum()), "n_interface": int(near.sum()), "n_distal": int(far.sum())})
    rec.n_used = int(other.sum())
    rec.limitations += ["Interface spots may still contain neurons below the neuron-rich threshold (partial-volume effect), so a gradient can arise from residual neuronal content.",
                        "Spots are spatially autocorrelated; p-values treat them as independent and are optimistic."]
    return df, dist, finish(rec)


def neighbour_lr(a, nbrs: list[np.ndarray], n_perm: int | None = None, seed: int | None = None) -> tuple[pd.DataFrame, object]:
    seed = get_settings().seed if seed is None else seed
    n_perm = n_perm or SPATIAL_PARAMS["n_perm"]
    rng = np.random.default_rng(seed)
    dom = a.obs["domain"].astype(str).values
    dom_idx = {d: np.where(dom == d)[0] for d in np.unique(dom)}
    rows = []
    for lig, rec_, note in LR_PAIRS:
        if lig not in a.raw.var_names or rec_ not in a.raw.var_names:
            continue
        L, R = _gene(a, lig), _gene(a, rec_)
        Rn = np.array([R[n].mean() if len(n) else 0.0 for n in nbrs])
        obs = float((L * Rn).mean())
        glob = np.array([(rng.permutation(L) * Rn).mean() for _ in range(n_perm)])
        strat = np.empty(n_perm)
        for k in range(n_perm):
            Lp = L.copy()
            for idx in dom_idx.values():
                Lp[idx] = rng.permutation(L[idx])
            strat[k] = (Lp * Rn).mean()
        rows.append({"ligand": lig, "receptor": rec_, "annotation": note, "score": obs, "ligand_detected": float((L > 0).mean()), "receptor_detected": float((R > 0).mean()),
                     "z_global": float((obs - glob.mean()) / (glob.std() + 1e-12)), "p_global": float((1 + np.sum(glob >= obs)) / (1 + n_perm)),
                     "z_domain": float((obs - strat.mean()) / (strat.std() + 1e-12)), "p_domain": float((1 + np.sum(strat >= obs)) / (1 + n_perm))})
    df = pd.DataFrame(rows)
    df["q_global"] = bh_adjust(df["p_global"].values)
    df["q_domain"] = bh_adjust(df["p_domain"].values)
    rec = new_record("spatial_ligand_receptor", AnalysisKind.EXPLORATORY, [a.uns.get("dataset", {}).get("dataset_id", "visium")], "mean over spots of ligand(spot) x mean receptor(hexagonal neighbours) on log1p(CP10k); global and domain-stratified ligand permutation nulls; BH", {"n_perm": n_perm, "pairs": [p[:2] for p in LR_PAIRS]}, seed=seed)
    rec.n_used = int(a.n_obs)
    rec.limitations += ["Transcript co-localisation in neighbouring spots is not evidence of protein interaction or synapse formation.",
                        "The domain-stratified null controls for shared tissue composition at domain level only; within-domain gradients can still produce co-localisation.",
                        f"Minimum attainable p = {1 / (n_perm + 1):.4f}."]
    return df.sort_values("p_domain").reset_index(drop=True), finish(rec)
