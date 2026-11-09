"""Single-cell analyses with scanpy on the Darmanis GSE84465 cohort (or any AnnData with author cell types)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

from neurobridge.analysis.common import bh_adjust, finish, new_record
from neurobridge.config import get_settings
from neurobridge.schemas.analysis import AnalysisKind, AnalysisRecord

QC_PARAMS = {"min_genes": 500, "min_cells_per_gene": 3, "max_pct_mito": 20.0, "n_top_hvg": 2000, "n_pcs": 30, "n_neighbors": 15, "leiden_resolution": 0.5}


def preprocess(adata: ad.AnnData, params: dict | None = None, seed: int | None = None) -> tuple[ad.AnnData, AnalysisRecord]:
    """QC -> normalise_total -> log1p -> HVG -> PCA -> neighbours -> UMAP -> Leiden. Returns processed AnnData and record."""
    p = dict(QC_PARAMS, **(params or {}))
    seed = get_settings().seed if seed is None else seed
    rec = new_record("sc_preprocess", AnalysisKind.EXPLORATORY, [adata.uns.get("dataset", {}).get("dataset_id", "unknown")], "scanpy QC, library-size normalisation (1e4), log1p, HVG (seurat flavour), PCA, kNN graph, UMAP, Leiden", p, seed=seed)
    a = adata.copy()
    a.layers["counts"] = a.X.copy()
    a.var["mt"] = a.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(a, qc_vars=["mt"], inplace=True, percent_top=None)
    n0 = a.n_obs
    sc.pp.filter_cells(a, min_genes=p["min_genes"])
    a = a[a.obs["pct_counts_mt"] < p["max_pct_mito"]].copy()
    sc.pp.filter_genes(a, min_cells=p["min_cells_per_gene"])
    rec.n_used, rec.n_excluded = int(a.n_obs), int(n0 - a.n_obs)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    a.raw = a
    sc.pp.highly_variable_genes(a, n_top_genes=p["n_top_hvg"], flavor="seurat", batch_key="patient" if "patient" in a.obs else None)
    sc.tl.pca(a, n_comps=p["n_pcs"], mask_var="highly_variable", random_state=seed)
    sc.pp.neighbors(a, n_neighbors=p["n_neighbors"], n_pcs=p["n_pcs"], random_state=seed)
    sc.tl.umap(a, random_state=seed)
    sc.tl.leiden(a, resolution=p["leiden_resolution"], random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    rec.limitations.append("UMAP is a visualisation; distances between clusters are not quantitatively meaningful.")
    rec.limitations.append("Smart-seq2 data from four patients: neoplastic cells cluster by patient; HVG selection is batch-aware by patient but no integration method was applied.")
    return a, finish(rec)


def cell_type_expression(a: ad.AnnData, genes: list[str], group_key: str = "cell_type_author") -> tuple[pd.DataFrame, AnalysisRecord]:
    """Per cell type: fraction of cells with >0 counts and mean log-normalised expression, for each gene."""
    found = [g for g in genes if g in a.raw.var_names]
    rec = new_record("sc_celltype_expression", AnalysisKind.EXPLORATORY, [a.uns.get("dataset", {}).get("dataset_id", "unknown")], "per-cell-type mean log1p(CP10k) and detection fraction", {"genes": genes, "group_key": group_key})
    X = a.raw[:, found].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    df = pd.DataFrame(X, columns=found, index=a.obs_names)
    df[group_key] = a.obs[group_key].values
    mean = df.groupby(group_key, observed=True)[found].mean()
    frac = df.groupby(group_key, observed=True)[found].apply(lambda d: (d > 0).mean())
    n = df.groupby(group_key, observed=True).size()
    out = pd.concat({"mean_log_expr": mean, "fraction_detected": frac}, axis=1)
    out[("n_cells", "")] = n
    missing = [g for g in genes if g not in found]
    if missing:
        rec.warnings.append("genes absent: " + ", ".join(missing))
    small = n[n < 30].index.tolist()
    if small:
        rec.limitations.append(f"cell types with < 30 cells ({small}) give imprecise estimates")
    return out, finish(rec)


def marker_test(a: ad.AnnData, group_key: str, group: str, reference: str = "rest", genes: list[str] | None = None, n_genes: int = 50) -> tuple[pd.DataFrame, AnalysisRecord]:
    """Wilcoxon rank-sum test for one group vs reference (scanpy rank_genes_groups) with BH correction and log2 fold change."""
    rec = new_record("sc_marker_test", AnalysisKind.EXPLORATORY, [a.uns.get("dataset", {}).get("dataset_id", "unknown")], "Wilcoxon rank-sum (scanpy.tl.rank_genes_groups), BH FDR, log2 fold change of mean expression", {"group_key": group_key, "group": group, "reference": reference})
    sc.tl.rank_genes_groups(a, groupby=group_key, groups=[group], reference=reference, method="wilcoxon", use_raw=True, n_genes=a.raw.n_vars)
    df = sc.get.rank_genes_groups_df(a, group=group)
    df = df.rename(columns={"names": "gene", "logfoldchanges": "log2_fc", "pvals": "p", "pvals_adj": "q"})
    if genes:
        df = df[df["gene"].isin(genes)]
    else:
        df = df.head(n_genes)
    rec.limitations.append("Cells are treated as independent replicates (pseudoreplication); p-values are optimistic. Patient-level pseudobulk would be the conservative alternative but only four patients are available.")
    return df.reset_index(drop=True), finish(rec)


def score_cells(a: ad.AnnData, genes: list[str], score_name: str, seed: int | None = None) -> AnalysisRecord:
    seed = get_settings().seed if seed is None else seed
    found = [g for g in genes if g in a.raw.var_names]
    rec = new_record("sc_score_genes", AnalysisKind.EXPLORATORY, [a.uns.get("dataset", {}).get("dataset_id", "unknown")], "scanpy score_genes (mean of gene set minus binned control genes)", {"genes": genes, "score_name": score_name}, seed=seed)
    sc.tl.score_genes(a, found, score_name=score_name, random_state=seed, use_raw=True)
    missing = [g for g in genes if g not in found]
    if missing:
        rec.warnings.append("genes absent: " + ", ".join(missing))
    return finish(rec)


def compare_score_by_group(a: ad.AnnData, score_name: str, group_key: str) -> pd.DataFrame:
    from scipy import stats

    df = a.obs[[score_name, group_key]].dropna()
    groups = df[group_key].unique().tolist()
    rows = []
    for g in groups:
        x = df.loc[df[group_key] == g, score_name].values
        y = df.loc[df[group_key] != g, score_name].values
        u = stats.mannwhitneyu(x, y, alternative="two-sided") if len(x) > 1 and len(y) > 1 else None
        rows.append({"group": g, "n": len(x), "median": float(np.median(x)), "mean": float(np.mean(x)), "p_vs_rest": float(u.pvalue) if u else np.nan})
    out = pd.DataFrame(rows)
    out["q_vs_rest"] = bh_adjust(out["p_vs_rest"].values)
    return out.sort_values("median", ascending=False).reset_index(drop=True)
