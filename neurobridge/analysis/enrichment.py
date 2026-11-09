"""Pathway enrichment: over-representation (Enrichr online, or local hypergeometric on cached libraries) and
pre-ranked GSEA (gseapy, local)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobridge.analysis.common import finish, new_record
from neurobridge.config import get_settings
from neurobridge.knowledge.genesets import load_library, library_metadata
from neurobridge.schemas.analysis import AnalysisKind, EnrichmentResult


def over_representation(genes: list[str], library: str, background: list[str], dataset_id: str, *, top_n: int = 25) -> EnrichmentResult:
    """Local hypergeometric ORA against a cached Enrichr library with the measured genes as background.
    Running locally with an explicit background avoids the Enrichr default (whole-genome) background,
    which inflates enrichment for tissue-restricted gene sets."""
    import gseapy as gp

    sets = load_library(library)
    bg = sorted(set(background))
    enr = gp.enrich(gene_list=list(genes), gene_sets=sets, background=bg, outdir=None, verbose=False)
    res = enr.results.sort_values("Adjusted P-value").head(top_n)
    terms = []
    for _, r in res.iterrows():
        k, n = str(r["Overlap"]).split("/")
        terms.append({"term": r["Term"], "overlap": r["Overlap"], "k": int(k), "set_size": int(n), "p": float(r["P-value"]), "q": float(r["Adjusted P-value"]), "odds_ratio": float(r["Odds Ratio"]), "genes": r["Genes"]})
    meta = library_metadata(library) or {}
    rec = new_record("over_representation", AnalysisKind.EXPLORATORY, [dataset_id], "hypergeometric over-representation (gseapy.enrich) with measured-gene background, BH FDR", {"library": library, "n_genes": len(genes), "background_size": len(bg), "library_retrieved_at": meta.get("retrieved_at")})
    rec.limitations.append("Gene-set libraries are curated resources with uneven coverage; neural gene sets are large and overlapping, so several related terms typically co-enrich.")
    return EnrichmentResult(record=finish(rec), library=library, method="ORA (hypergeometric)", terms=terms, n_input_genes=len(genes), background=f"{len(bg)} measured genes")


def prerank_gsea(ranking: pd.Series, library: str, dataset_id: str, *, permutations: int = 1000, min_size: int = 15, max_size: int = 500, top_n: int = 25, seed: int | None = None) -> EnrichmentResult:
    """Pre-ranked GSEA (gseapy.prerank) on a gene-level statistic (e.g. signed -log10 p or Hedges' g)."""
    import gseapy as gp

    seed = get_settings().seed if seed is None else seed
    sets = load_library(library)
    rnk = ranking.dropna().sort_values(ascending=False)
    rnk = rnk[~rnk.index.duplicated()]
    pre = gp.prerank(rnk=rnk, gene_sets=sets, permutation_num=permutations, min_size=min_size, max_size=max_size, seed=seed, threads=4, outdir=None, verbose=False)
    res = pre.res2d.copy()
    res["FDR q-val"] = pd.to_numeric(res["FDR q-val"], errors="coerce")
    res["NES"] = pd.to_numeric(res["NES"], errors="coerce")
    res = res.sort_values("FDR q-val").head(top_n)
    terms = []
    for _, r in res.iterrows():
        tag = str(r.get("Tag %", ""))
        set_size = int(tag.split("/")[1]) if "/" in tag else None
        terms.append({"term": r["Term"], "nes": float(r["NES"]), "es": float(r["ES"]), "p": float(r["NOM p-val"]), "q": float(r["FDR q-val"]), "set_size": set_size, "tag": tag, "lead_genes": r.get("Lead_genes", "")})
    meta = library_metadata(library) or {}
    rec = new_record("prerank_gsea", AnalysisKind.EXPLORATORY, [dataset_id], f"pre-ranked GSEA (gseapy), {permutations} gene-set permutations, weighted enrichment statistic", {"library": library, "permutations": permutations, "min_size": min_size, "max_size": max_size, "n_ranked_genes": int(len(rnk)), "library_retrieved_at": meta.get("retrieved_at")}, seed=seed)
    return EnrichmentResult(record=finish(rec), library=library, method="pre-ranked GSEA", terms=terms, n_input_genes=int(len(rnk)))
