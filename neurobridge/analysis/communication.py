"""Ligand-receptor co-expression between cell types (CellPhoneDB-style mean-product score with a
cell-label permutation null). Pairs are a small curated list relevant to neuron-glioma signalling;
this is an expression-based inference, not evidence that the interaction occurs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobridge.analysis.common import bh_adjust, finish, new_record
from neurobridge.config import get_settings
from neurobridge.schemas.analysis import AnalysisKind

# (ligand, receptor, annotation). Receptor complexes are represented by one subunit for transparency.
LR_PAIRS = [
    ("NLGN3", "NRXN1", "neuroligin-3 / neurexin-1 (synaptic adhesion; NLGN3 also acts as a secreted mitogen)"),
    ("BDNF", "NTRK2", "BDNF / TrkB"),
    ("SLC17A7", "GRIA2", "glutamatergic presynapse (VGLUT1) / AMPA receptor GluA2 (proxy pairing)"),
    ("SLC17A7", "GRIN2B", "glutamatergic presynapse (VGLUT1) / NMDA receptor GluN2B (proxy pairing)"),
    ("THBS1", "CACNA2D1", "thrombospondin-1 / alpha2delta-1 (synaptogenic)"),
    ("NRXN1", "NLGN1", "neurexin-1 / neuroligin-1"),
    ("PTN", "PTPRZ1", "pleiotrophin / PTPRZ1 (glioma-enriched)"),
    ("SEMA3A", "NRP1", "semaphorin-3A / neuropilin-1"),
]


def ligand_receptor_scores(a, group_key: str = "cell_type_author", n_perm: int = 500, min_cells: int = 10, seed: int | None = None) -> tuple[pd.DataFrame, object]:
    seed = get_settings().seed if seed is None else seed
    rng = np.random.default_rng(seed)
    genes = sorted({g for p in LR_PAIRS for g in p[:2] if g in a.raw.var_names})
    X = a.raw[:, genes].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    expr = pd.DataFrame(X, columns=genes)
    labels = a.obs[group_key].astype(str).values
    types = [t for t, n in pd.Series(labels).value_counts().items() if n >= min_cells]
    keep = np.isin(labels, types)
    expr, labels = expr[keep].reset_index(drop=True), labels[keep]

    def means(lab):
        return expr.groupby(lab).mean()

    obs_m = means(labels)
    perm_m = [means(rng.permutation(labels)) for _ in range(n_perm)]
    rows = []
    for lig, rec_, note in LR_PAIRS:
        if lig not in genes or rec_ not in genes:
            continue
        for s in types:
            for r in types:
                score = float(obs_m.loc[s, lig] * obs_m.loc[r, rec_])
                null = np.array([pm.loc[s, lig] * pm.loc[r, rec_] for pm in perm_m])
                p = float((1 + np.sum(null >= score)) / (1 + n_perm))
                rows.append({"ligand": lig, "receptor": rec_, "sender": s, "receiver": r, "score": score, "ligand_mean": float(obs_m.loc[s, lig]), "receptor_mean": float(obs_m.loc[r, rec_]), "p": p, "annotation": note})
    df = pd.DataFrame(rows)
    if len(df):
        df["q"] = bh_adjust(df["p"].values)
    rec = new_record("ligand_receptor", AnalysisKind.EXPLORATORY, [a.uns.get("dataset", {}).get("dataset_id", "unknown")], "mean(ligand in sender) x mean(receptor in receiver) on log1p(CP10k); one-sided cell-label permutation test; BH FDR across all pairs", {"n_perm": n_perm, "min_cells": min_cells, "pairs": [p[:2] for p in LR_PAIRS]}, seed=seed)
    rec.limitations += ["Co-expression does not show that an interaction occurs (no spatial proximity, protein or functional data).",
                        "Mature neurons are rare in dissociated GBM samples (n=21 here); sender estimates for neurons are imprecise.",
                        "Each complex is represented by one subunit; curated pair list, not a database-wide screen."]
    return df.sort_values(["q", "score"], ascending=[True, False]).reset_index(drop=True), finish(rec)
