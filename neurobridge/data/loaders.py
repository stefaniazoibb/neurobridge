"""Unified loaders with in-process caching."""

from __future__ import annotations

from functools import lru_cache

from neurobridge.data import cgga, geo, xena
from neurobridge.data.catalog import get_dataset
from neurobridge.data.cohort import Cohort


@lru_cache(maxsize=8)
def load_cohort(dataset_id: str) -> Cohort:
    rec = get_dataset(dataset_id)
    if dataset_id == "tcga_gbm_rnaseq":
        return xena.load_tcga_gbm_rnaseq(rec)
    if dataset_id == "tcga_gbm_u133a":
        return xena.load_tcga_gbm_u133a(rec)
    if dataset_id in ("cgga_325", "cgga_693"):
        return cgga.load_cgga(dataset_id, rec)
    raise KeyError(f"{dataset_id} is not an ingestible bulk cohort")


@lru_cache(maxsize=2)
def load_single_cell(dataset_id: str):
    rec = get_dataset(dataset_id)
    if dataset_id == "gse84465_darmanis":
        return geo.load_darmanis_gse84465(rec)
    raise KeyError(f"{dataset_id} is not an ingestible single-cell dataset")
