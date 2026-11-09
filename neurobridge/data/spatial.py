"""10x Genomics Visium: Human Glioblastoma, Whole Transcriptome Analysis (Space Ranger 1.2.0, CC BY 4.0).
One tissue section from one patient. Files are downloaded from the 10x CDN, recorded in the cache manifest,
and the spatial image archive is extracted with tarfile's 'data' filter into its own directory."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from neurobridge.config import get_settings
from neurobridge.data.cache import cached_file, manifest_entry

BASE = "https://cf.10xgenomics.com/samples/spatial-exp/1.2.0/Parent_Visium_Human_Glioblastoma/Parent_Visium_Human_Glioblastoma_"
FILES = {
    "visium_gbm_matrix": ("filtered_feature_bc_matrix.h5", 1_000_000),
    "visium_gbm_spatial": ("spatial.tar.gz", 100_000),
    "visium_gbm_metrics": ("metrics_summary.csv", 200),
}


def _fetch(key: str) -> Path:
    fname, min_bytes = FILES[key]
    return cached_file(key, BASE + fname, f"spatial/visium_gbm_10x/{fname}", "10x Genomics datasets (CC BY 4.0)", min_bytes=min_bytes)


def _extract_spatial(archive: Path) -> Path:
    out = archive.parent / "extracted"
    if (out / "spatial" / "scalefactors_json.json").exists():
        return out / "spatial"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(out, filter="data")  # refuses absolute paths, links outside, device files
    return out / "spatial"


def read_positions(spatial_dir: Path) -> pd.DataFrame:
    cols = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
    p = spatial_dir / "tissue_positions_list.csv"
    if p.exists():  # Space Ranger < 2.0: no header
        df = pd.read_csv(p, header=None, names=cols)
    else:
        df = pd.read_csv(spatial_dir / "tissue_positions.csv")
    return df.set_index("barcode")


def load_visium_gbm(dataset_record) -> ad.AnnData:
    m = _fetch("visium_gbm_matrix")
    arch = _fetch("visium_gbm_spatial")
    metrics = _fetch("visium_gbm_metrics")
    sdir = _extract_spatial(arch)
    a = sc.read_10x_h5(m)
    a.var_names_make_unique()
    pos = read_positions(sdir).reindex(a.obs_names)
    if pos["in_tissue"].isna().any():
        raise ValueError("barcodes in the matrix lack spatial positions")
    a.obs = a.obs.join(pos)
    a.obsm["spatial"] = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    sf = json.loads((sdir / "scalefactors_json.json").read_text())
    a.uns["spatial_meta"] = {"scalefactors": sf, "lowres_image": str(sdir / "tissue_lowres_image.png"), "hires_image": str(sdir / "tissue_hires_image.png")}
    a.uns["metrics_summary"] = pd.read_csv(metrics).iloc[0].to_dict()
    a.uns["dataset"] = dataset_record.model_dump(mode="json")
    a.uns["provenance"] = {k: manifest_entry(k) for k in FILES}
    a.uns["notes"] = [
        "Space Ranger 1.2.0 filtered feature-barcode matrix (UMI counts), tissue-covered spots only, as distributed by 10x Genomics.",
        "One fresh-frozen glioblastoma section from one patient; spots are ~55 um and contain several cells each.",
    ]
    return a


def lowres_image(a) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(a.uns["spatial_meta"]["lowres_image"]).convert("RGB"))
