"""CELLxGENE Discover curation API (public). Used to list glioblastoma single-cell collections live.
Files are 1-7 GB each, so they are listed with real metadata but not ingested by default."""

from __future__ import annotations

import requests

from neurobridge.config import get_settings

API = "https://api.cellxgene.cziscience.com/curation/v1"
KEYWORDS = ("glioblastoma", "glioma", "gbm")


def list_glioma_collections(timeout: float | None = None) -> list[dict]:
    s = get_settings()
    if not s.allow_network:
        return []
    r = requests.get(f"{API}/collections", timeout=timeout or s.request_timeout_s)
    r.raise_for_status()
    out = []
    for c in r.json():
        name = (c.get("name") or "")
        if any(k in name.lower() for k in KEYWORDS):
            out.append({"collection_id": c["collection_id"], "name": name, "doi": c.get("doi"), "url": f"https://cellxgene.cziscience.com/collections/{c['collection_id']}"})
    return out


def collection_datasets(collection_id: str, timeout: float | None = None) -> list[dict]:
    s = get_settings()
    r = requests.get(f"{API}/collections/{collection_id}", timeout=timeout or s.request_timeout_s)
    r.raise_for_status()
    c = r.json()
    rows = []
    for d in c.get("datasets", []):
        h5ad = [a for a in d.get("assets", []) if a.get("filetype") == "H5AD"]
        rows.append({
            "dataset_id": d["dataset_id"],
            "title": d.get("title"),
            "cell_count": d.get("cell_count"),
            "assays": [a["label"] for a in d.get("assay", [])],
            "disease": [x["label"] for x in d.get("disease", [])],
            "tissue": [x["label"] for x in d.get("tissue", [])],
            "h5ad_size_bytes": h5ad[0].get("filesize") if h5ad else None,
            "h5ad_url": h5ad[0].get("url") if h5ad else None,
        })
    return rows
