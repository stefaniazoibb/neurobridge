"""Thin client for the public cBioPortal REST API (no authentication required)."""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd
import requests

from neurobridge.config import get_settings
from neurobridge.logging_utils import get_logger, log_event

BASE = "https://www.cbioportal.org/api"
log = get_logger(__name__)


class CBioPortalError(RuntimeError):
    pass


def _get(path: str, **params: Any) -> Any:
    s = get_settings()
    if not s.allow_network:
        raise CBioPortalError("network disabled")
    r = requests.get(BASE + path, params=params, timeout=s.request_timeout_s)
    if r.status_code != 200:
        raise CBioPortalError(f"GET {path} -> {r.status_code}: {r.text[:200]}")
    return r.json()


def _post(path: str, body: Any, **params: Any) -> Any:
    s = get_settings()
    if not s.allow_network:
        raise CBioPortalError("network disabled")
    r = requests.post(BASE + path, params=params, data=json.dumps(body), headers={"Content-Type": "application/json"}, timeout=s.request_timeout_s)
    if r.status_code != 200:
        raise CBioPortalError(f"POST {path} -> {r.status_code}: {r.text[:200]}")
    return r.json()


def search_studies(keyword: str = "glioblastoma") -> list[dict]:
    return _get("/studies", keyword=keyword, projection="SUMMARY")


def study_summary(study_id: str) -> dict:
    return _get(f"/studies/{study_id}")


def molecular_profiles(study_id: str) -> list[dict]:
    return _get(f"/studies/{study_id}/molecular-profiles")


def clinical_attributes(study_id: str) -> list[str]:
    return [a["clinicalAttributeId"] for a in _get(f"/studies/{study_id}/clinical-attributes")]


def genes_by_symbol(symbols: list[str]) -> dict[str, int]:
    res = _post("/genes/fetch", symbols, geneIdType="HUGO_GENE_SYMBOL", projection="SUMMARY")
    return {g["hugoGeneSymbol"]: g["entrezGeneId"] for g in res}


def clinical_table(study_id: str, attribute_ids: Optional[list[str]] = None) -> pd.DataFrame:
    """Sample-level table of clinical attributes (patient attributes are propagated to samples)."""
    rows = _get(f"/studies/{study_id}/clinical-data", clinicalDataType="SAMPLE", projection="SUMMARY")
    prow = _get(f"/studies/{study_id}/clinical-data", clinicalDataType="PATIENT", projection="SUMMARY")
    samp = pd.DataFrame(rows).pivot_table(index="sampleId", columns="clinicalAttributeId", values="value", aggfunc="first") if rows else pd.DataFrame()
    pat = pd.DataFrame(prow).pivot_table(index="patientId", columns="clinicalAttributeId", values="value", aggfunc="first") if prow else pd.DataFrame()
    if rows:
        s2p = pd.DataFrame(rows).drop_duplicates("sampleId").set_index("sampleId")["patientId"]
        pat_s = pat.reindex(s2p.loc[samp.index].values)
        pat_s.index = samp.index
        out = pd.concat([samp, pat_s[[c for c in pat_s.columns if c not in samp.columns]]], axis=1)
    else:
        out = samp
    if attribute_ids:
        out = out[[c for c in attribute_ids if c in out.columns]]
    return out


def expression_for_genes(study_id: str, profile_id: str, symbols: list[str]) -> pd.DataFrame:
    """Genes x samples matrix for a short gene list (API is meant for targeted queries, not full matrices)."""
    ids = genes_by_symbol(symbols)
    body = {"entrezGeneIds": list(ids.values()), "sampleListId": f"{study_id}_all"}
    res = _post(f"/molecular-profiles/{profile_id}/molecular-data/fetch", body, projection="SUMMARY")
    inv = {v: k for k, v in ids.items()}
    df = pd.DataFrame(res)
    if df.empty:
        return df
    df["gene"] = df["entrezGeneId"].map(inv)
    mat = df.pivot_table(index="gene", columns="sampleId", values="value", aggfunc="first")
    log_event(log, "cbioportal.expression", study=study_id, profile=profile_id, genes=len(mat), samples=mat.shape[1])
    return mat
