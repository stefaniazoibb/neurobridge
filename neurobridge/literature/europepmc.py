"""Europe PMC client with an on-disk cache. Never fabricates records: if the API is unreachable and the
query is not cached, an empty result with an explanatory error is returned."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import requests

from neurobridge.config import get_settings
from neurobridge.logging_utils import get_logger, log_event
from neurobridge.provenance import sha256_text, utc_now_iso
from neurobridge.schemas.evidence import Citation

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
log = get_logger(__name__)


class LiteratureUnavailable(RuntimeError):
    pass


def _cache_dir() -> Path:
    d = get_settings().cache_dir / "literature"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_citation(r: dict, retrieved_at: str) -> Citation:
    pmid = r.get("pmid")
    year = r.get("pubYear")
    return Citation(
        pmid=pmid, doi=r.get("doi"), title=(r.get("title") or "").rstrip("."), authors=r.get("authorString"),
        journal=r.get("journalTitle") or (r.get("journalInfo") or {}).get("journal", {}).get("title"),
        year=int(year) if year and str(year).isdigit() else None, source="Europe PMC",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else (f"https://doi.org/{r['doi']}" if r.get("doi") else None),
        verified=True, retrieved_at=retrieved_at, cited_by=r.get("citedByCount"), abstract=r.get("abstractText"),
    )


def search(query: str, page_size: int = 10, sort: str = "relevance", with_abstracts: bool = False) -> tuple[list[Citation], dict]:
    """Search Europe PMC (MEDLINE records only). Returns (citations, meta). meta['from_cache'] tells whether the result came from disk."""
    q = f"({query}) AND SRC:MED"
    params = {"query": q + (" sort_cited:y" if sort == "cited" else ""), "format": "json", "pageSize": page_size, "resultType": "core" if with_abstracts else "lite"}
    key = sha256_text(json.dumps(params, sort_keys=True))[:20]
    path = _cache_dir() / f"search_{key}.json"
    s = get_settings()
    if s.allow_network:
        try:
            r = requests.get(API, params=params, timeout=30)
            r.raise_for_status()
            payload = {"query": query, "params": params, "retrieved_at": utc_now_iso(), "response": r.json()}
            path.write_text(json.dumps(payload))
            log_event(log, "literature.search", query=query, hits=payload["response"].get("hitCount"))
        except Exception as exc:  # noqa: BLE001
            log_event(log, "literature.search_failed", query=query, error=str(exc))
            if not path.exists():
                raise LiteratureUnavailable(f"Europe PMC unreachable and query not cached: {exc}") from exc
    elif not path.exists():
        raise LiteratureUnavailable("network disabled and query not cached")
    payload = json.loads(path.read_text())
    res = payload["response"].get("resultList", {}).get("result", [])
    cites = [_to_citation(x, payload["retrieved_at"]) for x in res]
    meta = {"query": query, "hit_count": payload["response"].get("hitCount"), "retrieved_at": payload["retrieved_at"], "from_cache": not s.allow_network, "cache_file": path.name}
    return cites, meta


def fetch_pmids(pmids: list[str]) -> list[Citation]:
    """Fetch exact records by PMID (one OR query). Records not found are simply absent from the output."""
    if not pmids:
        return []
    q = " OR ".join(f"EXT_ID:{p}" for p in pmids)
    cites, _ = search(q, page_size=max(25, len(pmids)))
    order = {p: i for i, p in enumerate(pmids)}
    return sorted([c for c in cites if c.pmid in order], key=lambda c: order[c.pmid])
