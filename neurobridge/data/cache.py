"""Download cache with a JSON manifest (URL, SHA-256, size, timestamp) for every file."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import requests

from neurobridge.config import get_settings
from neurobridge.logging_utils import get_logger, log_event
from neurobridge.provenance import sha256_file, utc_now_iso

log = get_logger(__name__)


class DownloadError(RuntimeError):
    pass


def _manifest_path() -> Path:
    return get_settings().cache_dir / "manifest.json"


def read_manifest() -> dict:
    p = _manifest_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _write_manifest(m: dict) -> None:
    _manifest_path().write_text(json.dumps(m, indent=2, sort_keys=True))


def record_file(key: str, path: Path, url: str, source: str) -> dict:
    m = read_manifest()
    entry = {
        "path": str(path.relative_to(get_settings().cache_dir)),
        "url": url,
        "source": source,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "retrieved_at": utc_now_iso(),
    }
    m[key] = entry
    _write_manifest(m)
    return entry


def cached_file(
    key: str,
    url: str,
    relpath: str,
    source: str,
    *,
    min_bytes: int = 1000,
    retries: int = 3,
    timeout: Optional[float] = None,
) -> Path:
    """Return a local path for `url`, downloading it if needed. Raises DownloadError if offline."""
    s = get_settings()
    path = s.cache_dir / relpath
    if path.exists() and path.stat().st_size >= min_bytes:
        m = read_manifest()
        if key not in m:
            record_file(key, path, url, source)
        return path
    if not s.allow_network:
        raise DownloadError(f"{key}: not cached and network access is disabled (NEUROBRIDGE_ALLOW_NETWORK=false)")
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            log_event(log, "download.start", key=key, url=url, attempt=attempt)
            with requests.get(url, stream=True, timeout=timeout or s.request_timeout_s, allow_redirects=True) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            if tmp.stat().st_size < min_bytes:
                raise DownloadError(f"{key}: downloaded file too small ({tmp.stat().st_size} bytes)")
            tmp.rename(path)
            entry = record_file(key, path, url, source)
            log_event(log, "download.done", key=key, sha256=entry["sha256"], size=entry["size_bytes"])
            return path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log_event(log, "download.retry", key=key, error=str(exc), attempt=attempt)
            time.sleep(min(2**attempt, 10))
    raise DownloadError(f"{key}: download failed after {retries} attempts: {last_err}")


def manifest_entry(key: str) -> Optional[dict]:
    return read_manifest().get(key)
