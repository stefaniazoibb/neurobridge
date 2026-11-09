"""Reproducibility helpers: software versions, hashes, timestamps, run records."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = [
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "lifelines",
    "scanpy", "anndata", "gseapy", "plotly", "streamlit", "pydantic",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def software_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in TRACKED_PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(obj: Any) -> str:
    """Deterministic hash of a JSON-serialisable object (used for parameter fingerprints)."""
    return sha256_text(json.dumps(obj, sort_keys=True, default=str))[:16]
