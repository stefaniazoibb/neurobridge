"""List and reload persisted investigations (data/runs/<id>/)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.io as pio

from neurobridge.config import get_settings
from neurobridge.schemas.agents import InvestigationResult
from neurobridge.schemas.evidence import Citation


@dataclass
class RunView:
    """Duck-typed stand-in for a Workspace when a run is reloaded from disk."""

    figures: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    citations: dict[str, Citation] = field(default_factory=dict)


def list_runs(limit: int = 20) -> list[dict]:
    d = get_settings().runs_dir
    out = []
    for p in sorted(d.glob("nb-*"), reverse=True):
        f = p / "investigation.json"
        if not f.exists():
            continue
        try:
            j = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        out.append({"id": j["investigation_id"], "question": j["question"], "status": j["status"], "started_at": j["log"]["started_at"], "n_tools": len(j["log"]["tool_calls"]), "n_claims": len(j["claims"]), "path": str(p)})
        if len(out) >= limit:
            break
    return out


def load_run(run_id: str) -> tuple[InvestigationResult, RunView]:
    p = get_settings().runs_dir / run_id
    j = json.loads((p / "investigation.json").read_text())
    j["figures"] = {}
    res = InvestigationResult.model_validate(j)
    view = RunView()
    for f in sorted((p / "figures").glob("*.json")):
        view.figures[f.stem] = pio.from_json(f.read_text())
    for f in sorted((p / "tables").glob("*.csv")):
        view.tables[f.stem] = pd.read_csv(f, index_col=0)
    for pm, c in res.artefacts.get("citations", {}).items():
        view.citations[pm] = Citation.model_validate(c)
    return res, view
