"""Shared, typed artefact store for one investigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass
class Workspace:
    investigation_id: str
    cohorts: dict[str, Any] = field(default_factory=dict)  # dataset_id -> Cohort
    single_cell: dict[str, Any] = field(default_factory=dict)  # dataset_id -> processed AnnData
    artefacts: dict[str, Any] = field(default_factory=dict)  # ref -> result object
    figures: dict[str, Any] = field(default_factory=dict)  # figure id -> plotly Figure
    tables: dict[str, Any] = field(default_factory=dict)  # table id -> DataFrame
    citations: dict[str, Any] = field(default_factory=dict)  # pmid -> Citation
    unavailable: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    _counter: int = 0

    def put(self, kind: str, obj: Any) -> str:
        self._counter += 1
        ref = f"{kind}:{self._counter:03d}"
        self.artefacts[ref] = obj
        return ref

    def put_figure(self, fig_id: str, fig) -> str:
        self.figures[fig_id] = fig
        return fig_id

    def put_table(self, table_id: str, df) -> str:
        self.tables[table_id] = df
        return table_id

    def find(self, kind: str) -> list[tuple[str, Any]]:
        return [(k, v) for k, v in self.artefacts.items() if k.startswith(kind + ":")]

    def serialise_artefacts(self) -> dict[str, Any]:
        out = {}
        for k, v in self.artefacts.items():
            if isinstance(v, BaseModel):
                out[k] = v.model_dump(mode="json")
            elif isinstance(v, (dict, list, str, int, float)):
                out[k] = v
            else:
                out[k] = repr(v)[:500]
        return out
