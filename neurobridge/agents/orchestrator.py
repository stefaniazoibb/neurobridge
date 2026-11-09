"""Orchestrator: runs the agents in dependency order, keeps the execution log, survives partial failure,
builds claims, the review and the report, and persists the investigation for reproducibility."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Callable, Optional

from neurobridge.agents.agents import BioinformaticsAgent, Context, DataAgent, HypothesisAgent, LiteratureAgent, StatisticsAgent, ValidationAgent
from neurobridge.agents.claims import build_claims
from neurobridge.agents.executor import ToolExecutor
from neurobridge.agents.reviewer import ReviewerAgent
from neurobridge.agents.workspace import Workspace
from neurobridge.config import get_settings
from neurobridge.provenance import software_versions, utc_now_iso
from neurobridge.schemas.agents import ExecutionLog, InvestigationResult

PIPELINE = [HypothesisAgent, LiteratureAgent, DataAgent, BioinformaticsAgent, StatisticsAgent, ValidationAgent]
DEMO_QUESTION = "Investigate whether neuronal activity-related signalling is associated with glioblastoma molecular subtypes and patient survival."


class Investigation:
    def __init__(self, question: str, options: Optional[dict] = None, progress: Optional[Callable[[dict], None]] = None):
        self.id = f"nb-{utc_now_iso().replace(':', '').replace('-', '')[:15]}-{uuid.uuid4().hex[:6]}"
        self.question = question
        self.options = dict(options or {})
        self.progress = progress
        self.log = ExecutionLog(investigation_id=self.id, started_at=utc_now_iso())
        self.ws = Workspace(self.id)
        self.ctx = Context(question=question, plan=None, ws=self.ws, ex=ToolExecutor(self.ws, self.log, progress), log=self.log, options=self.options, progress=progress)

    def run(self) -> InvestigationResult:
        self.log.add_event("investigation_started", question=self.question, options={k: v for k, v in self.options.items() if k != "literature_queries"})
        for cls in PIPELINE:
            agent = cls()
            agent.run(self.ctx, {"question": self.question} if cls is HypothesisAgent else {"workflows": self.ctx.plan.requested_workflows if self.ctx.plan else []})
            if cls is HypothesisAgent and self.ctx.plan is None:
                break
        self.ws.claims = build_claims(self.ctx)
        ReviewerAgent().run(self.ctx, {"n_claims": len(self.ws.claims)})
        self.ctx.ex.run("generate_scientific_report", "Orchestrator", title="NeuroBridge evidence report")
        from neurobridge.report.builder import build_markdown

        result = self.result()
        result.report_markdown = build_markdown(result, self.ws)
        self.log.finished_at = utc_now_iso()
        self.log.add_event("investigation_finished", n_tool_calls=len(self.log.tool_calls))
        if self.options.get("persist", True):
            self.persist(result)
        return result

    def result(self) -> InvestigationResult:
        failed = [t for t in self.log.tool_calls if t.status.value == "failed"]
        return InvestigationResult(
            investigation_id=self.id,
            question=self.question,
            plan=self.ctx.plan,
            datasets_used=list(self.ws.cohorts) + [k.split(":")[0] for k in self.ws.single_cell if k.endswith(":raw")],
            datasets_unavailable=self.ws.unavailable,
            log=self.log,
            artefacts=self.ws.serialise_artefacts() | {
                "dataset_summaries": {d: c.summary() for d, c in self.ws.cohorts.items()},
                "citations": {p: c.model_dump(mode="json") for p, c in self.ws.citations.items()},
                "outputs": {k: v.model_dump(mode="json") for k, v in self.ctx.outputs.items() if hasattr(v, "model_dump")},
            },
            figures={k: json.loads(v.to_json()) for k, v in self.ws.figures.items()},
            claims=self.ws.claims,
            review=self.ctx.outputs.get("review", {}),
            status="completed with failures" if failed or self.ws.unavailable else "completed",
        )

    def persist(self, result: InvestigationResult) -> Path:
        d = get_settings().runs_dir / self.id
        (d / "figures").mkdir(parents=True, exist_ok=True)
        (d / "tables").mkdir(exist_ok=True)
        payload = result.model_dump(mode="json")
        figs = payload.pop("figures")
        (d / "investigation.json").write_text(json.dumps(payload, indent=1, default=str))
        for k, v in figs.items():
            (d / "figures" / f"{k}.json").write_text(json.dumps(v, default=str))
        for k, df in self.ws.tables.items():
            df.to_csv(d / "tables" / f"{k}.csv")
        (d / "report.md").write_text(result.report_markdown or "")
        from neurobridge.data.cache import read_manifest

        (d / "reproducibility.json").write_text(json.dumps({"investigation_id": self.id, "software": software_versions(), "seed": get_settings().seed, "data_manifest": read_manifest(), "options": {k: v for k, v in self.options.items()}, "started_at": self.log.started_at, "finished_at": self.log.finished_at}, indent=1, default=str))
        return d


def run_investigation(question: str = DEMO_QUESTION, **options) -> tuple[InvestigationResult, Investigation]:
    inv = Investigation(question, options)
    return inv.run(), inv
