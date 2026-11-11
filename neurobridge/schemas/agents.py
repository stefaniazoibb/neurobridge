"""Messages and logs exchanged between the orchestrator, agents and tools."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    DEGRADED = "degraded"


class ToolCall(BaseModel):
    call_id: str
    tool: str
    agent: str
    inputs: dict[str, Any]
    status: ToolStatus = ToolStatus.PENDING
    attempts: int = 0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    output_summary: Optional[str] = None
    output_ref: Optional[str] = Field(None, description="Key into the investigation artefact store")
    error: Optional[str] = None
    limitations: list[str] = Field(default_factory=list)


class AgentMessage(BaseModel):
    agent: str
    stage: str
    timestamp: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    actions: list[str] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    status: ToolStatus = ToolStatus.SUCCEEDED


class ExecutionLog(BaseModel):
    investigation_id: str
    started_at: str
    finished_at: Optional[str] = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)

    def add_event(self, kind: str, **fields: Any) -> None:
        from neurobridge.provenance import utc_now_iso

        self.events.append({"ts": utc_now_iso(), "kind": kind, **fields})


class ResearchPlan(BaseModel):
    question: str
    restated_question: str
    sub_questions: list[str]
    hypotheses: list[dict[str, str]] = Field(default_factory=list, description="hypothesis / null / testable prediction")
    candidate_genes: list[str]
    gene_set_ids: list[str]
    pathways: list[str]
    outcome_variables: list[str]
    molecular_groupings: list[str]
    requested_workflows: list[str]
    datasets_preferred: list[str]
    confounders: list[str]
    evidence_ceiling: str = Field(description="Highest evidence level this design can reach")
    restrictions: list[str] = Field(default_factory=list, description="Population restrictions requested in the question, e.g. IDH-wildtype")
    gene_level: list[str] = Field(default_factory=list, description="Genes named in the question that are analysed individually")
    custom_panels: dict[str, list[str]] = Field(default_factory=dict, description="Investigation-specific gene panels (panel id -> genes), e.g. genes named in the question; resolved through the investigation's Workspace, never registered globally")
    planner: str = Field("deterministic", description="deterministic or LLM-assisted")
    notes: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    investigation_id: str
    question: str
    plan: ResearchPlan
    datasets_used: list[str]
    datasets_unavailable: list[dict[str, str]] = Field(default_factory=list)
    log: ExecutionLog
    artefacts: dict[str, Any] = Field(default_factory=dict, description="Serialised analysis results keyed by reference")
    figures: dict[str, Any] = Field(default_factory=dict, description="Plotly figure JSON keyed by figure id")
    claims: list[dict[str, Any]] = Field(default_factory=list)
    review: dict[str, Any] = Field(default_factory=dict)
    report_markdown: Optional[str] = None
    status: str = "completed"
