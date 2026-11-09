"""Specialised agents. Each agent decides which tools to call from the plan and from what earlier agents
produced, adapts when inputs are missing, and reports inputs, actions, outputs and limitations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from neurobridge.agents.executor import ToolExecutor
from neurobridge.agents.workspace import Workspace
from neurobridge.provenance import utc_now_iso
from neurobridge.schemas.agents import AgentMessage, ExecutionLog, ResearchPlan, ToolStatus

PRIMARY = "neuronal_activity_candidates"


@dataclass
class Context:
    question: str
    plan: Optional[ResearchPlan]
    ws: Workspace
    ex: ToolExecutor
    log: ExecutionLog
    options: dict = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)  # tool outputs keyed by a short name
    progress: Optional[Callable[[dict], None]] = None

    def wants(self, workflow: str) -> bool:
        return self.plan is not None and workflow in self.plan.requested_workflows


class Agent:
    name = "agent"
    role = ""

    def __init__(self) -> None:
        self.actions: list[str] = []
        self.limitations: list[str] = []
        self.outputs: dict[str, Any] = {}
        self.status = ToolStatus.SUCCEEDED

    def call(self, ctx: Context, key: str, tool: str, **inputs):
        out = ctx.ex.run(tool, self.name, **inputs)
        self.actions.append(f"{tool}({', '.join(f'{k}={v}' for k, v in inputs.items() if not isinstance(v, list) or len(v) < 6)})")
        if out is None:
            self.status = ToolStatus.DEGRADED
            last = ctx.log.tool_calls[-1]
            self.limitations.append(f"{tool} failed: {last.error}")
            return None
        ctx.outputs[key] = out
        self.outputs[key] = out.summary
        self.limitations += [l for l in out.limitations if l not in self.limitations]
        return out

    def execute(self, ctx: Context) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, ctx: Context, inputs: dict) -> AgentMessage:
        if ctx.progress:
            ctx.progress({"kind": "agent_started", "agent": self.name, "role": self.role})
        try:
            self.execute(ctx)
        except Exception as exc:  # noqa: BLE001
            self.status = ToolStatus.FAILED
            self.limitations.append(f"agent error: {type(exc).__name__}: {exc}")
        msg = AgentMessage(agent=self.name, stage=self.role, timestamp=utc_now_iso(), inputs=inputs, actions=self.actions, outputs=self.outputs, limitations=self.limitations, status=self.status)
        ctx.log.messages.append(msg)
        if ctx.progress:
            ctx.progress({"kind": "agent_finished", "agent": self.name, "status": self.status.value})
        return msg


class HypothesisAgent(Agent):
    name, role = "Hypothesis Agent", "decompose the question into testable hypotheses and select workflows"

    def execute(self, ctx: Context) -> None:
        from neurobridge.agents.planner import make_plan

        plan, queries = make_plan(ctx.question, use_llm=ctx.options.get("use_llm", True), workflows_override=ctx.options.get("workflows"))
        ctx.plan = plan
        ctx.options["literature_queries"] = queries
        self.actions.append(f"planner={plan.planner}")
        self.outputs = {"sub_questions": plan.sub_questions, "hypotheses": [h["hypothesis"] for h in plan.hypotheses], "workflows": plan.requested_workflows, "candidate_genes": plan.candidate_genes, "evidence_ceiling": plan.evidence_ceiling}
        self.limitations += plan.notes + ["Hypotheses are pre-specified before analysis; contrasts chosen after looking at the discovery cohort are labelled exploratory and must be tested in validation cohorts."]


class LiteratureAgent(Agent):
    name, role = "Literature Agent", "retrieve and organise verified literature"

    def execute(self, ctx: Context) -> None:
        if not ctx.wants("literature"):
            return
        out = self.call(ctx, "literature", "search_literature", queries=ctx.options.get("literature_queries", []), page_size=8)
        if out is None:
            self.limitations.append("No literature could be retrieved; biological rationale relies on the curated reference list only if cached.")


class DataAgent(Agent):
    name, role = "Data Agent", "discover, load and inspect datasets"

    def execute(self, ctx: Context) -> None:
        self.call(ctx, "discover", "discover_datasets", modalities=["bulk RNA-seq", "single-cell RNA-seq", "survival"], live_listing=ctx.options.get("live_listing", True))
        wanted = list(ctx.options.get("datasets") or ctx.plan.datasets_preferred)
        for did in wanted:
            out = self.call(ctx, f"load_{did}", "load_expression_data", dataset_id=did)
            if out is None:
                ctx.ws.unavailable.append({"dataset_id": did, "consequence": _consequence(did)})
                continue
            if did != "gse84465_darmanis":
                self.call(ctx, f"inspect_{did}", "inspect_clinical_metadata", dataset_id=did)
        if "tcga_gbm_rnaseq" not in ctx.ws.cohorts:
            self.status = ToolStatus.FAILED
            self.limitations.append("Discovery cohort unavailable: bulk analyses cannot run.")


def _consequence(did: str) -> str:
    return {
        "tcga_gbm_rnaseq": "No discovery cohort: subtype, survival and DE analyses are skipped.",
        "cgga_325": "One fewer validation cohort; replication assessed on remaining cohorts only.",
        "cgga_693": "One fewer validation cohort; replication assessed on remaining cohorts only.",
        "gse84465_darmanis": "No single-cell localisation: cannot tell whether candidate genes are expressed by tumour cells.",
    }.get(did, "Dataset-specific analyses skipped.")


class BioinformaticsAgent(Agent):
    name, role = "Bioinformatics Agent", "molecular analyses: scores, subtypes, DE, enrichment, co-expression, cell types"

    def execute(self, ctx: Context) -> None:
        if "tcga_gbm_rnaseq" in ctx.ws.cohorts:
            d = "tcga_gbm_rnaseq"
            panel = "user_genes" if "user_genes" in ctx.plan.gene_set_ids else PRIMARY
            ctx.options["panel"] = panel
            self.call(ctx, "score", "score_signature", dataset_id=d, panel_id=panel)
            if ctx.wants("subtype"):
                self.call(ctx, "subtype_tcga", "compare_molecular_groups", dataset_id=d, panel_id=panel)
                self.call(ctx, "subtype_tcga_ext", "compare_molecular_groups", dataset_id=d, panel_id="neuron_glioma_extended")
                self.call(ctx, "subtype_tcga_kegg", "compare_molecular_groups", dataset_id=d, panel_id="kegg_glutamatergic_synapse")
            if ctx.wants("correlation") or ctx.wants("subtype") or ctx.wants("differential"):
                self.call(ctx, "correlation", "compute_gene_correlation", dataset_id=d, panel_id="neuron_glioma_extended")
            if ctx.wants("differential"):
                de = self.call(ctx, "de", "perform_differential_expression", dataset_id=d, panel_id=panel)
                if de is not None:
                    self.call(ctx, "ora_up", "perform_pathway_enrichment", de_ref=de.data["ref"], library="GO_Biological_Process_2023", method="ora", direction="up")
                    self.call(ctx, "ora_down", "perform_pathway_enrichment", de_ref=de.data["ref"], library="GO_Biological_Process_2023", method="ora", direction="down")
                    self.call(ctx, "gsea", "perform_pathway_enrichment", de_ref=de.data["ref"], library="MSigDB_Hallmark_2020", method="prerank", permutations=ctx.options.get("gsea_permutations", 1000))
        else:
            self.limitations.append("Bulk analyses skipped: discovery cohort not loaded.")
        if ctx.wants("single_cell"):
            if ctx.ws.single_cell.get("gse84465_darmanis:raw") is not None:
                self.call(ctx, "single_cell", "perform_cell_type_analysis", dataset_id="gse84465_darmanis", panel_id=PRIMARY)
            else:
                self.limitations.append("Single-cell analysis skipped: GSE84465 unavailable.")


class StatisticsAgent(Agent):
    name, role = "Statistics Agent", "survival models, assumption checks, confounding"

    def execute(self, ctx: Context) -> None:
        panel = ctx.options.get("panel", PRIMARY)
        bulk = [d for d in ("tcga_gbm_rnaseq", "cgga_325", "cgga_693") if d in ctx.ws.cohorts]
        if ctx.wants("survival"):
            for d in bulk:
                self.call(ctx, f"survival_{d}", "perform_survival_analysis", dataset_id=d, panel_id=panel)
        if ctx.wants("composition") and bulk:
            self.call(ctx, "composition", "assess_composition_confounding", dataset_ids=bulk, panel_id=panel)


class ValidationAgent(Agent):
    name, role = "Validation Agent", "independent-cohort replication and sensitivity analyses"

    def execute(self, ctx: Context) -> None:
        panel = ctx.options.get("panel", PRIMARY)
        val = [d for d in ("cgga_325", "cgga_693") if d in ctx.ws.cohorts]
        if "tcga_gbm_rnaseq" not in ctx.ws.cohorts:
            self.limitations.append("No discovery cohort: validation not possible.")
            return
        if not val:
            self.limitations.append("No independent cohort available: every result remains single-cohort and is reported as NOT validated.")
        if ctx.wants("subtype") and val:
            self.call(ctx, "projection", "project_subtypes", train_id="tcga_gbm_rnaseq", target_ids=val)
            self.call(ctx, "val_subtype", "validate_across_cohorts", hypothesis="subtype_difference", validation_ids=val, panel_id=panel)
            self.call(ctx, "val_subtype_resid", "validate_across_cohorts", hypothesis="subtype_difference", validation_ids=val, panel_id=panel, residualise_on_neuron_markers=True)
        if ctx.wants("survival"):
            self.call(ctx, "val_survival", "validate_across_cohorts", hypothesis="survival_association", validation_ids=val, panel_id=panel)
            self.call(ctx, "sensitivity", "run_sensitivity_analysis", dataset_ids=["tcga_gbm_rnaseq"] + val, panel_id=panel)
