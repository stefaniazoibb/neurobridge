"""Scientific Reviewer Agent: audits the claims and the execution log with explicit rules."""

from __future__ import annotations

import re

from neurobridge.agents.agents import Agent, Context
from neurobridge.agents import llm
from neurobridge.schemas.agents import ToolStatus
from pydantic import BaseModel, Field

CAUSAL = re.compile(r"\b(drives?|causes?|caused|leads? to|induces?|promotes?|regulates?|mediates?|controls?)\b", re.I)


class ReviewNarrative(BaseModel):
    overall_assessment: str = Field(description="3-5 sentences, plain language, no new numbers")
    main_concerns: list[str]
    suggested_experiments: list[str]


class ReviewerAgent(Agent):
    name, role = "Scientific Reviewer Agent", "audit conclusions against evidence, rules and limitations"

    def execute(self, ctx: Context) -> None:
        claims = ctx.ws.claims
        issues: list[dict] = []
        for c in claims:
            if c["kind"] in ("exploratory", "confirmatory", "sensitivity") and CAUSAL.search(c["text"]):
                issues.append({"severity": "major", "claim": c["id"], "issue": "causal verb in a claim derived from observational analysis", "action": "rephrase as association"})
            if c["kind"] == "literature" and not c["citations"]:
                issues.append({"severity": "major", "claim": c["id"], "issue": "literature claim without a verified citation", "action": "removed"})
                c["status"] = "removed"
            if c["kind"] in ("exploratory", "confirmatory", "sensitivity") and not c["analyses"]:
                issues.append({"severity": "minor", "claim": c["id"], "issue": "claim not linked to an analysis id"})
            if "replicat" in c["text"].lower() and c["kind"] == "confirmatory":
                if "not replicated" not in c["text"] and c["status"] == "supported" and c["level"] == "insufficient evidence":
                    issues.append({"severity": "major", "claim": c["id"], "issue": "replication claimed without replicating cohorts"})
        o = ctx.outputs
        n_val = len([d for d in ("cgga_325", "cgga_693") if d in ctx.ws.cohorts])
        if n_val == 0:
            issues.append({"severity": "major", "claim": "*", "issue": "only one cohort analysed: no result may be described as validated"})
            for c in claims:
                if c["kind"] == "confirmatory":
                    c["status"] = "not validated (single cohort)"
        comp = o.get("composition")
        if comp:
            hi = [r for r in comp.data["rows"] if r["spearman_rho"] > 0.5]
            if hi:
                issues.append({"severity": "major", "claim": "bulk", "issue": f"candidate score strongly tracks normal-neuron markers in {len(hi)}/{len(comp.data['rows'])} cohorts (rho > 0.5): bulk associations are confounded by sample composition", "action": "report composition-adjusted results alongside primary results"})
        for d in ("tcga_gbm_rnaseq", "cgga_325", "cgga_693"):
            so = o.get(f"survival_{d}")
            if so and so.data.get("ph_ok") is False:
                issues.append({"severity": "minor", "claim": f"O_{d}", "issue": "proportional-hazards assumption questionable; HR is a time-averaged effect"})
        for c in claims:
            if c["id"].startswith("O_") and "opposite direction" in c["text"] and c["status"] == "supported":
                issues.append({"severity": "major", "claim": c["id"], "issue": "significant survival association in a validation cohort points opposite to discovery; cannot be read as support for the hypothesis", "action": "reported as heterogeneity, not replication"})
        if o.get("projection"):
            issues.append({"severity": "minor", "claim": "S2", "issue": f"validation-cohort subtypes are classifier projections (TCGA CV balanced accuracy {o['projection'].data['cv_bacc']:.2f}); misclassification biases between-subtype differences toward zero"})
        if o.get("single_cell"):
            issues.append({"severity": "minor", "claim": "SC1", "issue": "single-cell tests treat cells as independent replicates (4 patients); p-values are optimistic"})
        failed = [t for t in ctx.log.tool_calls if t.status == ToolStatus.FAILED]
        for t in failed:
            issues.append({"severity": "major", "claim": "*", "issue": f"tool {t.tool} failed ({t.error}); dependent conclusions are missing"})
        for u in ctx.ws.unavailable:
            issues.append({"severity": "major", "claim": "*", "issue": f"dataset {u['dataset_id']} unavailable: {u['consequence']}"})
        n_conf = len([c for c in claims if c["kind"] == "confirmatory"])
        n_conf_sup = len([c for c in claims if c["kind"] == "confirmatory" and c["status"] == "supported"])
        highest = "statistical association" if any(c["level"] == "statistical association" and c["kind"] == "confirmatory" and c["status"] == "supported" for c in claims) else "insufficient for any confirmatory claim"
        supported = [c["id"] for c in claims if c["status"] == "supported" and c["kind"] != "literature"]
        uncertain = [
            "Whether the neuronal-activity signal in bulk tumours originates from tumour cells or from infiltrated normal neurons and glia.",
            "Whether neuron-glioma synaptic communication (as opposed to transcript abundance) is associated with patient outcome.",
            "Whether projected subtype labels in CGGA are accurate enough to estimate effect sizes precisely.",
        ]
        if o.get("val_survival") and "not replicated" in o["val_survival"].summary:
            uncertain.append("The survival association: estimates point in opposite directions across cohorts and are compatible with no effect.")
        review = {"issues": issues, "n_claims": len(claims), "n_confirmatory": n_conf, "n_confirmatory_supported": n_conf_sup, "highest_evidence_level_reached": highest, "evidence_ceiling_of_design": ctx.plan.evidence_ceiling if ctx.plan else None,
                  "supported_claims": supported, "remains_uncertain": uncertain, "narrative": None, "narrative_source": None}
        if ctx.options.get("use_llm", True) and llm.provider_status()["available"]:
            try:
                import json

                nar = llm.structured(
                    system="You are a sceptical scientific reviewer. Critique the conclusions using ONLY the supplied claims and issues. Do not introduce new numbers or citations. Distinguish association from mechanism and causation.",
                    prompt=json.dumps({"question": ctx.question, "claims": claims, "issues": issues}, default=str)[:60000],
                    schema=ReviewNarrative,
                )
                review["narrative"] = nar.model_dump()
                review["narrative_source"] = "LLM-generated (Claude), constrained to the claims above"
            except llm.LLMUnavailable as exc:
                self.limitations.append(f"LLM narrative unavailable: {exc}")
        ctx.outputs["review"] = review
        self.outputs = {"issues": len(issues), "major": len([i for i in issues if i["severity"] == "major"]), "highest_evidence": highest, "supported_claims": supported}
