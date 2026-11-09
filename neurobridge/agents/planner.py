"""Question decomposition. Deterministic by default; optionally LLM-assisted (validated, never trusted blindly)."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from neurobridge.agents import llm
from neurobridge.knowledge.genesets import CANDIDATE_PANEL, EXTENDED_PANEL, PANELS, register_custom_panel
from neurobridge.schemas.agents import ResearchPlan

WORKFLOWS = {
    "literature": "Retrieve and organise supporting literature",
    "subtype": "Signature score across molecular subtypes (+ cross-cohort replication)",
    "survival": "Survival analysis (KM, Cox) + cross-cohort replication + sensitivity analyses",
    "differential": "Genome-wide differential expression between score-high and score-low tumours + pathway enrichment",
    "correlation": "Candidate-gene co-expression",
    "single_cell": "Cell-type localisation of candidate genes in single-cell data",
    "spatial": "Spatial co-localisation of candidate genes with neuron-rich tissue and ligand-receptor neighbours (Visium)",
    "composition": "Composition confounding control (normal-neuron markers)",
}
KEYWORDS = {
    "survival": ["surviv", "outcome", "prognos", "hazard", "mortality", "death", "kaplan", "cox"],
    "subtype": ["subtype", "subgroup", "molecular group", "mesenchymal", "proneural", "classical", "classif", "stratif"],
    "differential": ["pathway", "enrich", "differential", "programme", "program", "signalling", "signaling"],
    "single_cell": ["single-cell", "single cell", "scrna", "cell type", "cell-type", "tumour cell", "tumor cell", "neoplastic", "interaction", "microenvironment", "opc"],
    "correlation": ["correlat", "co-express", "coexpress", "network"],
    "spatial": ["spatial", "visium", "co-locali", "colocali", "infiltrat", "interface", "tumour edge", "tumor edge", "margin"],
}
GENE_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,9}(?:-[A-Z0-9]+)?)\b")
NOT_GENES = {"GBM", "TCGA", "CGGA", "RNA", "DNA", "IDH", "MGMT", "AND", "OR", "NOT", "WHO", "OPC", "UMAP", "PCA", "HR", "CI", "KM", "CNS", "MRI", "TMZ", "AMPA", "NMDA", "GEO", "ML", "AI", "USA", "II", "III", "IV", "SD", "BH", "FDR", "CIMP", "G-CIMP", "PH", "SC", "MCP", "OS", "PFS", "EGFR-AMP"}


def _known_genes() -> set[str]:
    try:
        from neurobridge.data.loaders import load_cohort

        return set(load_cohort("tcga_gbm_rnaseq").expression.index)
    except Exception:  # noqa: BLE001
        return set().union(*[set(p.genes) for p in PANELS.values()])


class HypothesisDraft(BaseModel):
    hypothesis: str
    null: str
    prediction: str


class PlanDraft(BaseModel):
    restated_question: str
    sub_questions: list[str] = Field(description="3-6 concrete, testable sub-questions")
    hypotheses: list[HypothesisDraft] = Field(description="testable hypotheses with explicit null hypotheses")
    literature_queries: list[str] = Field(description="2-4 Europe PMC boolean queries")
    workflows: list[str] = Field(description=f"subset of {list(WORKFLOWS)}")
    extra_genes: list[str] = Field(default_factory=list, description="HGNC symbols explicitly relevant to the question")


def deterministic_plan(question: str, workflows_override: list[str] | None = None) -> ResearchPlan:
    q = question.lower()
    found = {w for w, kws in KEYWORDS.items() if any(k in q for k in kws)}
    if not found:
        found = {"subtype", "survival", "differential", "single_cell"}
    if "survival" in found or "subtype" in found:
        # A bulk association with neuronal genes is uninterpretable without (i) a composition control and
        # (ii) knowing which cells express the genes, so both are added automatically.
        found |= {"composition", "single_cell", "spatial"}
    found |= {"literature"}
    if workflows_override:
        found = set(workflows_override) | {"literature"}
    genes_mentioned = [g for g in GENE_RE.findall(question) if g not in NOT_GENES]
    known = _known_genes()
    valid = [g for g in genes_mentioned if g in known]
    panel_ids = ["neuronal_activity_candidates", "neuron_glioma_extended", "normal_neuron_markers", "kegg_glutamatergic_synapse"]
    notes = []
    candidate = list(CANDIDATE_PANEL.genes)
    gene_level: list[str] = []
    if valid and set(valid) != set(candidate):
        register_custom_panel(valid)
        panel_ids.insert(0, "user_genes")
        notes.append(f"Genes named in the question are analysed as their own panel ('user_genes'): {', '.join(valid)}")
        if len(valid) <= 4:
            gene_level = list(dict.fromkeys(valid))
            for g in gene_level:
                register_custom_panel([g], panel_id=f"gene_{g}", name=f"{g} (single gene)")
            notes.append(f"Each named gene is also tested individually: {', '.join(gene_level)}")
    restrictions = []
    if re.search(r"idh[\s-]*(wild[\s-]*type|wt)\b", q):
        restrictions.append("IDH-wildtype")
        notes.append("Restriction requested: survival and subtype analyses use IDH-wildtype tumours (non-G-CIMP as the proxy in TCGA).")
    rejected = sorted(set(genes_mentioned) - set(valid))
    if rejected:
        notes.append(f"Tokens not recognised as measured gene symbols were ignored: {', '.join(rejected)}")
    hyps = []
    if "subtype" in found:
        hyps.append({"hypothesis": "The neuronal-activity candidate score differs between glioblastoma transcriptional subtypes.", "null": "Score distributions are identical across subtypes.", "prediction": "Higher scores in Proneural than Mesenchymal tumours (neural-progenitor/OPC-like vs injury/inflammatory programmes), replicated in independent cohorts with projected labels."})
    if "survival" in found:
        hyps.append({"hypothesis": "The candidate score is associated with overall survival after adjustment for age, sex and IDH/G-CIMP status.", "null": "Adjusted HR per SD = 1.", "prediction": "Krishna et al. 2023 link functional neuron-glioma connectivity to shorter survival, predicting HR > 1; replication required in independent cohorts."})
    if "single_cell" in found:
        hyps.append({"hypothesis": "Candidate genes are expressed by neoplastic cells, not only by non-malignant neural cells.", "null": "Candidate-gene expression in neoplastic cells is not higher than in other cell types.", "prediction": "Postsynaptic receptor genes (GRIA1/2, GRIN2B) detectable in neoplastic cells; presynaptic SLC17A7 restricted to neurons."})
    if "spatial" in found:
        hyps.append({"hypothesis": "In tissue, candidate genes are expressed beyond neuron-rich regions and neuron-glioma ligand-receptor pairs co-localise in neighbouring spots.", "null": "Candidate expression tracks neuronal content only; ligand-receptor neighbour co-expression is no higher than expected from shared tissue domains.", "prediction": "Postsynaptic receptors (GRIA1/2) spread through tumour regions; presynaptic SLC17A7 confined to neuron-rich spots; BDNF-NTRK2 and VGLUT1-glutamate receptor pairs co-localise. Single patient: descriptive only."})
    if "composition" in found:
        hyps.append({"hypothesis": "Bulk associations are not explained by normal-brain (neuronal) content of the sample.", "null": "Associations disappear after adjusting for mature-neuron marker expression.", "prediction": "Effects attenuated but persistent after neuron-marker adjustment."})
    subq = [
        "Which neuronal-activity and synaptic genes are candidates, and what is the prior evidence for each?",
        "Which public cohorts provide bulk expression with subtypes, IDH/G-CIMP status and survival, and which provide single-cell resolution?",
    ]
    if "subtype" in found:
        subq.append("Does a candidate-gene score differ across Verhaak subtypes, and does the difference replicate in independent cohorts?")
    if "survival" in found:
        subq.append("Is the score associated with overall survival after adjustment for known prognostic factors, and does that replicate?")
    if "spatial" in found:
        subq.append("In intact tissue, do candidate genes follow neuron-rich regions, and do neuron-glioma ligand-receptor pairs co-localise?")
    if "single_cell" in found:
        subq.append("Which cell types express the candidate genes, and are they expressed by neoplastic cells?")
    if "differential" in found:
        subq.append("Which pathways distinguish score-high from score-low tumours?")
    subq.append("How much of the bulk signal could be explained by tumour purity / infiltrated normal brain?")
    return ResearchPlan(
        question=question,
        restated_question=question.strip().rstrip("?") + "?",
        sub_questions=subq,
        hypotheses=hyps,
        candidate_genes=valid if valid and "user_genes" in panel_ids else candidate,
        gene_set_ids=panel_ids,
        pathways=["KEGG Glutamatergic synapse", "GO Chemical synaptic transmission", "GO Regulation of neuronal synaptic plasticity"],
        outcome_variables=["overall survival"] if "survival" in found else [],
        molecular_groupings=["Verhaak 2010 transcriptional subtype", "IDH / G-CIMP status"] if "subtype" in found else [],
        requested_workflows=[w for w in WORKFLOWS if w in found],
        datasets_preferred=["tcga_gbm_rnaseq", "cgga_325", "cgga_693"] + (["gse84465_darmanis"] if "single_cell" in found else []) + (["visium_gbm_10x"] if "spatial" in found else []),
        confounders=["age", "sex", "IDH mutation / G-CIMP status", "MGMT promoter methylation", "tumour purity / normal-brain content", "sequencing platform and cohort"],
        evidence_ceiling="statistical association (observational transcriptomics; no perturbation or longitudinal data, so mechanistic or causal claims cannot be supported by these analyses)",
        planner="deterministic",
        notes=notes,
        restrictions=restrictions,
        gene_level=gene_level,
    )


def literature_queries(plan: ResearchPlan) -> list[str]:
    qs = ['glioblastoma AND ("neuronal activity" OR "neuron-glioma" OR "neural circuit")']
    genes = [g for g in plan.candidate_genes if g in set(CANDIDATE_PANEL.genes) | set(EXTENDED_PANEL.genes)][:6] or plan.candidate_genes[:6]
    if genes:
        qs.append("glioma AND (" + " OR ".join(genes) + ")")
    if "survival" in plan.requested_workflows:
        qs.append('glioblastoma AND (synaptic OR "neuronal signature") AND survival')
    if "subtype" in plan.requested_workflows:
        qs.append('glioblastoma AND "transcriptional subtype" AND (neural OR proneural)')
    return qs


def make_plan(question: str, use_llm: bool = True, workflows_override: list[str] | None = None) -> tuple[ResearchPlan, list[str]]:
    plan = deterministic_plan(question, workflows_override)
    queries = literature_queries(plan)
    if not use_llm or not llm.provider_status()["available"]:
        return plan, queries
    try:
        draft = llm.structured(
            system="You are the Hypothesis Agent of a biomedical research platform. Decompose the question into testable sub-questions and hypotheses with explicit null hypotheses. Do not state results; do not cite papers.",
            prompt=f"Question: {question}\nAvailable workflows: {WORKFLOWS}\nAvailable data: TCGA-GBM RNA-seq with Verhaak subtypes and survival; CGGA-325 and CGGA-693 RNA-seq with IDH/MGMT and survival; GSE84465 single-cell RNA-seq (4 patients); one 10x Visium GBM section (1 patient).",
            schema=PlanDraft,
        )
    except llm.LLMUnavailable as exc:
        plan.notes.append(f"LLM-assisted decomposition unavailable ({exc}); deterministic plan used.")
        return plan, queries
    known = _known_genes()
    extra = [g.upper() for g in draft.extra_genes if g.upper() in known]
    wf = [w for w in draft.workflows if w in WORKFLOWS]
    if not workflows_override and wf:
        plan.requested_workflows = sorted(set(wf) | {"literature", "composition"}, key=list(WORKFLOWS).index)
    plan.restated_question = draft.restated_question
    plan.sub_questions = draft.sub_questions or plan.sub_questions
    plan.hypotheses = [h.model_dump() for h in draft.hypotheses] or plan.hypotheses
    if extra:
        plan.notes.append(f"LLM suggested additional genes (validated against the expression matrix): {', '.join(extra)}")
    plan.planner = "LLM-assisted (validated)"
    queries = (draft.literature_queries[:4] or []) + queries[:2]
    return plan, queries
