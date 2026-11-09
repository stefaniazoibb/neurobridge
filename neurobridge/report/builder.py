"""Build the evidence report from an executed investigation. Every claim is tagged with its evidence level and
type; numbers come only from executed analyses; citations only from verified Europe PMC records."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from neurobridge.data.catalog import get_dataset
from neurobridge.schemas.agents import InvestigationResult

LEVEL_TAG = {
    "biological plausibility": "PLAUSIBILITY",
    "statistical association": "ASSOCIATION",
    "mechanistic evidence": "MECHANISTIC",
    "causal evidence": "CAUSAL",
    "insufficient evidence": "INSUFFICIENT",
}

REPORT_FIGURES = [
    ("groups_neuronal_activity_candidates_tcga_gbm_rnaseq", "Candidate score by transcriptional subtype (TCGA)"),
    ("validation_subtype_neuronal_activity_candidates", "Replication of the subtype difference"),
    ("composition_tcga_gbm_rnaseq", "Composition control"),
    ("km_neuronal_activity_candidates_tcga_gbm_rnaseq", "Survival in the discovery cohort"),
    ("validation_survival_neuronal_activity_candidates", "Replication of the survival association"),
    ("sensitivity_neuronal_activity_candidates", "Sensitivity analyses"),
    ("dotplot_celltype", "Cell-type expression"),
    ("umap_celltype", "Single-cell landscape"),
    ("spatial_domains", "Spatial domains (Visium)"),
    ("spatial_candidate", "Candidate score in tissue"),
    ("spatial_interface_gene", "Gradient from neuron-rich tissue"),
    ("spatial_lr", "Ligand-receptor co-localisation"),
    ("volcano_neuronal_activity_candidates_tcga_gbm_rnaseq", "Differential expression"),
    ("enrichment_gsea_MSigDB_Hallmark_2020_tcga_gbm_rnaseq", "Hallmark GSEA"),
]


def _md_table(df: pd.DataFrame, floatfmt: str = ".3g", max_rows: int = 30) -> str:
    df = df.head(max_rows).copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    return df.to_markdown(index=False)


def _cite(c) -> str:
    authors = (c.authors or "").split(",")
    a = authors[0].strip() + (" et al." if len(authors) > 1 else "")
    return f"{a} ({c.year}). {c.title}. *{c.journal}*. PMID [{c.pmid}](https://pubmed.ncbi.nlm.nih.gov/{c.pmid}/)" + (f", doi:{c.doi}" if c.doi else "")


def _cite_inline(pmids: list[str], index: dict[str, int]) -> str:
    nums = sorted(index[p] for p in pmids if p in index)
    return f" [{', '.join(map(str, nums))}]" if nums else ""


def build_markdown(result: InvestigationResult, ws, figure_paths: Optional[dict[str, str]] = None) -> str:
    plan = result.plan
    claims = [c for c in result.claims if c.get("status") != "removed"]
    review = result.review or {}
    cited = []
    for c in claims:
        for p in c["citations"]:
            if p not in cited and p in ws.citations:
                cited.append(p)
    for p in ws.citations:
        if p not in cited and ws.citations[p].relevance and not ws.citations[p].relevance.startswith("Retrieved by query"):
            cited.append(p)
    idx = {p: i + 1 for i, p in enumerate(cited)}
    L: list[str] = []
    add = L.append
    add("# NeuroBridge evidence report")
    add("")
    add(f"**Investigation** `{result.investigation_id}`  ")
    add(f"**Generated** {result.log.finished_at or result.log.started_at} (UTC)  ")
    add(f"**Status** {result.status}  ")
    add("**Scope** Research exploration only. Not for diagnostic or treatment decisions.")
    add("")
    add("> Evidence tags: **[ASSOCIATION]** statistical association measured here; **[MECHANISTIC]** experimental mechanism reported in the cited literature; **[PLAUSIBILITY]** biological plausibility or interpretation; **[INSUFFICIENT]** the data do not support the statement. Claim types: *exploratory* (discovery cohort), *confirmatory* (pre-specified test in independent cohorts), *sensitivity*, *literature*, *interpretation*.")
    add("")

    # 1 executive summary
    add("## 1. Executive summary")
    add("")
    key_ids = ["S1", "S2", "S3", "C1", "O2", "SC1", "SP1", "SP3", "I1"]
    key_ids = key_ids[:5] + [c["id"] for c in claims if c["id"].startswith("OG_")] + key_ids[5:]
    for k in key_ids:
        c = next((x for x in claims if x["id"] == k), None)
        if c:
            add(f"- **[{LEVEL_TAG[c['level']]}]** *{c['kind']}* ({c['status']}). {c['text']}{_cite_inline(c['citations'], idx)}")
    add("")
    add(f"**Highest evidence level reached by a confirmatory analysis:** {review.get('highest_evidence_level_reached', 'n/a')}. **Design ceiling:** {plan.evidence_ceiling}.")
    add("")

    # 2 question
    add("## 2. Research question")
    add("")
    add(f"> {result.question}")
    add("")
    add(f"Planner: {plan.planner}. Workflows selected: {', '.join(plan.requested_workflows)}.")
    add("")
    add("**Sub-questions**")
    for s in plan.sub_questions:
        add(f"1. {s}")
    add("")
    add("**Pre-specified hypotheses**")
    add("")
    add(_md_table(pd.DataFrame(plan.hypotheses)))
    add("")

    # 3 rationale
    add("## 3. Biological rationale")
    add("")
    for c in claims:
        if c["kind"] == "literature":
            add(f"- **[{LEVEL_TAG[c['level']]}]** {c['text']}{_cite_inline(c['citations'], idx)}")
    add("")
    from neurobridge.knowledge.genesets import CANDIDATE_PANEL, EXTENDED_PANEL, NEURON_CONTROL_PANEL

    add("**Candidate genes (investigation targets, not assumed biomarkers)**")
    add("")
    add(_md_table(pd.DataFrame([{"gene": g, "rationale": r} for g, r in {**CANDIDATE_PANEL.gene_rationale, **EXTENDED_PANEL.gene_rationale}.items()])))
    add("")
    add(f"Panels: *{CANDIDATE_PANEL.name}* ({', '.join(CANDIDATE_PANEL.genes)}; origin: {CANDIDATE_PANEL.origin}); *{EXTENDED_PANEL.name}* ({len(EXTENDED_PANEL.genes)} genes); negative-control panel *{NEURON_CONTROL_PANEL.name}* ({', '.join(NEURON_CONTROL_PANEL.genes)}). {NEURON_CONTROL_PANEL.description}")
    add("")

    # 4 datasets
    add("## 4. Datasets")
    add("")
    rows = []
    for d in result.datasets_used:
        r = get_dataset(d)
        n = result.artefacts.get("dataset_summaries", {}).get(d)
        rows.append({"dataset": d, "source": r.source, "publication (PMID)": r.publication, "modalities": ", ".join(m.value for m in r.modalities), "n analysed": (f"{n['n_samples']} samples ({n['n_primary']} primary)" if n else f"{r.n_cells} cells"), "role": r.role})
    add(_md_table(pd.DataFrame(rows)))
    add("")
    for d in result.datasets_used:
        r = get_dataset(d)
        add(f"- **{d}** limitations: " + "; ".join(r.limitations))
    if result.datasets_unavailable:
        add("")
        add("**Unavailable datasets and consequences**")
        for u in result.datasets_unavailable:
            add(f"- {u['dataset_id']}: {u['consequence']}")
    add("")

    # 5 methods
    add("## 5. Methods")
    add("")
    add("- **Expression scale.** Bulk: log2(RSEM + 1) (TCGA as distributed by UCSC Xena; CGGA RSEM log-transformed by NeuroBridge). Primary tumours only for all confirmatory tests.")
    add("- **Signature score.** Per-gene z-score across primary tumours within each cohort, averaged over the panel's measured genes (rank-based alternative in sensitivity analyses).")
    add("- **Subtype comparison.** Kruskal-Wallis with epsilon-squared; pairwise Mann-Whitney with Benjamini-Hochberg correction; Hedges' g with 95% CI and Cliff's delta.")
    add("- **Subtype transfer.** L2-regularised multinomial logistic regression trained on TCGA primary tumours (Proneural, Classical, Mesenchymal; Neural excluded), 1,500 most variable shared genes after excluding all candidate, extended-panel and neuron-marker genes, within-cohort gene z-scoring, 5-fold stratified CV. Applied unchanged to CGGA (no refitting, no information from validation cohorts).")
    add("- **Differential expression.** Welch t-test per expressed gene (mean log2 >= 1, detected in >= 20% of samples) between median-split score groups; BH FDR; Hedges' g.")
    add("- **Enrichment.** Hypergeometric over-representation with the measured genes as background (GO Biological Process 2023) and pre-ranked GSEA on Hedges' g (MSigDB Hallmark 2020, gene-set permutations) via GSEApy; libraries retrieved from Enrichr and cached with a timestamp.")
    add("- **Survival.** Overall survival in months. Kaplan-Meier with Greenwood CIs and log-rank test; Cox proportional hazards (lifelines, Efron ties) with the score standardised (HR per SD); covariates age, sex and the IDH proxy available in each cohort (G-CIMP in TCGA; IDH and MGMT in CGGA); complete-case analysis; PH assumption tested on scaled Schoenfeld residuals.")
    add("- **Validation rule.** A hypothesis is called replicated in a cohort only if the estimate has the same sign as in discovery and p < 0.05 in that independent cohort; random-effects (DerSimonian-Laird) pooling is reported descriptively with I^2.")
    add("- **Composition control.** Mean z-score of eight mature-neuron markers as a proxy for infiltrated normal brain; used as a covariate and for residualisation.")
    add("- **Spatial.** 10x Visium GBM section: spots with >= 500 UMIs and >= 250 genes; normalise_total, log1p, HVG, PCA, Leiden domains; programme scores (neuron, tumour/reactive-glial, myeloid, vascular). Moran's I on the hexagonal neighbour graph (row-standardised, 999 permutations). Neuron-rich spots = top 20% of the neuron programme; among other spots, Spearman of expression with ring distance, also partialled on the spot's own neuron score; interface (1-2 rings) vs distal (>= 5 rings) Hedges' g. Ligand-receptor co-localisation = mean of ligand(spot) x mean receptor(neighbours), tested against global and domain-stratified ligand permutations, BH. One patient: descriptive.")
    add("- **Single cell.** Scanpy: cells with >= 500 genes and < 20% mitochondrial reads; normalise_total (1e4), log1p, 2,000 HVGs (patient-aware), PCA (30), kNN (k = 15), UMAP, Leiden; author cell-type annotations; Wilcoxon rank-sum with BH.")
    add("")

    # 6 results / 7 statistical evidence
    add("## 6. Results")
    add("")
    for c in claims:
        if c["kind"] in ("exploratory",):
            add(f"- **{c['id']} [{LEVEL_TAG[c['level']]}]** ({c['status']}) {c['text']}{_cite_inline(c['citations'], idx)}" + (f" *Analyses: {', '.join(c['analyses'])}.*" if c["analyses"] else ""))
            for cv in c["caveats"]:
                add(f"    - caveat: {cv}")
    add("")
    for fid, name in REPORT_FIGURES:
        if fid in ws.figures:
            cap = ws.figures[fid].layout.meta.get("caption", "") if ws.figures[fid].layout.meta else ""
            if figure_paths and fid in figure_paths:
                add(f"![{name}]({figure_paths[fid]})")
            add(f"*Figure: {name}.* {cap}")
            add("")

    add("## 7. Statistical evidence")
    add("")
    for tid in ["pairwise_neuronal_activity_candidates_tcga_gbm_rnaseq", "composition_confounding", "sensitivity_neuronal_activity_candidates", "sc_neoplastic_vs_rest", "sc_score_by_celltype", "spatial_correlations", "spatial_interface", "spatial_ligand_receptor"]:
        if tid in ws.tables:
            add(f"**{tid.replace('_', ' ')}**")
            add("")
            df = ws.tables[tid]
            df = df[[c for c in df.columns if c not in ("residual_medians", "label_source")]]
            add(_md_table(df))
            add("")

    # 8 validation
    add("## 8. Cross-dataset validation")
    add("")
    for c in claims:
        if c["kind"] in ("confirmatory", "sensitivity"):
            add(f"- **{c['id']} [{LEVEL_TAG[c['level']]}]** *{c['kind']}* ({c['status']}). {c['text']}")
            for cv in c["caveats"]:
                add(f"    - caveat: {cv}")
    add("")
    for tid in [t for t in ws.tables if t.startswith("validation_")] + (["subtype_classifier_cv_confusion"] if "subtype_classifier_cv_confusion" in ws.tables else []):
        add(f"**{tid.replace('_', ' ')}**")
        add("")
        df = ws.tables[tid].copy()
        if "notes" in df:
            df["notes"] = df["notes"].map(lambda v: "; ".join(v[:2]) if isinstance(v, list) else v)
        add(_md_table(df.reset_index() if tid.startswith("subtype") else df))
        add("")

    # 9 interpretation
    add("## 9. Biological interpretation")
    add("")
    for c in claims:
        if c["kind"] == "interpretation":
            add(f"- **[{LEVEL_TAG[c['level']]}]** {c['text']}{_cite_inline(c['citations'], idx)}")
    add("- **[PLAUSIBILITY]** Interpretation: a replicated association with transcript abundance does not show that neuron-to-glioma synapses exist or function in these tumours; electrophysiology, synapse imaging or spatial co-localisation would be needed.")
    add("")

    # 10 alternatives
    add("## 10. Alternative explanations")
    add("")
    for a in [
        "**Sample composition.** Bulk neuronal transcripts can come from infiltrated or adjacent normal cortex; Proneural and Neural samples may contain more normal brain. The neuron-marker control addresses this only partly.",
        "**Tumour-cell state.** OPC-like and NPC-like neoplastic states (Neftel et al. 2019) express synaptic genes intrinsically, which would also raise the score in Proneural tumours.",
        "**Spot resolution.** Visium spots mix several cells, so gradients near neuron-rich tissue can reflect residual neuronal content even after adjustment.",
        "**Sampling site.** Resections at the infiltrative margin contain more neurons than necrotic core samples; anatomical sampling is not recorded in these cohorts.",
        "**Subtype misclassification.** Projected labels in CGGA can bias contrasts.",
        "**Platform and population.** TCGA (US, HiSeq 2000) and CGGA (China, HiSeq 2000/2500 and 4000) differ in platform, ancestry and treatment era.",
        "**Survival confounding.** Unmeasured factors (extent of resection, performance status, second-line therapy) are not available for adjustment.",
    ]:
        add(f"- {a}")
    add("")

    # 11 limitations
    add("## 11. Limitations")
    add("")
    for i in review.get("issues", []):
        add(f"- *{i['severity']}* ({i['claim']}): {i['issue']}" + (f". Action: {i['action']}" if i.get("action") else ""))
    lims = []
    for m in result.log.messages:
        for l in m.limitations:
            if l not in lims:
                lims.append(l)
    add("")
    add("<details><summary>All limitations reported by agents and tools</summary>")
    add("")
    for l in lims:
        add(f"- {l}")
    add("")
    add("</details>")
    add("")
    add("**What remains uncertain**")
    for u in review.get("remains_uncertain", []):
        add(f"- {u}")
    if review.get("narrative"):
        add("")
        add(f"**Reviewer narrative** ({review['narrative_source']})")
        add("")
        add(review["narrative"]["overall_assessment"])
    add("")

    # 12 follow-up
    add("## 12. Proposed follow-up experiments")
    add("")
    for f in [
        "Deconvolve bulk cohorts with a GBM single-cell reference (e.g. Neftel 2019 states) and test whether the score is carried by malignant OPC/NPC-like fractions or by normal neurons.",
        "Extend the spatial analysis from one section to many patients (e.g. Ravi et al. 2022 Visium cohort) so that co-localisation and interface gradients can be tested across patients rather than described in one case; add single-cell-resolution spatial data (Xenium / MERFISH) to resolve partial-volume effects.",
        "Ligand-receptor inference (e.g. CellChat / LIANA) on a larger 10x GBM atlas with more captured neurons.",
        "Test the survival hypothesis with an exposure closer to the biology (e.g. a tumour-intrinsic synaptic programme, or MRI-based functional connectivity as in Krishna et al. 2023).",
        "Experimental follow-up in patient-derived models: NLGN3 or AMPA-receptor perturbation in co-culture with neurons, stratified by transcriptional state.",
    ]:
        add(f"- {f}")
    add("")

    # 13 references
    add("## 13. References")
    add("")
    add("All references were retrieved from Europe PMC during this investigation (verified records; titles and metadata are not typed by hand).")
    add("")
    for p in cited:
        c = ws.citations[p]
        add(f"{idx[p]}. {_cite(c)}")
    add("")

    # reproducibility
    add("## Appendix: reproducibility")
    add("")
    from neurobridge.data.cache import read_manifest
    from neurobridge.provenance import software_versions
    from neurobridge.config import get_settings

    man = read_manifest()
    add(f"Random seed: {get_settings().seed}. Tool calls: {len(result.log.tool_calls)} ({sum(1 for t in result.log.tool_calls if t.status.value == 'failed')} failed).")
    add("")
    add(_md_table(pd.DataFrame([{"file": k, "source": v["source"], "sha256": v["sha256"][:16] + "...", "retrieved_at": v["retrieved_at"]} for k, v in man.items()])))
    add("")
    add(_md_table(pd.DataFrame([{"package": k, "version": v} for k, v in software_versions().items()])))
    add("")
    return "\n".join(L)
