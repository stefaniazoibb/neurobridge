"""Built-in NeuroBridge tools. Each tool is a thin, typed wrapper around a function in `neurobridge.analysis`,
`neurobridge.data` or `neurobridge.literature`; it stores results in the workspace and reports limitations."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from neurobridge.agents.tools.registry import ToolOutput, TransientToolError, register
from neurobridge.agents.workspace import Workspace
from neurobridge.analysis import singlecell as scm
from neurobridge.analysis.correlation import gene_correlation
from neurobridge.analysis.differential import differential_expression
from neurobridge.analysis.enrichment import over_representation, prerank_gsea
from neurobridge.analysis.groups import compare_groups
from neurobridge.analysis.sensitivity import composition_check, residualise, survival_sensitivity
from neurobridge.analysis.signatures import dichotomise, score_signature
from neurobridge.analysis.subtyping import train_and_project
from neurobridge.analysis.survival import cox_model, exposure_row, kaplan_meier
from neurobridge.analysis.validation import validate_group_difference, validate_signature_survival
from neurobridge.data.catalog import INGESTIBLE, build_catalog, get_dataset
from neurobridge.data.loaders import load_cohort, load_single_cell
from neurobridge.knowledge.genesets import all_panels
from neurobridge.schemas.analysis import AnalysisKind
from neurobridge.viz import figures as F

SUBTYPE_ORDER = ["Proneural", "Neural", "Classical", "Mesenchymal"]


def _panel(panel_id: str):
    panels = all_panels()
    if panel_id not in panels:
        raise KeyError(f"unknown gene panel {panel_id!r}; available: {sorted(panels)}")
    return panels[panel_id]


def _cohort(ws: Workspace, dataset_id: str):
    if dataset_id not in ws.cohorts:
        raise KeyError(f"cohort {dataset_id} not loaded in this investigation (run load_expression_data first)")
    return ws.cohorts[dataset_id]


def _subtype_labels(cohort) -> tuple[pd.Series, str]:
    """Original labels if present, otherwise NeuroBridge projections (if computed)."""
    c = cohort.clinical
    if "subtype" in c and c["subtype"].notna().sum() > 0:
        return c["subtype"], "original Verhaak 2010 labels (TCGA clinical matrix)"
    if "subtype_projected" in c and c["subtype_projected"].notna().sum() > 0:
        return c["subtype_projected"], "projected by NeuroBridge classifier trained on TCGA"
    return pd.Series(np.nan, index=c.index), "none"


def _idh_wt_mask(cohort) -> tuple[pd.Series, str]:
    c = cohort.clinical
    if "idh_status" in c and c["idh_status"].notna().sum() > 0:
        return c["idh_status"].eq("Wildtype"), "IDH-wildtype (clinical annotation)"
    if "g_cimp" in c and c["g_cimp"].notna().sum() > 0:
        return c["g_cimp"].eq("non-G-CIMP"), "non-G-CIMP (proxy for IDH-wildtype)"
    return pd.Series(True, index=c.index), "no IDH information"


# ---------------------------------------------------------------- literature
class LiteratureInput(BaseModel):
    queries: list[str] = Field(..., description="Europe PMC queries (boolean syntax allowed)")
    page_size: int = Field(8, ge=1, le=50)
    include_key_references: bool = True


@register("search_literature", "Search Europe PMC (MEDLINE) for supporting literature and fetch the curated key references by PMID. Citations are verified records from the live API, cached locally.", LiteratureInput, "literature", retryable=True)
def search_literature(ws: Workspace, inp: LiteratureInput) -> ToolOutput:
    from neurobridge.literature.europepmc import LiteratureUnavailable, fetch_pmids, search
    from neurobridge.literature.key_references import KEY_REFERENCES

    results, lims, n_ok = {}, [], 0
    try:
        if inp.include_key_references:
            rel = {r["pmid"]: r for r in KEY_REFERENCES}
            for c in fetch_pmids(list(rel)):
                c.relevance = rel[c.pmid]["relevance"]
                ws.citations[c.pmid] = c
            results["key_references"] = {"n": len(rel), "retrieved": sum(1 for p in rel if p in ws.citations)}
        for q in inp.queries:
            cites, meta = search(q, page_size=inp.page_size)
            for c in cites:
                if c.pmid and c.pmid not in ws.citations:
                    c.relevance = f"Retrieved by query: {q}"
                    ws.citations[c.pmid] = c
            results[q] = {"hit_count": meta["hit_count"], "pmids": [c.pmid for c in cites], "retrieved_at": meta["retrieved_at"]}
            n_ok += 1
    except LiteratureUnavailable as exc:
        raise TransientToolError(str(exc)) from exc
    lims.append("Search results are ranked by Europe PMC relevance; they are evidence leads, not a systematic review. Abstract-level relevance has not been screened by a human.")
    ref = ws.put("literature", results)
    return ToolOutput(summary=f"{len(ws.citations)} verified records ({results.get('key_references', {}).get('retrieved', 0)} curated key references; {n_ok} live queries)", refs=[ref], limitations=lims, data=results)


# ---------------------------------------------------------------- datasets
class DiscoverInput(BaseModel):
    modalities: list[str] = Field(default_factory=lambda: ["bulk RNA-seq", "single-cell RNA-seq", "survival"])
    require_survival: bool = False
    live_listing: bool = True


@register("discover_datasets", "List datasets relevant to glioblastoma from the NeuroBridge catalogue, and (optionally) live listings from cBioPortal and CELLxGENE.", DiscoverInput, "data", retryable=True)
def discover_datasets(ws: Workspace, inp: DiscoverInput) -> ToolOutput:
    cat = build_catalog()
    sel = [d for d in cat if any(m.value in inp.modalities or m.name.lower() in [x.lower() for x in inp.modalities] for m in d.modalities) or not inp.modalities]
    if inp.require_survival:
        sel = [d for d in sel if d.survival_available]
    live, lims, degraded = {}, [], False
    if inp.live_listing:
        try:
            from neurobridge.data import cbioportal

            live["cbioportal_gbm_studies"] = [{"studyId": s["studyId"], "name": s["name"], "samples": s.get("allSampleCount")} for s in cbioportal.search_studies("glioblastoma")]
        except Exception as exc:  # noqa: BLE001
            degraded = True
            lims.append(f"cBioPortal live listing failed: {exc}")
        try:
            from neurobridge.data import cellxgene

            live["cellxgene_glioma_collections"] = cellxgene.list_glioma_collections(timeout=40)
        except Exception as exc:  # noqa: BLE001
            degraded = True
            lims.append(f"CELLxGENE live listing failed: {exc}")
    ref = ws.put("datasets", {"catalogue": [d.model_dump(mode="json") for d in sel], "live": live})
    ing = [d.dataset_id for d in sel if d.dataset_id in INGESTIBLE]
    return ToolOutput(summary=f"{len(sel)} catalogue datasets match ({len(ing)} ingestible: {', '.join(ing)}); live listings: {', '.join(f'{k}={len(v)}' for k, v in live.items()) or 'none'}", refs=[ref], limitations=lims, degraded=degraded, data={"ingestible": ing, "selected": [d.dataset_id for d in sel]})


class LoadInput(BaseModel):
    dataset_id: str


@register("load_expression_data", "Download (if needed) and load a bulk expression cohort or single-cell dataset with harmonised clinical variables and file provenance (SHA-256).", LoadInput, "data", retryable=True)
def load_expression_data(ws: Workspace, inp: LoadInput) -> ToolOutput:
    rec = get_dataset(inp.dataset_id)
    if inp.dataset_id == "visium_gbm_10x":
        from neurobridge.data.loaders import load_spatial

        a = load_spatial(inp.dataset_id)
        ws.single_cell[inp.dataset_id + ":raw"] = a
        return ToolOutput(summary=f"{a.n_obs} tissue spots x {a.n_vars} genes (one section, one patient)", limitations=list(a.uns.get("notes", [])) + rec.limitations, data={"n_spots": int(a.n_obs)})
    if inp.dataset_id == "gse84465_darmanis":
        a = load_single_cell(inp.dataset_id)
        ws.single_cell[inp.dataset_id + ":raw"] = a
        return ToolOutput(summary=f"{a.n_obs} cells x {a.n_vars} genes; {a.obs['patient'].nunique()} patients; author cell types: {a.obs['cell_type_author'].value_counts().to_dict()}", limitations=list(a.uns.get("notes", [])) + rec.limitations, data={"n_cells": int(a.n_obs), "n_genes": int(a.n_vars)})
    c = load_cohort(inp.dataset_id)
    ws.cohorts[inp.dataset_id] = c
    s = c.summary()
    return ToolOutput(summary=f"{s['n_samples']} samples ({s['n_primary']} primary) x {s['n_genes']} genes; {s['n_with_survival']} with survival ({s['n_events']} deaths)", limitations=c.notes + rec.limitations, data=s)


class InspectInput(BaseModel):
    dataset_id: str


@register("inspect_clinical_metadata", "Report clinical-variable completeness, group sizes and potential confounders for a loaded cohort.", InspectInput, "data")
def inspect_clinical_metadata(ws: Workspace, inp: InspectInput) -> ToolOutput:
    c = _cohort(ws, inp.dataset_id)
    clin = c.clinical
    prim = clin[clin["sample_type"] == "Primary"]
    rows = []
    for col in ["os_months", "os_event", "age", "sex", "subtype", "idh_status", "g_cimp", "mgmt_status", "radiotherapy", "chemotherapy"]:
        if col not in clin:
            continue
        nn = int(prim[col].notna().sum())
        vc = prim[col].value_counts().head(6).to_dict() if prim[col].dtype == object else {}
        rows.append({"variable": col, "available_primary": nn, "missing_primary": int(len(prim) - nn), "pct_complete": round(100 * nn / max(1, len(prim)), 1), "levels": ", ".join(f"{k}: {v}" for k, v in vc.items())})
    df = pd.DataFrame(rows)
    tid = ws.put_table(f"clinical_completeness_{inp.dataset_id}", df)
    lims = []
    for r in rows:
        if r["pct_complete"] == 0:
            lims.append(f"{r['variable']} is not available in {inp.dataset_id}; analyses cannot adjust for it here")
        elif r["pct_complete"] < 80:
            lims.append(f"{r['variable']} is {r['pct_complete']}% complete in primary tumours of {inp.dataset_id}; complete-case analyses will lose samples")
    if len(prim) < 100:
        lims.append(f"only {len(prim)} primary tumours: limited power for multivariable models")
    lims.append("Treatment (radiotherapy / temozolomide) is post-diagnosis and partly determined by prognosis (performance status); it is reported but not used as an adjustment covariate.")
    return ToolOutput(summary=f"{len(prim)} primary tumours; completeness: " + ", ".join(f"{r['variable']} {r['pct_complete']}%" for r in rows), tables=[tid], limitations=lims, data={"n_primary": int(len(prim)), "completeness": {r["variable"]: r["pct_complete"] for r in rows}})


# ---------------------------------------------------------------- molecular
class ScoreInput(BaseModel):
    dataset_id: str
    panel_id: str = "neuronal_activity_candidates"
    method: Literal["mean_z", "rank"] = "mean_z"


@register("score_signature", "Score a gene panel per sample (mean within-cohort z-score of log2 expression) in primary tumours.", ScoreInput, "bioinformatics")
def score_signature_tool(ws: Workspace, inp: ScoreInput) -> ToolOutput:
    c = _cohort(ws, inp.dataset_id).primary_tumours()
    p = _panel(inp.panel_id)
    res = score_signature(c, p.genes, inp.panel_id, method=inp.method)
    ref = ws.put("signature", res)
    ws.cohorts[inp.dataset_id].clinical.loc[list(res.scores), f"score__{inp.panel_id}"] = pd.Series(res.scores)
    return ToolOutput(summary=f"{inp.panel_id}: {len(res.genes_found)}/{len(res.genes_requested)} genes found, {len(res.scores)} primary tumours scored", refs=[ref], limitations=res.record.limitations + res.record.warnings, data={"genes_found": res.genes_found, "genes_missing": res.genes_missing, "analysis_id": res.record.analysis_id})


class GroupInput(BaseModel):
    dataset_id: str
    panel_id: str = "neuronal_activity_candidates"
    grouping: str = "subtype"
    idh_wildtype_only: bool = False


@register("compare_molecular_groups", "Compare a signature score across molecular subgroups (Kruskal-Wallis, pairwise Mann-Whitney with BH, Hedges' g, Cliff's delta).", GroupInput, "bioinformatics")
def compare_molecular_groups(ws: Workspace, inp: GroupInput) -> ToolOutput:
    cohort = _cohort(ws, inp.dataset_id)
    c = cohort.primary_tumours()
    p = _panel(inp.panel_id)
    s = pd.Series(score_signature(c, p.genes, inp.panel_id).scores)
    if inp.grouping == "subtype":
        labs, source = _subtype_labels(c)
    else:
        labs, source = c.clinical[inp.grouping], f"clinical variable {inp.grouping}"
    note = ""
    if inp.idh_wildtype_only:
        m, note = _idh_wt_mask(c)
        s = s[m.reindex(s.index).fillna(False).astype(bool)]
    if labs.notna().sum() == 0:
        return ToolOutput(summary=f"no {inp.grouping} labels available in {inp.dataset_id}", degraded=True, limitations=[f"{inp.grouping} not available in {inp.dataset_id}; run project_subtypes for transferred labels"])
    res = compare_groups(s, labs, f"{inp.panel_id} score", inp.grouping, inp.dataset_id)
    res.record.parameters["label_source"] = source
    res.record.parameters["restriction"] = note or "none"
    ref = ws.put("groups", res)
    order = [g for g in SUBTYPE_ORDER if g in set(labs.dropna())] or None
    cap = (f"Signature score (mean z of {', '.join(p.genes[:8])}{'...' if len(p.genes) > 8 else ''}) in primary tumours of {inp.dataset_id}{'; ' + note if note else ''}. Labels: {source}. "
           f"Kruskal-Wallis p = {res.omnibus_p:.2e}, epsilon-squared = {res.effect_size:.3f}. Pairwise BH-adjusted Mann-Whitney in the table view. Exploratory.")
    fid = ws.put_figure(f"groups_{inp.panel_id}_{inp.dataset_id}{'_idhwt' if inp.idh_wildtype_only else ''}", F.group_box_figure(s, labs, "Signature score (mean z)", f"{p.name}: {inp.dataset_id} by {inp.grouping}", cap, order=order))
    tid = ws.put_table(f"pairwise_{inp.panel_id}_{inp.dataset_id}{'_idhwt' if inp.idh_wildtype_only else ''}", pd.DataFrame(res.pairwise))
    return ToolOutput(summary=f"Kruskal-Wallis p = {res.omnibus_p:.2e}, eps2 = {res.effect_size:.3f} across {len(res.group_stats)} groups (labels: {source})", refs=[ref], figures=[fid], tables=[tid], limitations=res.record.warnings + ["bulk-tissue score: composition (normal brain, immune infiltration) differs between subtypes"], data={"analysis_id": res.record.analysis_id, "p": res.omnibus_p, "eps2": res.effect_size})


class DEInput(BaseModel):
    dataset_id: str
    panel_id: str = "neuronal_activity_candidates"
    split: Literal["median", "tertiles"] = "median"
    test: Literal["welch", "mannwhitney"] = "welch"


@register("perform_differential_expression", "Genome-wide differential expression between high- and low-score primary tumours (Welch t / Mann-Whitney, BH FDR, Hedges' g with CI).", DEInput, "bioinformatics")
def perform_differential_expression(ws: Workspace, inp: DEInput) -> ToolOutput:
    c = _cohort(ws, inp.dataset_id).primary_tumours()
    p = _panel(inp.panel_id)
    s = pd.Series(score_signature(c, p.genes, inp.panel_id).scores)
    labs = dichotomise(s, inp.split)
    de = differential_expression(c, labs, "high", "low", test=inp.test)
    de.record.parameters.update({"panel": inp.panel_id, "split": inp.split})
    ref = ws.put("de", de)
    fid = ws.put_figure(f"volcano_{inp.panel_id}_{inp.dataset_id}", F.volcano_figure(de, highlight=p.genes))
    df = de.to_frame()
    tid = ws.put_table(f"de_{inp.panel_id}_{inp.dataset_id}", df.head(500))
    up = df[(df.q_value < 0.05) & (df.log2_fc > 0.5)].gene.tolist()
    dn = df[(df.q_value < 0.05) & (df.log2_fc < -0.5)].gene.tolist()
    lims = de.record.limitations + ["Groups are defined by a score built from some of the tested genes: those genes are trivially differential and are reported only for orientation."]
    return ToolOutput(summary=f"{de.n_tested} genes tested; {de.n_significant} at q < 0.05; {len(up)} up and {len(dn)} down with |log2FC| > 0.5", refs=[ref], figures=[fid], tables=[tid], limitations=lims, data={"ref": ref, "n_up": len(up), "n_down": len(dn), "analysis_id": de.record.analysis_id})


class EnrichInput(BaseModel):
    de_ref: str
    library: str = "GO_Biological_Process_2023"
    method: Literal["ora", "prerank"] = "ora"
    direction: Literal["up", "down"] = "up"
    permutations: int = Field(1000, ge=100, le=5000)


@register("perform_pathway_enrichment", "Pathway enrichment of a differential-expression result: hypergeometric ORA with measured-gene background, or pre-ranked GSEA on Hedges' g.", EnrichInput, "bioinformatics")
def perform_pathway_enrichment(ws: Workspace, inp: EnrichInput) -> ToolOutput:
    de = ws.artefacts[inp.de_ref]
    df = de.to_frame()
    did = de.record.dataset_ids[0]
    if inp.method == "ora":
        sel = df[(df.q_value < 0.05) & ((df.log2_fc > 0.5) if inp.direction == "up" else (df.log2_fc < -0.5))].gene.tolist()
        if len(sel) < 5:
            return ToolOutput(summary="too few significant genes for ORA", degraded=True, limitations=["fewer than five genes passed the DE threshold"])
        enr = over_representation(sel, inp.library, df.gene.tolist(), did)
        tag = f"ora_{inp.direction}"
    else:
        enr = prerank_gsea(df.set_index("gene")["hedges_g"], inp.library, did, permutations=inp.permutations)
        tag = "gsea"
    ref = ws.put("enrichment", enr)
    fid = ws.put_figure(f"enrichment_{tag}_{inp.library}_{did}", F.enrichment_figure(enr))
    tid = ws.put_table(f"enrichment_{tag}_{inp.library}_{did}", pd.DataFrame(enr.terms))
    top = "; ".join(f"{t['term']} (q={t['q']:.1e})" for t in enr.terms[:3])
    return ToolOutput(summary=f"{enr.method} on {inp.library}: top terms {top}", refs=[ref], figures=[fid], tables=[tid], limitations=enr.record.limitations, data={"analysis_id": enr.record.analysis_id, "top_terms": [t["term"] for t in enr.terms[:10]]})


class CorrInput(BaseModel):
    dataset_id: str
    panel_id: str = "neuron_glioma_extended"
    method: Literal["spearman", "pearson"] = "spearman"


@register("compute_gene_correlation", "Gene-gene correlation matrix of a panel in primary tumours, BH-adjusted.", CorrInput, "bioinformatics")
def compute_gene_correlation(ws: Workspace, inp: CorrInput) -> ToolOutput:
    c = _cohort(ws, inp.dataset_id).primary_tumours()
    p = _panel(inp.panel_id)
    cr = gene_correlation(c, p.genes, inp.method)
    ref = ws.put("correlation", cr)
    fid = ws.put_figure(f"correlation_{inp.panel_id}_{inp.dataset_id}", F.correlation_figure(cr))
    hm = ws.put_figure(f"subtype_heatmap_{inp.panel_id}_{inp.dataset_id}", F.subtype_gene_heatmap(c, p.genes)) if c.clinical["subtype"].notna().any() else None
    m = np.array(cr.matrix)
    iu = np.triu_indices(len(cr.genes), 1)
    return ToolOutput(summary=f"{len(cr.genes)} genes, n = {cr.n}; median pairwise rho = {np.median(m[iu]):.2f}", refs=[ref], figures=[fid] + ([hm] if hm else []), limitations=cr.record.limitations + cr.record.warnings, data={"median_rho": float(np.median(m[iu]))})


# ---------------------------------------------------------------- statistics
class SurvivalInput(BaseModel):
    dataset_id: str
    panel_id: str = "neuronal_activity_candidates"
    covariates: list[str] = Field(default_factory=lambda: ["age", "sex", "subtype", "g_cimp", "idh_status", "mgmt_status"])
    split: Literal["median", "tertiles"] = "median"
    idh_wildtype_only: bool = False


@register("perform_survival_analysis", "Kaplan-Meier (score split) with log-rank, univariable and multivariable Cox PH (per-SD HR) with Schoenfeld-residual PH tests, in primary tumours.", SurvivalInput, "statistics")
def perform_survival_analysis(ws: Workspace, inp: SurvivalInput) -> ToolOutput:
    cohort = _cohort(ws, inp.dataset_id)
    c = cohort.primary_tumours()
    restr = ""
    if inp.idh_wildtype_only:
        m, restr = _idh_wt_mask(c)
        c = c.subset(c.clinical.index[m.fillna(False).astype(bool)])
    p = _panel(inp.panel_id)
    s = pd.Series(score_signature(c, p.genes, inp.panel_id).scores)
    km = kaplan_meier(c.clinical, dichotomise(s, inp.split), f"{inp.panel_id} score ({inp.split} split)", inp.dataset_id)
    cov = [v for v in inp.covariates if v in c.clinical and c.clinical[v].notna().mean() >= 0.5 and not (inp.idh_wildtype_only and v in ("idh_status", "g_cimp"))]
    uni = cox_model(c.clinical, s, inp.panel_id, inp.dataset_id, [])
    multi = cox_model(c.clinical, s, inp.panel_id, inp.dataset_id, cov) if cov else None
    refs = [ws.put("km", km), ws.put("cox", uni)] + ([ws.put("cox", multi)] if multi else [])
    if restr:
        km.record.parameters["restriction"] = restr
    figs = [ws.put_figure(f"km_{inp.panel_id}_{inp.dataset_id}", F.kaplan_meier_figure(km, f"Overall survival by {p.name.split('(')[0].strip()} score: {inp.dataset_id}"))]
    if multi:
        figs.append(ws.put_figure(f"cox_{inp.panel_id}_{inp.dataset_id}", F.cox_forest_figure(multi, f"Multivariable Cox model: {inp.dataset_id}")))
    r_uni = exposure_row(uni, inp.panel_id)
    r_multi = exposure_row(multi, inp.panel_id) if multi else None
    lims = km.record.limitations + uni.record.limitations + (multi.record.limitations + multi.record.warnings if multi else [])
    dropped = sorted(set(inp.covariates) - set(cov))
    if dropped:
        lims.append(f"covariates unavailable or >50% missing in {inp.dataset_id}: {dropped}")
    summ = (f"[{restr}, n = {len(c.samples)}] " if restr else "") + f"log-rank p = {km.logrank_p:.3f}; univariable HR/SD = {r_uni.hr:.2f} ({r_uni.hr_ci_low:.2f}-{r_uni.hr_ci_high:.2f}), p = {r_uni.p:.3f}"
    if r_multi:
        summ += f"; adjusted HR/SD = {r_multi.hr:.2f} ({r_multi.hr_ci_low:.2f}-{r_multi.hr_ci_high:.2f}), p = {r_multi.p:.3f} [{', '.join(multi.covariates_adjusted)}]; PH ok: {multi.ph_assumption_ok}"
    return ToolOutput(summary=summ, refs=refs, figures=figs, limitations=lims, data={"logrank_p": km.logrank_p, "hr_uni": r_uni.hr, "p_uni": r_uni.p, "hr_adj": r_multi.hr if r_multi else None, "p_adj": r_multi.p if r_multi else None, "ph_ok": multi.ph_assumption_ok if multi else uni.ph_assumption_ok,
                                                                              "analysis_ids": [km.record.analysis_id, uni.record.analysis_id] + ([multi.record.analysis_id] if multi else [])})


class CompositionInput(BaseModel):
    dataset_ids: list[str]
    panel_id: str = "neuronal_activity_candidates"
    control_panel_id: str = "normal_neuron_markers"


@register("assess_composition_confounding", "Quantify how much of a bulk signature is explained by mature-neuron marker expression (a proxy for infiltrated normal brain) and whether subtype differences survive residualisation.", CompositionInput, "statistics")
def assess_composition_confounding(ws: Workspace, inp: CompositionInput) -> ToolOutput:
    p, ctrl = _panel(inp.panel_id), _panel(inp.control_panel_id)
    rows = []
    for did in inp.dataset_ids:
        if did not in ws.cohorts:
            continue
        cohort = ws.cohorts[did]
        out = composition_check(cohort, p.genes, ctrl.genes)
        c = cohort.primary_tumours()
        labs, source = _subtype_labels(c)
        if labs.notna().sum() > 10 and "kw_p_raw" not in out:
            from scipy import stats

            s = pd.Series(score_signature(c, p.genes, "c").scores)
            cs = pd.Series(score_signature(c, ctrl.genes, "n").scores)
            resid = residualise(s, cs)
            m, _ = _idh_wt_mask(c)
            keep = [k for k in labs.dropna().unique() if ((labs == k) & m).sum() >= 5]
            out["kw_p_raw"] = float(stats.kruskal(*[s[(labs == k) & m].dropna() for k in keep]).pvalue)
            out["kw_p_residualised"] = float(stats.kruskal(*[resid.reindex(labs.index[(labs == k) & m]).dropna() for k in keep]).pvalue)
        out["label_source"] = source
        rows.append(out)
        s = pd.Series(score_signature(c, p.genes, "c").scores)
        cs = pd.Series(score_signature(c, ctrl.genes, "n").scores).reindex(s.index)
        cap = f"Each point is a primary tumour in {did}. x: mean z of mature-neuron markers ({', '.join(ctrl.genes)}); y: candidate score. Spearman rho = {out['spearman_rho']:.2f} (p = {out['spearman_p']:.1e}). A strong correlation means the bulk candidate score largely tracks the amount of neuronal tissue in the sample."
        ws.put_figure(f"composition_{did}", F.scatter_figure(cs, s, "Neuron-marker score (mean z)", "Candidate score (mean z)", f"Candidate score vs neuron-marker content: {did}", cap, color=labs.reindex(s.index)))
    df = pd.DataFrame(rows)
    from neurobridge.analysis.common import finish, new_record

    crec = new_record("composition_check", AnalysisKind.EXPLORATORY, [r["cohort"] for r in rows], "Spearman correlation of candidate score with mature-neuron marker score; Kruskal-Wallis on raw and neuron-marker-residualised scores", {"panel": inp.panel_id, "control_panel": inp.control_panel_id})
    crec.n_used = int(sum(r["n"] for r in rows))
    ref = ws.put("composition", {"record": finish(crec).model_dump(mode="json"), "rows": rows})
    tid = ws.put_table("composition_confounding", df.drop(columns=[c for c in ["residual_medians"] if c in df]))
    rho = ", ".join(f"{r['cohort']} rho={r['spearman_rho']:.2f}" for r in rows)
    lims = ["Neuron-marker expression is a proxy, not a measurement, of normal-brain content; tumour purity estimates (e.g. ESTIMATE, ABSOLUTE) would be a complementary control.", "Residualising on a correlated marker set can also remove genuine tumour-intrinsic neuronal programmes (over-adjustment)."]
    return ToolOutput(summary=f"candidate score vs neuron markers: {rho}", refs=[ref], tables=[tid], figures=[f"composition_{r['cohort']}" for r in rows], limitations=lims, data={"rows": rows, "analysis_id": crec.analysis_id})


class CellTypeInput(BaseModel):
    dataset_id: str = "gse84465_darmanis"
    panel_id: str = "neuronal_activity_candidates"


@register("perform_cell_type_analysis", "Single-cell analysis: QC, normalisation, UMAP/Leiden, per-cell-type expression of the panel, neoplastic-vs-rest Wilcoxon tests, neoplastic core-vs-periphery comparison.", CellTypeInput, "bioinformatics")
def perform_cell_type_analysis(ws: Workspace, inp: CellTypeInput) -> ToolOutput:
    raw = ws.single_cell.get(inp.dataset_id + ":raw") or load_single_cell(inp.dataset_id)
    key = inp.dataset_id + ":processed"
    if key not in ws.single_cell:
        a, rec = scm.preprocess(raw)
        ws.single_cell[key] = a
        ws.put("sc_preprocess", rec)
    a = ws.single_cell[key]
    p = _panel(inp.panel_id)
    tab, rec_ct = scm.cell_type_expression(a, p.genes)
    ref_ct = ws.put("sc_celltype", {"record": rec_ct.model_dump(mode="json"), "table": tab.round(4).reset_index().to_dict(orient="list")})
    mk, rec_mk = scm.marker_test(a, "cell_type_author", "Neoplastic", "rest", genes=p.genes)
    ws.put("sc_marker", {"record": rec_mk.model_dump(mode="json"), "table": mk.to_dict(orient="records")})
    rec_sc = scm.score_cells(a, p.genes, "candidate_score")
    by_type = scm.compare_score_by_group(a, "candidate_score", "cell_type_author")
    neo = a[a.obs["cell_type_author"] == "Neoplastic"].copy()
    loc_tab, rec_loc = None, None
    if neo.obs["location"].nunique() > 1:
        loc_tab, rec_loc = scm.marker_test(neo, "location", "Periphery", "Tumor", genes=p.genes)
        ws.put("sc_marker", {"record": rec_loc.model_dump(mode="json"), "table": loc_tab.to_dict(orient="records")})
    n_cells = a.n_obs
    um = a.obsm["X_umap"]
    cap_u = f"UMAP of {n_cells} cells after QC (scanpy: normalise_total 1e4, log1p, 2,000 HVGs batch-aware by patient, 30 PCs, k = 15, seed {rec_sc.seed}). Colours: author cell-type annotation (GEO series matrix). Dataset GSE84465 (Darmanis et al. 2017)."
    figs = [ws.put_figure("umap_celltype", F.umap_figure(a.obs, um, "cell_type_author", "Single-cell landscape (GSE84465)", cap_u)),
            ws.put_figure("umap_patient", F.umap_figure(a.obs, um, "patient", "Cells by patient (GSE84465)", "Same embedding coloured by patient: neoplastic cells cluster by patient, non-neoplastic cells mix across patients. No integration was applied.")),
            ws.put_figure("umap_score", F.umap_figure(a.obs, um, "candidate_score", "Candidate-gene score per cell (GSE84465)", "scanpy score_genes: mean expression of the candidate genes minus a binned random control set (seeded).", continuous=True)),
            ws.put_figure("dotplot_celltype", F.dotplot_figure(tab, "Candidate-gene expression by cell type (GSE84465)", "Dot size: fraction of cells with >0 counts; colour: mean log1p(CP10k) over all cells of the type. Author annotations. Neurons are rare (n shown); estimates for small groups are imprecise."))]
    tables = [ws.put_table("sc_neoplastic_vs_rest", mk), ws.put_table("sc_score_by_celltype", by_type)]
    if loc_tab is not None:
        tables.append(ws.put_table("sc_neoplastic_periphery_vs_core", loc_tab))
    top = by_type.iloc[0]
    lims = rec_mk.limitations + rec_ct.limitations + ["Smart-seq2 dissociation under-samples mature neurons; neuronal transcripts in neoplastic cells cannot be distinguished from ambient RNA without additional controls."]
    det = tab["fraction_detected"]
    return ToolOutput(summary=f"{n_cells} cells; highest candidate score in {top['group']} (median {top['median']:.2f}); neoplastic-vs-rest q: " + ", ".join(f"{r.gene} {r.q:.1e}" for r in mk.itertuples()),
                      refs=[ref_ct], figures=figs, tables=tables, limitations=lims,
                      data={"score_by_type": by_type.to_dict(orient="records"), "neoplastic_vs_rest": mk.to_dict(orient="records"), "periphery_vs_core": loc_tab.to_dict(orient="records") if loc_tab is not None else None, "periphery_analysis_id": rec_loc.analysis_id if rec_loc else None,
                            "fraction_detected": det.round(3).to_dict(orient="index"), "analysis_ids": [rec_ct.analysis_id, rec_mk.analysis_id, rec_sc.analysis_id]})


# ---------------------------------------------------------------- validation
class ProjectInput(BaseModel):
    train_id: str = "tcga_gbm_rnaseq"
    target_ids: list[str] = Field(default_factory=lambda: ["cgga_325", "cgga_693"])


@register("project_subtypes", "Train a transcriptional-subtype classifier on TCGA (candidate and neuron-marker genes excluded from features) and project labels onto cohorts without subtype annotation.", ProjectInput, "validation")
def project_subtypes(ws: Workspace, inp: ProjectInput) -> ToolOutput:
    train = _cohort(ws, inp.train_id)
    targets = {t: ws.cohorts[t] for t in inp.target_ids if t in ws.cohorts}
    if not targets:
        return ToolOutput(summary="no target cohorts loaded", degraded=True, limitations=["validation cohorts unavailable; subtype hypothesis cannot be validated"])
    panels = all_panels()
    excl = sorted(set(panels["neuronal_activity_candidates"].genes) | set(panels["neuron_glioma_extended"].genes) | set(panels["normal_neuron_markers"].genes) | set(panels.get("kegg_glutamatergic_synapse").genes if "kegg_glutamatergic_synapse" in panels else []))
    res = train_and_project(train, targets, excl)
    for tid, pr in res["projections"].items():
        ws.cohorts[tid].clinical["subtype_projected"] = pd.Series(pr["labels"])
        ws.cohorts[tid].clinical["subtype_projected_confidence"] = pd.Series(pr["confidence"])
    ref = ws.put("subtype_projection", {k: v for k, v in res.items() if k not in ("features",)} | {"n_features": len(res["features"])})
    cm = pd.DataFrame(res["cv_confusion"]["matrix"], index=[f"true {l}" for l in res["cv_confusion"]["labels"]], columns=[f"pred {l}" for l in res["cv_confusion"]["labels"]])
    ws.put_table("subtype_classifier_cv_confusion", cm)
    return ToolOutput(summary=f"TCGA 5-fold CV balanced accuracy = {res['cv_balanced_accuracy']:.2f} (n = {res['n_train']}); projected: " + "; ".join(f"{k} {v['counts']}" for k, v in res["projections"].items()),
                      refs=[ref], tables=["subtype_classifier_cv_confusion"], limitations=res["record"].limitations, data={"cv_bacc": res["cv_balanced_accuracy"], "analysis_id": res["record"].analysis_id})


class ValidateInput(BaseModel):
    hypothesis: Literal["subtype_difference", "survival_association"]
    discovery_id: str = "tcga_gbm_rnaseq"
    validation_ids: list[str] = Field(default_factory=lambda: ["cgga_325", "cgga_693"])
    panel_id: str = "neuronal_activity_candidates"
    group_a: str = "Mesenchymal"
    group_b: str = "Proneural"
    covariates: list[str] = Field(default_factory=lambda: ["age", "sex", "g_cimp", "idh_status", "mgmt_status"])
    residualise_on_neuron_markers: bool = False
    idh_wildtype_only: bool = False


@register("validate_across_cohorts", "Confirmatory replication of a pre-specified hypothesis in independent cohorts (direction + significance rule, random-effects meta-analysis). Never reports 'validated' from one cohort.", ValidateInput, "validation")
def validate_across_cohorts(ws: Workspace, inp: ValidateInput) -> ToolOutput:
    ids = [inp.discovery_id] + [v for v in inp.validation_ids if v in ws.cohorts]
    missing = [v for v in inp.validation_ids if v not in ws.cohorts]
    p = _panel(inp.panel_id)
    ctrl = _panel("normal_neuron_markers")
    lims = [f"validation cohort {m} unavailable" for m in missing]
    suffix = "_resid" if inp.residualise_on_neuron_markers else ""
    if inp.hypothesis == "survival_association":
        if inp.residualise_on_neuron_markers:
            lims.append("residualised survival validation is reported in the sensitivity analysis")
        cohorts = {i: ws.cohorts[i] for i in ids}
        covs = inp.covariates
        if inp.idh_wildtype_only:
            restricted = {}
            for i, co in cohorts.items():
                pt = co.primary_tumours()
                m, _ = _idh_wt_mask(pt)
                restricted[i] = pt.subset(pt.clinical.index[m.fillna(False).astype(bool)])
            cohorts = restricted
            covs = [v for v in covs if v not in ("idh_status", "g_cimp")]
            lims.append("Restricted to IDH-wildtype tumours (non-G-CIMP as proxy in TCGA); IDH/G-CIMP removed from the adjustment set.")
        vs, _ = validate_signature_survival(cohorts, inp.discovery_id, p.genes, inp.panel_id, covs)
        if inp.idh_wildtype_only:
            vs.hypothesis += " in IDH-wildtype tumours"
        fid = ws.put_figure(f"validation_survival_{inp.panel_id}", F.validation_forest_figure(vs, f"Cross-cohort validation: {p.name} and overall survival"))
    else:
        cs = {}
        for i in ids:
            c = ws.cohorts[i].primary_tumours()
            labs, source = _subtype_labels(c)
            m, mnote = _idh_wt_mask(c)
            s = pd.Series(score_signature(c, p.genes, inp.panel_id).scores)
            if inp.residualise_on_neuron_markers:
                s = residualise(s, pd.Series(score_signature(c, ctrl.genes, "n").scores))
            keep = m.reindex(s.index).fillna(False).astype(bool)
            cs[i] = (s[keep], labs.reindex(s.index)[keep], f"{source}; {mnote}")
        vs = validate_group_difference(cs, inp.discovery_id, inp.group_a, inp.group_b, f"{inp.panel_id} score{' (residualised on neuron markers)' if inp.residualise_on_neuron_markers else ''}")
        fid = ws.put_figure(f"validation_subtype_{inp.panel_id}{suffix}", F.validation_forest_figure(vs, f"Cross-cohort validation: {inp.group_a} vs {inp.group_b}{' (neuron-marker residualised)' if suffix else ''}"))
    ref = ws.put("validation", vs)
    tid = ws.put_table(f"validation_{inp.hypothesis}_{inp.panel_id}{suffix}", pd.DataFrame([r.model_dump() for r in vs.per_cohort]))
    return ToolOutput(summary=f"{vs.hypothesis}: {vs.verdict}", refs=[ref], figures=[fid], tables=[tid], limitations=lims + vs.record.limitations, degraded=bool(missing),
                      data={"verdict": vs.verdict, "n_replicating": vs.n_cohorts_replicating, "n_validation": len([r for r in vs.per_cohort if r.role == 'validation']), "analysis_id": vs.record.analysis_id})


class SensitivityInput(BaseModel):
    dataset_ids: list[str]
    panel_id: str = "neuronal_activity_candidates"
    covariates: list[str] = Field(default_factory=lambda: ["age", "sex", "g_cimp", "idh_status", "mgmt_status"])


@register("run_sensitivity_analysis", "Re-estimate the survival association under alternative specifications: unadjusted, rank score, neuron-marker adjustment, residualisation, IDH-wildtype / non-G-CIMP only, Neural subtype excluded, tertile split.", SensitivityInput, "validation")
def run_sensitivity_analysis(ws: Workspace, inp: SensitivityInput) -> ToolOutput:
    p, ctrl = _panel(inp.panel_id), _panel("normal_neuron_markers")
    rows = []
    for did in inp.dataset_ids:
        if did in ws.cohorts:
            rows += survival_sensitivity(ws.cohorts[did], p.genes, inp.panel_id, ctrl.genes, inp.covariates)
    df = pd.DataFrame(rows)
    from neurobridge.analysis.common import finish, new_record

    srec = new_record("survival_sensitivity", AnalysisKind.EXPLORATORY, list(inp.dataset_ids), "Cox PH re-estimated under alternative specifications (see table); tertile log-rank", {"panel": inp.panel_id, "covariates": inp.covariates, "n_specifications": len(rows)})
    ref = ws.put("sensitivity", {"record": finish(srec).model_dump(mode="json"), "rows": rows})
    tid = ws.put_table(f"sensitivity_{inp.panel_id}", df)
    fr = [{"label": f"{r['cohort']}: {r['specification']}", "est": r["hr"], "lo": r["ci_low"], "hi": r["ci_high"], "p": r["p"], "color": F.COHORT_COLORS.get(r["cohort"], F.CYAN)} for r in rows if not np.isnan(r["hr"])]
    fid = ws.put_figure(f"sensitivity_{inp.panel_id}", F.forest_figure(fr, "Sensitivity of the survival association", "Cox HR per 1 SD of the candidate score under alternative specifications, in primary tumours of each cohort. 'Adjusted' uses the cohort's available covariates (see table). Exploratory robustness checks; no multiplicity correction across specifications."))
    sig = df[(df.p < 0.05) & df.hr.notna()]
    return ToolOutput(summary=f"{len(df)} specifications across {df.cohort.nunique()} cohorts; {len(sig)} with p < 0.05 ({', '.join(sorted(set(sig.cohort)) ) or 'none'})", refs=[ref], figures=[fid], tables=[tid], limitations=["Specifications are not independent and are not corrected for multiplicity; they test robustness, not new hypotheses."], data={"rows": rows, "analysis_id": srec.analysis_id})


# ---------------------------------------------------------------- reporting
class ReportInput(BaseModel):
    title: str = "NeuroBridge evidence report"


@register("generate_scientific_report", "Assemble the structured evidence report (Markdown) from executed analyses, graded claims and verified citations.", ReportInput, "report")
def generate_scientific_report(ws: Workspace, inp: ReportInput) -> ToolOutput:
    # The report needs the full investigation context; the orchestrator calls neurobridge.report.builder directly
    # and this tool exists so the step is logged and can be invoked by an external (MCP) client.
    return ToolOutput(summary="report requested", data={"title": inp.title})


# ---------------------------------------------------------------- spatial
class SpatialInput(BaseModel):
    dataset_id: str = "visium_gbm_10x"
    panel_id: str = "neuronal_activity_candidates"
    extra_genes: list[str] = Field(default_factory=lambda: ["NTRK2", "THBS1", "GAP43", "GJA1"])
    n_perm: int = Field(999, ge=99, le=9999)


@register("perform_spatial_analysis", "Spatial transcriptomics (Visium): spot QC, spatial domains, programme scores, Moran's I, neuron-interface gradients and ligand-receptor neighbour co-localisation with domain-stratified permutation nulls.", SpatialInput, "bioinformatics", retryable=True)
def perform_spatial_analysis(ws: Workspace, inp: SpatialInput) -> ToolOutput:
    from neurobridge.analysis import spatial as spa
    from neurobridge.data.loaders import load_spatial
    from neurobridge.data.spatial import lowres_image

    key = inp.dataset_id + ":processed"
    if key not in ws.single_cell:
        a, rec_pre = spa.preprocess_spatial(load_spatial(inp.dataset_id))
        ws.single_cell[key] = a
        ws.put("spatial_preprocess", rec_pre)
    a = ws.single_cell[key]
    dom = spa.annotate_domains(a)
    p = _panel(inp.panel_id)
    res, rec_p = spa.spatial_programme_analysis(a, p.genes, n_perm=inp.n_perm)
    genes_if = list(dict.fromkeys(p.genes + inp.extra_genes))
    itab, dist, rec_i = spa.interface_analysis(a, genes_if, res["neighbours"])
    lr, rec_lr = spa.neighbour_lr(a, res["neighbours"], n_perm=inp.n_perm)
    for kind, rec, tab in [("spatial_programmes", rec_p, res["correlations"]), ("spatial_interface", rec_i, itab), ("spatial_lr", rec_lr, lr)]:
        ws.put(kind, {"record": rec.model_dump(mode="json"), "table": tab.to_dict(orient="records")})
    img = lowres_image(a)
    base = f"Visium section (10x Genomics, one GBM patient), {a.n_obs} spots after QC."
    ws.put_figure("spatial_domains", F.spatial_figure(a, a.obs["domain_programme"].values, "Spatial domains by dominant programme", base + f" {dom.shape[0]} Leiden domains on spot expression, each coloured by its most enriched programme (neuron, tumour/reactive-glial, myeloid, vascular); per-domain table in the table view. Programme labels are proxies, not cell-type calls.", continuous=False, image=img))
    ws.put_figure("spatial_candidate", F.spatial_figure(a, a.obs["candidate_score"].values, "Candidate-gene score per spot", base + f" scanpy score_genes of {', '.join(p.genes)}. Compare with the mature-neuron programme map. Moran's I = {res['correlations'].set_index('feature').loc['candidate_score', 'morans_I']:.2f}.", image=img))
    ws.put_figure("spatial_neuron", F.spatial_figure(a, a.obs["prog_neuron"].values, "Mature-neuron programme per spot", base + " Score of SNAP25, SYT1, RBFOX3, GABRA1, NEFL, STMN2, SYN1, GAD1. The top 20% of spots define the neuron-rich region used in the interface analysis.", image=img))
    top_if = itab[itab.feature != "candidate_score"].sort_values("p_distance_neuron_adjusted").iloc[0]["feature"]
    ws.put_figure("spatial_interface_gene", F.spatial_figure(a, spa._gene(a, top_if), f"{top_if} expression per spot", base + f" log1p(CP10k). {top_if} shows the strongest neuron-adjusted gradient with distance to neuron-rich spots (rings), adjusted for each spot's neuron score.", image=img))
    lrs = lr.copy()
    lrs["pair"] = lrs["ligand"] + " → " + lrs["receptor"]
    fr = [{"label": r.pair, "est": r.z_domain, "lo": r.z_domain, "hi": r.z_domain, "p": r.p_domain, "color": F.TEAL if r.q_domain < 0.05 else F.DIM} for r in lrs.itertuples()]
    ws.put_figure("spatial_lr", F.forest_figure(fr, "Ligand–receptor neighbour co-localisation (domain-stratified null)", f"z-score of the observed neighbour co-expression against a null in which the ligand is permuted within spatial domains ({inp.n_perm} permutations). Blue: BH q < 0.05. Co-localisation of transcripts, not evidence of interaction.", ref=0.0, xlabel="z (domain-stratified)", log_x=False))
    tids = [ws.put_table("spatial_correlations", res["correlations"]), ws.put_table("spatial_interface", itab), ws.put_table("spatial_ligand_receptor", lr), ws.put_table("spatial_domains", dom.reset_index())]
    cor = res["correlations"].set_index("feature")
    lims = list(dict.fromkeys(rec_p.limitations + rec_i.limitations + rec_lr.limitations))
    return ToolOutput(
        summary=f"{a.n_obs} spots, {dom.shape[0]} domains; candidate score vs neuron programme rho = {cor.loc['candidate_score', 'rho_neuron']:.2f}; {int((lr.q_domain < 0.05).sum())}/{len(lr)} ligand-receptor pairs co-localise beyond domain composition",
        figures=["spatial_domains", "spatial_candidate", "spatial_neuron", "spatial_interface_gene", "spatial_lr"], tables=tids, limitations=lims,
        data={"n_spots": int(a.n_obs), "correlations": res["correlations"].to_dict(orient="records"), "interface": itab.to_dict(orient="records"), "lr": lr.to_dict(orient="records"),
              "n_neuron_rich": rec_i.parameters["n_neuron_rich"], "analysis_ids": [rec_p.analysis_id, rec_i.analysis_id, rec_lr.analysis_id]})
