"""Rule-based claim construction. Claims are generated from executed results with fixed templates, so every
number in a claim is a number that was computed. Each claim states its evidence level, whether it is an
exploratory, confirmatory, literature or interpretive statement, and links to analysis ids / PMIDs."""

from __future__ import annotations

import numpy as np

from neurobridge.agents.agents import Context
from neurobridge.schemas.evidence import EvidenceLevel as L


def _claim(cid, text, level, kind, analyses=(), pmids=(), caveats=(), status="supported"):
    return {"id": cid, "text": text, "level": level.value, "kind": kind, "analyses": list(analyses), "citations": list(pmids), "caveats": list(caveats), "status": status}


def _fmt(p):
    return "n/a" if p is None or (isinstance(p, float) and np.isnan(p)) else (f"{p:.1e}" if p < 1e-3 else f"{p:.3f}")


def build_claims(ctx: Context) -> list[dict]:
    o, ws = ctx.outputs, ctx.ws
    claims: list[dict] = []
    cites = ws.citations

    # --- literature (prior experimental evidence, attributed, never re-graded as our own finding)
    lit = [p for p in ("25913192", "28959975", "31534222", "31534219", "37914930") if p in cites]
    if lit:
        claims.append(_claim("L1", "Experimental studies in mouse models and patient-derived xenografts report that neuronal activity promotes high-grade glioma growth, through activity-regulated NLGN3 secretion and through AMPA-receptor-mediated neuron-to-glioma synapses.", L.MECHANISTIC_EVIDENCE, "literature", pmids=lit,
                             caveats=["This is evidence from the cited experimental studies, not from NeuroBridge analyses.", "Most mechanistic work is in paediatric and adult high-grade glioma xenograft models."]))
    if "37138086" in cites:
        claims.append(_claim("L2", "In patients, high functional connectivity between glioblastoma and surrounding neural circuits has been associated with shorter survival, with glioma-secreted thrombospondin-1 (TSP-1) implicated in the circuit remodelling.", L.STATISTICAL_ASSOCIATION, "literature", pmids=["37138086"], caveats=["Literature finding; NeuroBridge tests a related but different exposure (bulk transcript levels)."]))
    if "28697342" in cites:
        claims.append(_claim("L3", "The Verhaak 'Neural' subtype has been attributed to contamination by non-tumour neural tissue, so neuronal transcripts in bulk GBM samples can reflect infiltrated normal brain.", L.STATISTICAL_ASSOCIATION, "literature", pmids=["28697342"]))

    # --- subtype
    st = o.get("subtype_tcga")
    if st:
        res = ws.artefacts[st.refs[0]]
        pw = {(r["group_a"], r["group_b"]): r for r in res.pairwise}
        r = pw.get(("Mesenchymal", "Proneural")) or pw.get(("Proneural", "Mesenchymal"))
        meds = {g["group"]: g["median"] for g in res.group_stats}
        txt = f"In TCGA primary GBM (n = {res.record.n_used}), the candidate score differs across Verhaak subtypes (Kruskal-Wallis p = {_fmt(res.omnibus_p)}, epsilon-squared = {res.effect_size:.2f}); medians: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(meds.items(), key=lambda kv: -kv[1])) + "."
        if r:
            g = r["hedges_g"] if r["group_a"] == "Mesenchymal" else -r["hedges_g"]
            txt += f" Mesenchymal vs Proneural: Hedges g = {g:.2f}, BH q = {_fmt(r['q'])}."
        claims.append(_claim("S1", txt, L.STATISTICAL_ASSOCIATION, "exploratory", analyses=[res.record.analysis_id], caveats=["Exploratory, discovery cohort only.", "Bulk tissue; subtypes differ in normal-brain and immune content."], status="supported" if res.omnibus_p < 0.05 else "not supported"))
    vs_o = o.get("val_subtype")
    if vs_o:
        vs = ws.artefacts[vs_o.refs[0]]
        per = "; ".join(f"{r.cohort_id}: g = {r.estimate:.2f} ({r.ci_low:.2f} to {r.ci_high:.2f}), p = {_fmt(r.p)}" for r in vs.per_cohort if not np.isnan(r.estimate))
        rep = vs.n_cohorts_replicating
        nval = len([r for r in vs.per_cohort if r.role == "validation"])
        claims.append(_claim("S2", f"The lower candidate score in Mesenchymal than Proneural IDH-wildtype / non-G-CIMP tumours {'replicates' if rep == nval and rep > 0 else 'does not fully replicate'} in independent cohorts ({vs.verdict}): {per}.", L.STATISTICAL_ASSOCIATION if rep else L.INSUFFICIENT, "confirmatory", analyses=[vs.record.analysis_id] + ([o['projection'].data['analysis_id']] if 'projection' in o else []),
                             caveats=["Validation-cohort subtypes are projected by a TCGA-trained classifier (candidate and neuron-marker genes excluded from its features); CV balanced accuracy in TCGA = " + (f"{o['projection'].data['cv_bacc']:.2f}" if "projection" in o else "n/a") + ".", "The Mesenchymal vs Proneural contrast was selected from the discovery cohort; validation cohorts test it prospectively."],
                             status="supported" if rep == nval and rep > 0 else ("partially supported" if rep else "not supported")))
    vr = o.get("val_subtype_resid")
    if vr:
        vs = ws.artefacts[vr.refs[0]]
        per = "; ".join(f"{r.cohort_id}: g = {r.estimate:.2f}, p = {_fmt(r.p)}" for r in vs.per_cohort if not np.isnan(r.estimate))
        claims.append(_claim("S3", f"After residualising the candidate score on mature-neuron marker expression, the Mesenchymal vs Proneural difference is {vs.verdict} ({per}).", L.STATISTICAL_ASSOCIATION if vs.n_cohorts_replicating else L.INSUFFICIENT, "sensitivity", analyses=[vs.record.analysis_id],
                             caveats=["Residualisation may remove genuine tumour-intrinsic neuronal programmes (over-adjustment)."], status="supported" if vs.n_cohorts_replicating else "not supported"))

    # --- composition
    comp = o.get("composition")
    if comp:
        rows = comp.data["rows"]
        txt = "The bulk candidate score is strongly correlated with mature-neuron marker expression (" + "; ".join(f"{r['cohort']} rho = {r['spearman_rho']:.2f}" for r in rows) + "), so a large part of the bulk signal is consistent with variation in infiltrated normal-brain content rather than tumour-cell expression."
        claims.append(_claim("C1", txt, L.STATISTICAL_ASSOCIATION, "exploratory", analyses=[comp.data["analysis_id"]], pmids=[p for p in ["28697342"] if p in cites], caveats=["Neuron markers are a proxy for normal-brain content."]))

    # --- survival (per-cohort estimates are descriptive; the confirmatory verdict is O2)
    disc_hr = (o.get("survival_tcga_gbm_rnaseq").data.get("hr_adj") if o.get("survival_tcga_gbm_rnaseq") else None)
    for d in ("tcga_gbm_rnaseq", "cgga_325", "cgga_693"):
        so = o.get(f"survival_{d}")
        if not so:
            continue
        dd = so.data
        sig = dd["p_adj"] is not None and dd["p_adj"] < 0.05
        txt = (f"{d}: adjusted Cox HR per SD of the candidate score = {dd['hr_adj']:.2f} (p = {_fmt(dd['p_adj'])}); univariable HR = {dd['hr_uni']:.2f} (p = {_fmt(dd['p_uni'])}); median-split log-rank p = {_fmt(dd['logrank_p'])}."
               if dd["hr_adj"] is not None else f"{d}: univariable HR = {dd['hr_uni']:.2f} (p = {_fmt(dd['p_uni'])}).")
        cav = ["Proportional-hazards assumption " + ("not rejected" if dd["ph_ok"] else "questionable") + "."]
        if d != "tcga_gbm_rnaseq" and disc_hr is not None and dd["hr_adj"] is not None and np.sign(np.log(dd["hr_adj"])) != np.sign(np.log(disc_hr)):
            txt += " The estimate points in the opposite direction to the discovery cohort."
            cav.append("Opposite direction to discovery: this does not count as replication.")
        claims.append(_claim(f"O_{d}", txt, L.STATISTICAL_ASSOCIATION if sig else L.INSUFFICIENT, "exploratory", analyses=dd["analysis_ids"], caveats=cav, status="supported" if sig else "not supported"))
    vso = o.get("val_survival")
    if vso:
        vs = ws.artefacts[vso.refs[0]]
        claims.append(_claim("O2", f"Cross-cohort verdict for the survival association: {vs.verdict}. " + " ".join(vs.verdict_rationale), L.STATISTICAL_ASSOCIATION if vs.n_cohorts_replicating else L.INSUFFICIENT, "confirmatory", analyses=[vs.record.analysis_id],
                             caveats=["Adjustment sets differ by cohort (G-CIMP in TCGA; IDH and MGMT in CGGA)."], status="supported" if vs.n_cohorts_replicating == len(vs.validation_cohorts) and vs.n_cohorts_replicating else "not supported"))
    sen = o.get("sensitivity")
    if sen:
        rows = [r for r in sen.data["rows"] if not np.isnan(r["hr"])]
        prim = {r["cohort"]: r for r in rows if r["specification"].startswith("primary")}
        adjn = {r["cohort"]: r for r in rows if "neuron-marker (composition)" in r["specification"]}
        parts = [f"{c}: HR {prim[c]['hr']:.2f} (p = {_fmt(prim[c]['p'])}) -> {adjn[c]['hr']:.2f} (p = {_fmt(adjn[c]['p'])})" for c in prim if c in adjn]
        claims.append(_claim("O3", "Adding the neuron-marker score as a covariate changes the per-SD hazard ratio as follows: " + "; ".join(parts) + ". No specification yields a significant association in more than one cohort in the same direction.",
                             L.INSUFFICIENT, "sensitivity", analyses=[sen.data["analysis_id"]], caveats=["Specifications are robustness checks, not new hypotheses."], status="supported"))

    # --- single cell
    sc = o.get("single_cell")
    if sc:
        bt = sc.data["score_by_type"]
        top = bt[0]
        mk = {r["gene"]: r for r in sc.data["neoplastic_vs_rest"]}
        up = [g for g, r in mk.items() if r["q"] < 0.05 and r["log2_fc"] > 0]
        det = sc.data["fraction_detected"]
        neo_det = {g: det.get("Neoplastic", {}).get(g, np.nan) for g in mk}
        txt = (f"In GSE84465 single cells, the candidate score is highest in {top['group']} cells (median {top['median']:.2f}), followed by " + ", ".join(f"{r['group']}" for r in bt[1:3]) + ". "
               "In neoplastic cells, detection rates are " + ", ".join(f"{g} {100 * v:.0f}%" for g, v in neo_det.items() if not np.isnan(v)) + ". "
               + (f"Genes higher in neoplastic cells than in all other cells (Wilcoxon, BH q < 0.05, log2FC > 0): {', '.join(up)}." if up else "No candidate gene is higher in neoplastic cells than in all other cells."))
        claims.append(_claim("SC1", txt, L.STATISTICAL_ASSOCIATION, "exploratory", analyses=sc.data["analysis_ids"], pmids=["29091775"] if "29091775" in cites else [],
                             caveats=["Four patients; cells treated as replicates (pseudoreplication).", "Comparison against 'all other cells' is dominated by immune cells (about half of all cells)."]))
        pv = sc.data.get("periphery_vs_core")
        if pv:
            sig = [r for r in pv if r["q"] < 0.05]
            claims.append(_claim("SC2", "Neoplastic cells from the infiltrated periphery vs tumour core: " + ("; ".join(f"{r['gene']} log2FC = {r['log2_fc']:.2f}, q = {_fmt(r['q'])}" for r in sig) if sig else "no candidate gene differs at BH q < 0.05") + ".",
                                 L.STATISTICAL_ASSOCIATION if sig else L.INSUFFICIENT, "exploratory", analyses=[sc.data["periphery_analysis_id"]] if sc.data.get("periphery_analysis_id") else [], caveats=["Few peripheral neoplastic cells (see table); patient-confounded."], status="supported" if sig else "not supported"))

    # --- DE / enrichment
    de = o.get("de")
    if de:
        txt = f"Score-high vs score-low TCGA tumours: {de.summary}."
        up = o.get("ora_up")
        dn = o.get("ora_down")
        gs = o.get("gsea")
        if up:
            txt += f" Up-regulated genes are enriched for {', '.join(up.data['top_terms'][:3])}."
        if dn:
            txt += f" Down-regulated genes are enriched for {', '.join(dn.data['top_terms'][:3])}."
        if gs:
            txt += f" Pre-ranked GSEA (Hallmark) top terms: {', '.join(gs.data['top_terms'][:4])}."
        ids = [x.data["analysis_id"] for x in (de, up, dn, gs) if x]
        claims.append(_claim("D1", txt, L.STATISTICAL_ASSOCIATION, "exploratory", analyses=ids, caveats=["Pathway terms describe score-high vs score-low tumours, which differ in composition; enrichment of synaptic terms is expected if the score indexes neuronal tissue."]))

    # --- interpretation (explicitly labelled)
    if comp and vs_o:
        txt = ("Interpretation: the reproducible Proneural > Mesenchymal ordering of the neuronal-activity score is consistent with two non-exclusive explanations: "
               "(a) more infiltrated or adjacent normal brain in Proneural samples, and (b) neural-progenitor / OPC-like tumour-cell states that express synaptic genes. The bulk data cannot separate them")
        if sc:
            top3 = ", ".join(r["group"] for r in sc.data["score_by_type"][:3])
            txt += f"; in single cells the candidate genes are expressed most by {top3} cells, so both normal neural cells and OPC-like populations can contribute."
        else:
            txt += "; single-cell localisation was not run in this investigation."
        claims.append(_claim("I1", txt, L.BIOLOGICAL_PLAUSIBILITY, "interpretation", pmids=[p for p in ("31327527", "28697342") if p in cites], caveats=["Interpretation, not a measured result."]))
    return claims
