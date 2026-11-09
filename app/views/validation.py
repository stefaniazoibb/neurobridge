import json

import numpy as np
import pandas as pd
import streamlit as st

from ui.common import cohort, disclaimer, safe_call, setup, show_fig

setup()
from neurobridge.analysis.sensitivity import survival_sensitivity  # noqa: E402
from neurobridge.analysis.signatures import dichotomise, score_signature  # noqa: E402
from neurobridge.analysis.subtyping import train_and_project  # noqa: E402
from neurobridge.analysis.survival import cox_model, exposure_row, kaplan_meier  # noqa: E402
from neurobridge.analysis.validation import validate_group_difference, validate_signature_survival  # noqa: E402
from neurobridge.analysis.sensitivity import residualise  # noqa: E402
from neurobridge.config import get_settings  # noqa: E402
from neurobridge.data.cache import read_manifest  # noqa: E402
from neurobridge.knowledge.genesets import all_panels  # noqa: E402
from neurobridge.provenance import software_versions, utc_now_iso  # noqa: E402
from neurobridge.viz import figures as F  # noqa: E402

st.markdown("<div class='nb-eyebrow'>Module D · Validation Lab</div>", unsafe_allow_html=True)
st.title("Validation Lab")
st.markdown("<div class='nb-banner'><b>Rules applied here.</b> Exploratory analyses (discovery cohort, any contrast) are labelled as such. "
            "A hypothesis is called <i>replicated</i> only when an <b>independent</b> cohort shows the same direction with p &lt; 0.05. "
            "One cohort is never called validated. Hazard ratios are associations, not causal effects.</div>", unsafe_allow_html=True)
st.write("")

panels = all_panels()
COH = ["tcga_gbm_rnaseq", "cgga_325", "cgga_693"]
loaded = {d: safe_call(cohort, d, what=f"Loading {d}") for d in COH}
loaded = {k: v for k, v in loaded.items() if v is not None}
missing = [d for d in COH if d not in loaded]
if missing:
    st.warning(f"Unavailable cohorts: {missing}. Validation is restricted to the remaining cohorts.")
if "tcga_gbm_rnaseq" not in loaded:
    st.error("The discovery cohort could not be loaded; validation is not possible.")
    st.stop()

tabs = st.tabs(["Survival models", "Cross-cohort validation", "Sensitivity analysis", "Missing data & power", "Reproducibility"])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    did = c1.selectbox("Cohort", list(loaded))
    expo_kind = c2.radio("Exposure", ["gene panel score", "single gene"], horizontal=False)
    co = loaded[did].primary_tumours()
    if expo_kind == "gene panel score":
        pid = c3.selectbox("Panel", list(panels), format_func=lambda k: panels[k].name)
        x = pd.Series(score_signature(co, panels[pid].genes, pid).scores)
        name = pid
    else:
        g = c3.selectbox("Gene", sorted(co.expression.index), index=sorted(co.expression.index).index("NLGN3"))
        x = co.expression.loc[g].astype(float)
        name = g
    split = c4.radio("KM split", ["median", "tertiles"])
    km = kaplan_meier(co.clinical, dichotomise(x, split), f"{name} ({split})", did)
    show_fig(F.kaplan_meier_figure(km), key="km")
    avail = [v for v in ["age", "sex", "subtype", "g_cimp", "idh_status", "mgmt_status"] if v in co.clinical and co.clinical[v].notna().mean() >= 0.5]
    covs = st.multiselect("Adjust Cox model for", avail, default=[v for v in avail if v != "subtype"], help="Variables with >50% missing in this cohort are not offered. Treatment variables are not offered: they are post-diagnosis and partly prognosis-driven.")
    uni = cox_model(co.clinical, x, name, did, [])
    multi = cox_model(co.clinical, x, name, did, covs) if covs else None
    m1, m2 = st.columns(2)
    for col, mod, lab in [(m1, uni, "Univariable"), (m2, multi, "Multivariable")]:
        if mod is None:
            continue
        r = exposure_row(mod, name)
        col.markdown(f"**{lab}** (n = {mod.n}, events = {mod.events}, C-index = {mod.concordance:.3f})")
        col.metric(f"HR per SD of {name}", f"{r.hr:.2f}", help=f"95% CI {r.hr_ci_low:.2f}–{r.hr_ci_high:.2f}")
        col.caption(f"95% CI {r.hr_ci_low:.2f}–{r.hr_ci_high:.2f}; p = {r.p:.3g}. PH assumption: {'not rejected' if mod.ph_assumption_ok else 'questionable' if mod.ph_assumption_ok is False else 'n/a'}.")
        for w in mod.record.warnings + mod.record.limitations:
            col.caption("· " + w)
    if multi:
        show_fig(F.cox_forest_figure(multi), key="coxf")
        st.markdown("**Proportional-hazards test (scaled Schoenfeld residuals, rank time)**")
        st.dataframe(pd.DataFrame(multi.ph_test).T.round(4), width="stretch")
        st.dataframe(pd.DataFrame([r.model_dump() for r in multi.rows]).round(4), hide_index=True, width="stretch")
    st.caption("Exploratory unless the exposure was pre-specified and is tested in an independent cohort (next tab).")

with tabs[1]:
    st.markdown("**Confirmatory tests of pre-specified hypotheses.** Discovery: TCGA. Validation: CGGA-325 and CGGA-693 (independent population, platform and institution).")
    pid = st.selectbox("Signature", ["neuronal_activity_candidates", "neuron_glioma_extended", "kegg_glutamatergic_synapse"], format_func=lambda k: panels[k].name, key="vpid")
    genes = panels[pid].genes
    val_ids = [d for d in ["cgga_325", "cgga_693"] if d in loaded]

    st.markdown("##### H1 · Score differs between Mesenchymal and Proneural tumours")
    excl = sorted(set(panels["neuronal_activity_candidates"].genes) | set(panels["neuron_glioma_extended"].genes) | set(panels["normal_neuron_markers"].genes) | set(panels["kegg_glutamatergic_synapse"].genes))
    proj = train_and_project(loaded["tcga_gbm_rnaseq"], {d: loaded[d] for d in val_ids}, excl) if val_ids else None
    if proj:
        k = st.columns(3)
        k[0].metric("Subtype classifier CV balanced accuracy (TCGA)", f"{proj['cv_balanced_accuracy']:.2f}")
        k[1].metric("Training tumours", proj["n_train"])
        k[2].metric("Excluded genes (no circularity)", len(excl))
        with st.expander("Classifier details"):
            st.dataframe(pd.DataFrame(proj["cv_confusion"]["matrix"], index=[f"true {l}" for l in proj["cv_confusion"]["labels"]], columns=[f"pred {l}" for l in proj["cv_confusion"]["labels"]]), width="stretch")
            for l in proj["record"].limitations:
                st.caption("· " + l)
    resid = st.toggle("Residualise the score on mature-neuron markers (composition control)", value=False)
    ctrl = panels["normal_neuron_markers"].genes
    cs = {}
    for d in ["tcga_gbm_rnaseq"] + val_ids:
        cc = loaded[d].primary_tumours()
        s = pd.Series(score_signature(cc, genes, pid).scores)
        if resid:
            s = residualise(s, pd.Series(score_signature(cc, ctrl, "n").scores))
        if d == "tcga_gbm_rnaseq":
            labs, src, m = cc.clinical["subtype"], "original labels", cc.clinical["g_cimp"].eq("non-G-CIMP")
        else:
            labs, src, m = pd.Series(proj["projections"][d]["labels"]).reindex(cc.samples), "projected labels", cc.clinical["idh_status"].eq("Wildtype")
        keep = m.reindex(s.index).fillna(False).astype(bool)
        cs[d] = (s[keep], labs.reindex(s.index)[keep], src)
    vs = validate_group_difference(cs, "tcga_gbm_rnaseq", "Mesenchymal", "Proneural", pid + (" (residualised)" if resid else ""))
    show_fig(F.validation_forest_figure(vs), key="vsub")
    st.markdown(f"**Verdict:** {vs.verdict}")
    for r in vs.verdict_rationale:
        st.caption("· " + r)

    st.markdown("##### H2 · Score is associated with overall survival (adjusted Cox)")
    vss, _ = validate_signature_survival({d: loaded[d] for d in ["tcga_gbm_rnaseq"] + val_ids}, "tcga_gbm_rnaseq", genes, pid, ["age", "sex", "g_cimp", "idh_status", "mgmt_status"])
    show_fig(F.validation_forest_figure(vss), key="vsurv")
    st.markdown(f"**Verdict:** {vss.verdict}")
    for r in vss.verdict_rationale:
        st.caption("· " + r)
    st.dataframe(pd.DataFrame([r.model_dump() for r in vss.per_cohort]), hide_index=True, width="stretch")

with tabs[2]:
    pid = st.selectbox("Signature", ["neuronal_activity_candidates", "neuron_glioma_extended"], format_func=lambda k: panels[k].name, key="spid")
    rows = []
    for d in loaded:
        rows += survival_sensitivity(loaded[d], panels[pid].genes, pid, panels["normal_neuron_markers"].genes, ["age", "sex", "g_cimp", "idh_status", "mgmt_status"])
    df = pd.DataFrame(rows)
    fr = [{"label": f"{r['cohort']}: {r['specification']}", "est": r["hr"], "lo": r["ci_low"], "hi": r["ci_high"], "p": r["p"], "color": F.COHORT_COLORS.get(r["cohort"])} for r in rows if not np.isnan(r["hr"])]
    show_fig(F.forest_figure(fr, "Survival association under alternative specifications", "Cox HR per SD of the score. Specifications are robustness checks; they are not corrected for multiplicity and do not constitute new hypotheses."), key="sens")
    st.dataframe(df.round(4), hide_index=True, width="stretch")

with tabs[3]:
    st.markdown("**Complete-case availability in primary tumours**")
    rows = []
    for d, co in loaded.items():
        p = co.clinical[co.clinical["sample_type"] == "Primary"]
        for v in ["os_months", "age", "sex", "subtype", "idh_status", "g_cimp", "mgmt_status"]:
            rows.append({"cohort": d, "variable": v, "n_available": int(p[v].notna().sum()) if v in p else 0, "pct": round(100 * p[v].notna().mean(), 1) if v in p else 0.0})
    mm = pd.DataFrame(rows).pivot(index="variable", columns="cohort", values="pct")
    show_fig(F.heatmap_figure(mm, "Variable completeness (% of primary tumours)", "0% = variable not distributed with the cohort. Models adjust only for variables available in each cohort, so adjustment sets differ across cohorts.", zmid=None, colorscale=[[i / 6, c] for i, c in enumerate(F.SEQUENTIAL_BLUE)], zlabel="%", text_fmt=".0f"), key="miss")
    st.markdown("**Approximate power for the survival test** (Schoenfeld formula, two-sided alpha 0.05, HR per SD)")
    from scipy import stats as _st

    hr = st.slider("Assumed true HR per SD", 1.05, 1.6, 1.25, 0.05)
    pw = []
    for d, co in loaded.items():
        p = co.clinical[(co.clinical["sample_type"] == "Primary")]
        ev = int(p["os_event"].fillna(0).sum())
        z = np.sqrt(ev) * np.log(hr) - _st.norm.ppf(0.975)
        pw.append({"cohort": d, "events (primary)": ev, "power": round(float(_st.norm.cdf(z)), 2)})
    st.dataframe(pd.DataFrame(pw), hide_index=True, width="stretch")
    st.caption("Power for a standardised continuous exposure ≈ Φ(√D·|log HR| − z₀.₉₇₅), with D events. Batch effects: TCGA and CGGA differ in platform (HiSeq 2000 vs 2000/2500 vs 4000); scores are standardised within each cohort and cohorts are never pooled at the sample level.")

with tabs[4]:
    st.markdown("**Reproducibility panel**")
    s = get_settings()
    c1, c2 = st.columns(2)
    c1.markdown(f"- Random seed: `{s.seed}`\n- Generated: `{utc_now_iso()}`\n- Network allowed: `{s.allow_network}`\n- LLM provider: `{s.llm_provider}`")
    c2.markdown("**Software versions**")
    c2.dataframe(pd.DataFrame([{"package": k, "version": v} for k, v in software_versions().items()]), hide_index=True, width="stretch")
    st.markdown("**Dataset versions and input files** (SHA-256 of every downloaded file)")
    st.dataframe(pd.DataFrame([{"key": k, **v} for k, v in read_manifest().items()]), hide_index=True, width="stretch")
    if "inv" in st.session_state:
        res, ws = st.session_state["inv"]
        st.markdown(f"**Analysis records of investigation `{res.investigation_id}`** (method, parameters, seed, timestamps, inputs and outputs)")
        recs = []
        for ref, obj in ws.artefacts.items():
            r = getattr(obj, "record", None)
            if r is None and isinstance(obj, dict) and "record" in obj:
                r = obj["record"]
                r = r if isinstance(r, dict) else r.model_dump(mode="json")
            elif r is not None:
                r = r.model_dump(mode="json")
            if r:
                recs.append({"ref": ref, "analysis_id": r["analysis_id"], "kind": r["kind"], "datasets": ", ".join(r["dataset_ids"]), "method": r["method"], "seed": r["seed"], "started": r["started_at"], "finished": r["finished_at"], "n_used": r.get("n_used"), "parameters": json.dumps(r["parameters"], default=str)[:300]})
        st.dataframe(pd.DataFrame(recs), hide_index=True, width="stretch")
    else:
        st.caption("Run an investigation in the Research Copilot to see per-analysis provenance records here.")
disclaimer()
