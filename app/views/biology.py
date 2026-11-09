import numpy as np
import pandas as pd
import streamlit as st

from ui.common import cohort, disclaimer, processed_single_cell, safe_call, setup, show_fig

setup()
from neurobridge.analysis.correlation import gene_correlation  # noqa: E402
from neurobridge.analysis.differential import differential_expression  # noqa: E402
from neurobridge.analysis.enrichment import over_representation, prerank_gsea  # noqa: E402
from neurobridge.analysis.groups import compare_groups  # noqa: E402
from neurobridge.analysis.signatures import dichotomise, score_signature  # noqa: E402
from neurobridge.knowledge.genesets import LIBRARIES, all_panels  # noqa: E402
from neurobridge.viz import figures as F  # noqa: E402

st.markdown("<div class='nb-eyebrow'>Module C</div>", unsafe_allow_html=True)
st.title("Biological Explorer")
st.caption("Interactive analyses on real public data. Candidate genes are investigation targets, not assumed biomarkers. All bulk analyses use primary tumours. Every figure states its method under the plot.")

panels = all_panels()
BULK = {"TCGA-GBM RNA-seq (discovery)": "tcga_gbm_rnaseq", "CGGA-325 GBM": "cgga_325", "CGGA-693 GBM": "cgga_693"}
with st.sidebar:
    st.markdown("**Bulk cohort**")
    cname = st.selectbox("Cohort", list(BULK), label_visibility="collapsed")
    did = BULK[cname]
    panel_id = st.selectbox("Gene panel", list(panels), format_func=lambda k: panels[k].name)
c = safe_call(cohort, did, what="Loading cohort")
if c is None:
    st.stop()
c = c.primary_tumours()
panel = panels[panel_id]
with st.expander(f"Panel: {panel.name} ({len(panel.genes)} genes · origin: {panel.origin})"):
    st.write(panel.description)
    st.write(", ".join(panel.genes[:60]) + (" …" if len(panel.genes) > 60 else ""))
    if panel.gene_rationale:
        st.dataframe(pd.DataFrame([{"gene": g, "rationale": r} for g, r in panel.gene_rationale.items()]), hide_index=True, width="stretch")

group_options = [g for g in ["subtype", "idh_status", "g_cimp", "mgmt_status", "sex"] if g in c.clinical and c.clinical[g].notna().sum() > 10]
if "subtype" not in group_options:
    st.caption("This cohort has no distributed subtype labels. Projected labels are produced in the Validation Lab (cross-cohort subtype replication).")

tabs = st.tabs(["Gene expression", "Molecular subgroups", "Correlation", "Differential expression & pathways", "Single-cell", "Cell–cell communication", "Spatial"])

with tabs[0]:
    genes_all = sorted(c.expression.index)
    default = [g for g in panels["neuronal_activity_candidates"].genes if g in c.expression.index]
    gsel = st.multiselect("Genes", genes_all, default=default[:6], max_selections=12)
    grp = st.selectbox("Group by", group_options, key="g0") if group_options else None
    if gsel and grp:
        cols = st.columns(2)
        for i, g in enumerate(gsel):
            v = c.expression.loc[g]
            res = compare_groups(v, c.clinical[grp], g, grp, did)
            cap = f"log2 expression of {g} in primary tumours of {did} ({c.expression_scale}). Kruskal-Wallis p = {res.omnibus_p:.2e}, epsilon-squared = {res.effect_size:.3f}. Exploratory; no correction across genes shown."
            with cols[i % 2]:
                show_fig(F.group_box_figure(v, c.clinical[grp], f"{g} log2 expression", f"{g} by {grp}", cap), key=f"ge_{g}_{grp}")
        st.markdown("**Expression heatmap (mean z by group)**")
        if "subtype" in group_options:
            show_fig(F.subtype_gene_heatmap(c, gsel), key="hm_sel")

with tabs[1]:
    if not group_options:
        st.info("No grouping variable with enough labelled samples in this cohort.")
    else:
        grp = st.selectbox("Grouping", group_options, key="g1")
        idhwt = st.toggle("IDH-wildtype / non-G-CIMP only", value=False)
        sres = score_signature(c, panel.genes, panel_id)
        s = pd.Series(sres.scores)
        if idhwt:
            m = c.clinical["idh_status"].eq("Wildtype") if c.clinical["idh_status"].notna().any() else c.clinical["g_cimp"].eq("non-G-CIMP")
            s = s[m.reindex(s.index).fillna(False).astype(bool)]
        res = compare_groups(s, c.clinical[grp], f"{panel_id} score", grp, did)
        cap = f"Mean z-score of {len(sres.genes_found)}/{len(sres.genes_requested)} panel genes. Kruskal-Wallis p = {res.omnibus_p:.2e}, epsilon-squared = {res.effect_size:.3f}. Pairwise Mann-Whitney with BH correction below. Exploratory."
        show_fig(F.group_box_figure(s, c.clinical[grp], "Signature score (mean z)", f"{panel.name}: score by {grp}", cap), key="sg")
        st.dataframe(pd.DataFrame(res.pairwise).round(4), hide_index=True, width="stretch")
        if sres.genes_missing:
            st.warning("Genes absent from this matrix: " + ", ".join(sres.genes_missing))

with tabs[2]:
    gl = [g for g in panel.genes if g in c.expression.index][:30]
    meth = st.radio("Method", ["spearman", "pearson"], horizontal=True)
    cr = gene_correlation(c, gl, meth)
    show_fig(F.correlation_figure(cr), key="corr")
    q = pd.DataFrame(cr.q_values, index=cr.genes, columns=cr.genes)
    with st.expander("BH-adjusted p-values"):
        st.dataframe(q.map(lambda v: f"{v:.1e}"), width="stretch")

with tabs[3]:
    a1, a2, a3 = st.columns(3)
    split = a1.radio("Split", ["median", "tertiles"], horizontal=True, help="Tertiles compare top vs bottom third (middle dropped).")
    test = a2.radio("Test", ["welch", "mannwhitney"], horizontal=True)
    fc = a3.slider("|log2FC| threshold for colouring", 0.0, 2.0, 0.5, 0.1)
    s = pd.Series(score_signature(c, panel.genes, panel_id).scores)
    de = differential_expression(c, dichotomise(s, split), "high", "low", test=test)
    show_fig(F.volcano_figure(de, highlight=panel.genes[:12], fc_thr=fc), key="volc")
    df = de.to_frame()
    st.dataframe(df.head(200).round(5), hide_index=True, width="stretch")
    st.download_button("Download full DE table (CSV)", df.to_csv(index=False), f"de_{did}_{panel_id}.csv", "text/csv")
    st.warning("Score-defined groups: genes in the panel are differential by construction, and score-high tumours differ in normal-brain content. Read pathways as a description of the contrast, not as tumour-intrinsic programmes.", icon=":material/info:")
    e1, e2, e3 = st.columns(3)
    lib = e1.selectbox("Library", list(LIBRARIES), format_func=lambda k: LIBRARIES[k])
    meth = e2.radio("Enrichment", ["ORA (up)", "ORA (down)", "pre-ranked GSEA"], horizontal=False)
    if e3.button("Run enrichment", type="primary"):
        if meth.startswith("ORA"):
            up = meth == "ORA (up)"
            sel = df[(df.q_value < 0.05) & ((df.log2_fc > fc) if up else (df.log2_fc < -fc))].gene.tolist()
            if len(sel) < 5:
                st.info("Fewer than five genes pass the thresholds; enrichment not run.")
            else:
                enr = safe_call(over_representation, sel, lib, df.gene.tolist(), did, what="Enrichment")
                if enr:
                    show_fig(F.enrichment_figure(enr), key="enr")
                    st.dataframe(pd.DataFrame(enr.terms), hide_index=True, width="stretch")
        else:
            with st.spinner("Running pre-ranked GSEA (1,000 permutations)…"):
                enr = safe_call(prerank_gsea, df.set_index("gene")["hedges_g"], lib, did, permutations=1000, what="GSEA")
            if enr:
                show_fig(F.enrichment_figure(enr), key="gsea")
                st.dataframe(pd.DataFrame(enr.terms), hide_index=True, width="stretch")

with tabs[4]:
    st.caption("GSE84465 (Darmanis et al. 2017): 3,589 Smart-seq2 cells from four patients, tumour core and infiltrated periphery, with author cell-type annotations.")
    if st.toggle("Load and process the single-cell dataset", value=st.session_state.get("sc_on", False), key="sc_on"):
        out = safe_call(processed_single_cell, what="Single-cell processing")
        if out:
            from neurobridge.analysis import singlecell as scm

            a, rec = out
            st.caption(f"Kept {rec.n_used} cells after QC ({rec.n_excluded} removed). Parameters: {rec.parameters}. Seed {rec.seed}.")
            colour = st.radio("Colour UMAP by", ["cell_type_author", "patient", "location", "leiden", "gene"], horizontal=True)
            if colour == "gene":
                g = st.selectbox("Gene", sorted(a.raw.var_names), index=sorted(a.raw.var_names).index("GRIA2"))
                obs = a.obs.copy()
                x = a.raw[:, g].X
                obs[g] = np.asarray(x.toarray() if hasattr(x, "toarray") else x).ravel()
                show_fig(F.umap_figure(obs, a.obsm["X_umap"], g, f"{g} expression (log1p CP10k)", "Per-cell log-normalised expression on the UMAP embedding.", continuous=True), key="umg")
            else:
                show_fig(F.umap_figure(a.obs, a.obsm["X_umap"], colour, f"UMAP coloured by {colour}", f"scanpy UMAP of {a.n_obs} cells; colour = {colour}."), key=f"um_{colour}")
            sc_genes = st.multiselect("Genes for cell-type comparison", sorted(a.raw.var_names), default=[g for g in panels["neuron_glioma_extended"].genes if g in a.raw.var_names])
            if sc_genes:
                tab, rec_ct = scm.cell_type_expression(a, sc_genes)
                show_fig(F.dotplot_figure(tab, "Expression by cell type", "Dot size: fraction of cells with >0 counts; colour: mean log1p(CP10k). Author annotations."), key="dot")
                ref = st.selectbox("Neoplastic cells compared against", ["rest", "OPC", "Astrocyte", "Neuron", "Oligodendrocyte"])
                mk, rec_mk = scm.marker_test(a, "cell_type_author", "Neoplastic", ref, genes=sc_genes)
                st.dataframe(mk.round(5), hide_index=True, width="stretch")
                st.caption(" ".join(rec_mk.limitations))

with tabs[5]:
    st.caption("Ligand–receptor co-expression between annotated cell types in GSE84465, for a curated set of neuron–glioma pairs. Score = mean ligand in sender × mean receptor in receiver; significance from a cell-label permutation null.")
    if st.session_state.get("sc_on"):
        out = processed_single_cell()
        a, _ = out
        from neurobridge.analysis.communication import ligand_receptor_scores

        nperm = st.select_slider("Permutations", [200, 500, 1000], value=500)
        df, rec = ligand_receptor_scores(a, n_perm=nperm)
        recv = st.selectbox("Receiver cell type", sorted(df["receiver"].unique()), index=sorted(df["receiver"].unique()).index("Neoplastic"))
        sub = df[df.receiver == recv]
        mat = sub.pivot_table(index=sub["ligand"] + " → " + sub["receptor"], columns="sender", values="score")
        show_fig(F.heatmap_figure(mat, f"Ligand–receptor scores with {recv} cells as receivers", f"{rec.method}. {nperm} permutations (minimum attainable p = {1 / (nperm + 1):.4f}). Exploratory.", zmid=None, colorscale=[[i / 6, col] for i, col in enumerate(F.SEQUENTIAL_BLUE)], zlabel="score", text_fmt=".2f"), key="lr")
        st.dataframe(sub.round(4), hide_index=True, width="stretch")
        for l in rec.limitations:
            st.caption("· " + l)
    else:
        st.info("Load the single-cell dataset in the Single-cell tab first.")

with tabs[6]:
    st.caption("10x Genomics Visium, Human Glioblastoma (whole transcriptome, Space Ranger 1.2.0, CC BY 4.0): one fresh-frozen section from one patient, ~55 µm spots. Every result here describes this tissue and is exploratory.")
    if st.toggle("Load and process the Visium section", value=st.session_state.get("sp_on", False), key="sp_on"):
        from ui.common import processed_spatial

        out = safe_call(processed_spatial, what="Spatial processing")
        if out:
            from neurobridge.analysis import spatial as spa

            a, rec, dom, nbrs, img = out
            st.caption(f"{rec.n_used} spots after QC ({rec.n_excluded} removed). Domains: {dom.shape[0]} (Leiden, resolution {rec.parameters['leiden_resolution']}). Seed {rec.seed}.")
            v1, v2 = st.columns([1, 1])
            what = v1.radio("Map", ["domains", "programme", "gene"], horizontal=True)
            if what == "domains":
                show_fig(F.spatial_figure(a, a.obs["domain_programme"].values, "Spatial domains by dominant programme", "Leiden domains on spot expression, coloured by their most enriched programme. Proxies, not cell-type calls.", continuous=False, image=img), key="spd")
                st.dataframe(dom.round(3), width="stretch")
            elif what == "programme":
                prog = v2.selectbox("Programme", [c.replace("prog_", "") for c in a.obs.columns if c.startswith("prog_")])
                show_fig(F.spatial_figure(a, a.obs[f"prog_{prog}"].values, f"{prog} programme per spot", "scanpy score_genes: " + ", ".join(spa.PROGRAMMES[prog]), image=img), key="spp")
            else:
                g = v2.selectbox("Gene", sorted(a.raw.var_names), index=sorted(a.raw.var_names).index("GRIA2"))
                x = spa._gene(a, g)
                show_fig(F.spatial_figure(a, x, f"{g} expression per spot", f"log1p(CP10k); detected in {100 * (x > 0).mean():.0f}% of spots.", image=img), key="spg")
            sp_genes = st.multiselect("Genes for spatial statistics", sorted(a.raw.var_names), default=[g for g in panels["neuron_glioma_extended"].genes if g in a.raw.var_names][:10], key="spgenes")
            nperm = st.select_slider("Permutations", [199, 499, 999], value=499, key="spperm")
            if sp_genes and st.button("Run spatial statistics", type="primary"):
                res, rec_p = spa.spatial_programme_analysis(a, sp_genes, n_perm=nperm)
                st.markdown("**Spot correlations with programmes and spatial autocorrelation (Moran's I)**")
                st.dataframe(res["correlations"].round(4), hide_index=True, width="stretch")
                itab, _, rec_i = spa.interface_analysis(a, sp_genes, nbrs)
                st.markdown(f"**Gradient from neuron-rich tissue** ({rec_i.parameters['n_neuron_rich']} neuron-rich spots; interface = 1–2 rings, distal ≥ 5 rings)")
                st.dataframe(itab.round(4), hide_index=True, width="stretch")
                lr, rec_lr = spa.neighbour_lr(a, nbrs, n_perm=nperm)
                st.markdown("**Ligand–receptor neighbour co-localisation** (global and domain-stratified permutation nulls)")
                st.dataframe(lr.round(4), hide_index=True, width="stretch")
                for l in dict.fromkeys(rec_p.limitations + rec_i.limitations + rec_lr.limitations):
                    st.caption("· " + l)
    st.caption("Next step: a multi-patient Visium cohort (Ravi et al. 2022, PMID 35700707) would turn these single-section descriptions into testable across-patient effects.")
disclaimer()
