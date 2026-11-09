import pandas as pd
import streamlit as st

from ui.common import catalog_frame, cohort, disclaimer, safe_call, setup

setup()
from neurobridge.data.cache import read_manifest  # noqa: E402
from neurobridge.data.catalog import INGESTIBLE  # noqa: E402

st.markdown("<div class='nb-eyebrow'>Module B</div>", unsafe_allow_html=True)
st.title("Dataset Explorer")
st.caption("Public glioblastoma resources known to NeuroBridge. Metadata below were checked against each source; sample counts in the analysis pages are recomputed from the downloaded files. Datasets that are listed but not ingested say why.")

cat = catalog_frame()
f1, f2, f3 = st.columns([2, 2, 1.3])
mods = sorted({m for ms in cat["modalities"] for m in ms})
msel = f1.multiselect("Modality", mods)
access = f2.multiselect("Access / status", sorted(cat["ingestion"].unique()))
surv = f3.toggle("Has survival", value=False)
view = cat.copy()
if msel:
    view = view[view["modalities"].map(lambda ms: any(m in ms for m in msel))]
if access:
    view = view[view["ingestion"].isin(access)]
if surv:
    view = view[view["survival_available"]]

table = view[["dataset_id", "name", "source", "publication", "n_samples", "n_cells", "survival_available", "ingestion", "role"]].copy()
st.dataframe(table, hide_index=True, width="stretch", column_config={"publication": st.column_config.TextColumn("PMID"), "survival_available": st.column_config.CheckboxColumn("survival")})

st.subheader("Dataset details")
pick = st.selectbox("Dataset", view["dataset_id"].tolist())
if pick:
    d = cat.set_index("dataset_id").loc[pick]
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown(f"### {d['name']}")
        st.markdown(f"**Source** [{d['source']}]({d['source_url']})  \n**Publication** {d['publication_title'] or '-'}" + (f" ([PMID {d['publication']}](https://pubmed.ncbi.nlm.nih.gov/{d['publication']}/))" if d["publication"] else ""))
        st.markdown(f"**Modalities** {', '.join(d['modalities'])}  \n**Platform** {d['platform'] or '-'}  \n**Units** {d['expression_units'] or '-'}  \n**Version** {d['version'] or '-'}")
        st.markdown("**Clinical variables** " + (", ".join(d["clinical_variables"]) or "-"))
        st.markdown("**Preprocessing**")
        for x in d["preprocessing"] or ["-"]:
            st.markdown(f"- {x}")
    with c2:
        st.markdown(f"**Access** {d['access']}  \n**Status** {d['ingestion']}  \n**Role** {d['role']}  \n**Demonstration data** {'yes' if d['is_demo'] else 'no (real public data)'}")
        st.markdown("**Known limitations**")
        for x in d["limitations"] or ["-"]:
            st.markdown(f"- {x}")
        if d["files"]:
            with st.expander("Download URLs"):
                for u in d["files"]:
                    st.code(u, language=None)

    if pick in INGESTIBLE and pick != "gse84465_darmanis":
        if st.button("Load and summarise this cohort", icon=":material/download:"):
            st.session_state[f"loaded_{pick}"] = True
        if st.session_state.get(f"loaded_{pick}"):
            c = safe_call(cohort, pick, what=f"Loading {pick}")
            if c is not None:
                s = c.summary()
                k = st.columns(4)
                k[0].metric("samples", s["n_samples"])
                k[1].metric("primary tumours", s["n_primary"])
                k[2].metric("with survival", s["n_with_survival"])
                k[3].metric("deaths", s["n_events"])
                clin = c.clinical
                prim = clin[clin["sample_type"] == "Primary"]
                comp = pd.DataFrame({"available (primary)": prim.notna().sum(), "complete %": (100 * prim.notna().mean()).round(1)})
                comp = comp.loc[[x for x in ["os_months", "os_event", "age", "sex", "subtype", "idh_status", "g_cimp", "mgmt_status", "radiotherapy", "chemotherapy"] if x in comp.index]]
                cc1, cc2 = st.columns(2)
                cc1.markdown("**Clinical completeness**")
                cc1.dataframe(comp, width="stretch")
                cc2.markdown("**Group sizes (primary tumours)**")
                for col in ["subtype", "idh_status", "g_cimp", "mgmt_status", "sex"]:
                    if col in prim and prim[col].notna().any():
                        cc2.markdown(f"*{col}*: " + ", ".join(f"{a} {b}" for a, b in prim[col].value_counts().items()))
                st.markdown("**Preprocessing notes**")
                for n in c.notes:
                    st.markdown(f"- {n}")
                st.markdown("**File provenance**")
                st.json(c.provenance, expanded=False)
    elif pick == "gse84465_darmanis":
        st.info("The single-cell dataset is processed in the Biological Explorer (QC, normalisation, UMAP, cell-type analyses).")

st.divider()
st.subheader("Live listings")
st.caption("Queried from public APIs at request time. These are listings: large atlases (1-7 GB) are not downloaded automatically.")
cA, cB = st.columns(2)
if cA.button("Query cBioPortal glioblastoma studies", icon=":material/travel_explore:"):
    from neurobridge.data import cbioportal

    res = safe_call(cbioportal.search_studies, "glioblastoma", what="cBioPortal query")
    if res is not None:
        cA.dataframe(pd.DataFrame([{"studyId": s["studyId"], "name": s["name"], "samples": s.get("allSampleCount")} for s in res]), hide_index=True, width="stretch")
if cB.button("Query CELLxGENE glioma collections", icon=":material/travel_explore:"):
    from neurobridge.data import cellxgene

    res = safe_call(cellxgene.list_glioma_collections, what="CELLxGENE query")
    if res is not None:
        cB.dataframe(pd.DataFrame(res), hide_index=True, width="stretch", column_config={"url": st.column_config.LinkColumn()})
        st.session_state["cxg"] = res
if st.session_state.get("cxg"):
    col = st.selectbox("Inspect a CELLxGENE collection", [c["name"] for c in st.session_state["cxg"]])
    cid = next(c["collection_id"] for c in st.session_state["cxg"] if c["name"] == col)
    from neurobridge.data import cellxgene

    ds = safe_call(cellxgene.collection_datasets, cid, what="CELLxGENE collection query")
    if ds:
        df = pd.DataFrame(ds)
        df["h5ad_size_GB"] = (df["h5ad_size_bytes"] / 1e9).round(2)
        st.dataframe(df.drop(columns=["h5ad_size_bytes"]), hide_index=True, width="stretch")

st.divider()
st.subheader("Local data cache")
man = read_manifest()
if man:
    st.dataframe(pd.DataFrame([{"key": k, **v} for k, v in man.items()]), hide_index=True, width="stretch")
else:
    st.info("Nothing cached yet.")
disclaimer()
