"""Chinese Glioma Genome Atlas (CGGA) mRNAseq_325 and mRNAseq_693 cohorts (public, direct download).

The two batches were sequenced on different platforms (Illumina HiSeq 2000/2500 for the 325 batch,
HiSeq 4000 for the 693 batch) and are distributed as separate matrices, so they are treated as two
independent validation cohorts. Values are RSEM gene-level estimates as distributed; we apply log2(x+1).
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from neurobridge.data.cache import cached_file, manifest_entry
from neurobridge.data.cohort import Cohort, empty_clinical

CGGA_BASE = "http://www.cgga.org.cn/download?file=download/20200506/{fname}&type={typ}&time=20200506"

BATCHES = {
    "cgga_325": {
        "expr": ("CGGA.mRNAseq_325.RSEM-genes.20200506.txt.zip", "mRNAseq_325"),
        "clin": ("CGGA.mRNAseq_325_clinical.20200506.txt.zip", "mRNAseq_325_clinical"),
        "platform": "Illumina HiSeq 2000/2500",
    },
    "cgga_693": {
        "expr": ("CGGA.mRNAseq_693.RSEM-genes.20200506.txt.zip", "mRNAseq_693"),
        "clin": ("CGGA.mRNAseq_693_clinical.20200506.txt.zip", "mRNAseq_693_clinical"),
        "platform": "Illumina HiSeq 4000",
    },
}


def _fetch(batch: str, kind: str):
    fname, typ = BATCHES[batch][kind]
    url = CGGA_BASE.format(fname=fname, typ=typ)
    return cached_file(f"{batch}_{kind}", url, f"cgga/{fname}", source="CGGA (cgga.org.cn)", min_bytes=2000)


def _read_zip_table(path, index_col=None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        name = [n for n in zf.namelist() if n.endswith(".txt")][0]
        raw = zf.read(name).decode("utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")  # CGGA clinical files use CR line endings
    return pd.read_csv(io.StringIO(raw), sep="\t", index_col=index_col, na_values=["NA"])


def harmonise_clinical(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.set_index("CGGA_ID")
    # Strip stray whitespace in categorical fields (e.g. "Female " occurs in the 693 clinical file).
    for col in raw.columns:
        if raw[col].dtype == object:
            raw[col] = raw[col].astype(str).str.strip().replace({"nan": None, "NA": None})
    clin = empty_clinical(raw.index)
    censor_col = [c for c in raw.columns if c.startswith("Censor")][0]
    radio_col = [c for c in raw.columns if c.startswith("Radio_status")][0]
    chemo_col = [c for c in raw.columns if c.startswith("Chemo_status")][0]
    clin["os_months"] = pd.to_numeric(raw["OS"], errors="coerce") / 30.4375
    clin["os_event"] = pd.to_numeric(raw[censor_col], errors="coerce")
    clin["age"] = pd.to_numeric(raw["Age"], errors="coerce")
    clin["sex"] = raw["Gender"]
    clin["idh_status"] = raw["IDH_mutation_status"]
    clin["mgmt_status"] = raw["MGMTp_methylation_status"].map({"methylated": "methylated", "un-methylated": "unmethylated"})
    clin["histology"] = raw["Histology"]
    clin["grade"] = raw["Grade"]
    clin["sample_type"] = raw["PRS_type"].map({"Primary": "Primary", "Recurrent": "Recurrent", "Secondary": "Secondary"})
    clin["radiotherapy"] = pd.to_numeric(raw[radio_col], errors="coerce")
    clin["chemotherapy"] = pd.to_numeric(raw[chemo_col], errors="coerce")
    clin["codeletion_1p19q"] = raw["1p19q_codeletion_status"]
    clin["patient_id"] = raw.index
    return clin


def load_cgga(batch: str, dataset_record, *, gbm_only: bool = True) -> Cohort:
    expr_path = _fetch(batch, "expr")
    clin_path = _fetch(batch, "clin")
    expr = _read_zip_table(expr_path, index_col=0)
    expr.index.name = "gene"
    expr = expr[~expr.index.duplicated(keep="first")]
    expr = np.log2(expr.clip(lower=0) + 1.0)
    clin = harmonise_clinical(_read_zip_table(clin_path))
    common = expr.columns.intersection(clin.index)
    expr = expr.loc[:, common]
    clin = clin.loc[common]
    notes = [
        f"RSEM gene-level estimates as distributed by CGGA (platform: {BATCHES[batch]['platform']}); log2(x+1) applied by NeuroBridge.",
        "The CGGA cohorts contain all glioma grades; GBM (WHO IV, histology 'GBM'/'rGBM'/'sGBM') is selected for validation analyses.",
        "No transcriptional (Verhaak) subtype labels are distributed with CGGA; IDH and MGMT status are available.",
    ]
    if gbm_only:
        keep = clin.index[clin["histology"].isin(["GBM", "rGBM", "sGBM"])]
        expr = expr.loc[:, keep]
        clin = clin.loc[keep]
    return Cohort(
        dataset=dataset_record,
        expression=expr.astype(np.float32),
        clinical=clin,
        expression_scale="log2(RSEM+1)",
        provenance={"expression_file": manifest_entry(f"{batch}_expr"), "clinical_file": manifest_entry(f"{batch}_clin")},
        notes=notes,
    )
