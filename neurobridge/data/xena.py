"""TCGA-GBM via the UCSC Xena TCGA hub (public, direct download)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neurobridge.data.cache import cached_file, manifest_entry
from neurobridge.data.cohort import Cohort, empty_clinical

XENA_BASE = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/"
FILES = {
    "tcga_gbm_hiseq": ("TCGA.GBM.sampleMap%2FHiSeqV2.gz", "xena/TCGA.GBM.sampleMap_HiSeqV2.gz"),
    "tcga_gbm_u133a": ("TCGA.GBM.sampleMap%2FHT_HG-U133A.gz", "xena/TCGA.GBM.sampleMap_HT_HG-U133A.gz"),
    "tcga_gbm_clinical": ("TCGA.GBM.sampleMap%2FGBM_clinicalMatrix", "xena/TCGA.GBM.sampleMap_GBM_clinicalMatrix"),
}


def _fetch(key: str):
    fname, rel = FILES[key]
    return cached_file(key, XENA_BASE + fname, rel, source="UCSC Xena TCGA hub")


def load_clinical_raw() -> pd.DataFrame:
    path = _fetch("tcga_gbm_clinical")
    return pd.read_csv(path, sep="\t", index_col=0, low_memory=False)


def harmonise_clinical(raw: pd.DataFrame, sample_ids) -> pd.DataFrame:
    raw = raw.reindex(sample_ids)
    clin = empty_clinical(sample_ids)
    death = pd.to_numeric(raw["days_to_death"], errors="coerce")
    follow = pd.to_numeric(raw["days_to_last_followup"], errors="coerce")
    vital = raw["vital_status"]
    event = vital.map({"DECEASED": 1.0, "LIVING": 0.0})
    time_days = np.where(event == 1.0, death, follow)
    # If death date missing for a deceased patient fall back to last follow-up.
    time_days = np.where(np.isnan(time_days), follow, time_days)
    clin["os_months"] = pd.Series(time_days, index=sample_ids) / 30.4375
    clin["os_event"] = event
    clin.loc[clin["os_months"].isna(), "os_event"] = np.nan
    clin["age"] = pd.to_numeric(raw["age_at_initial_pathologic_diagnosis"], errors="coerce")
    clin["sex"] = raw["gender"].map({"FEMALE": "Female", "MALE": "Male"})
    clin["subtype"] = raw["GeneExp_Subtype"]
    clin["g_cimp"] = raw["G_CIMP_STATUS"].map({"G-CIMP": "G-CIMP", "NON G-CIMP": "non-G-CIMP"})
    clin["sample_type"] = raw["sample_type"].map(
        {"Primary Tumor": "Primary", "Recurrent Tumor": "Recurrent", "Solid Tissue Normal": "Normal"}
    )
    clin["histology"] = raw["histological_type"]
    clin["grade"] = "WHO IV"
    clin["radiotherapy"] = pd.to_numeric(raw.get("CDE_radiation_any"), errors="coerce")
    clin["chemotherapy"] = pd.to_numeric(raw.get("CDE_chemo_tmz"), errors="coerce")
    clin["patient_id"] = raw["_PATIENT"]
    return clin


def load_tcga_gbm_rnaseq(dataset_record) -> Cohort:
    """TCGA-GBM IlluminaHiSeq RNA-seq V2, log2(RSEM+1), as distributed by Xena."""
    path = _fetch("tcga_gbm_hiseq")
    expr = pd.read_csv(path, sep="\t", index_col=0)
    expr.index.name = "gene"
    expr = expr[~expr.index.duplicated(keep="first")]
    clin = harmonise_clinical(load_clinical_raw(), expr.columns)
    prov = {"expression_file": manifest_entry("tcga_gbm_hiseq"), "clinical_file": manifest_entry("tcga_gbm_clinical")}
    return Cohort(
        dataset=dataset_record,
        expression=expr.astype(np.float32),
        clinical=clin,
        expression_scale="log2(RSEM+1) (Xena HiSeqV2 'gene expression RNAseq' track)",
        provenance=prov,
        notes=[
            "Expression values are log2(RSEM normalised_count + 1) exactly as distributed by UCSC Xena; no re-normalisation applied.",
            "Subtype labels (GeneExp_Subtype) are the Verhaak 2010 calls distributed with the TCGA clinical matrix; the 'Neural' subtype was later attributed to normal neural contamination (Wang et al. 2017).",
            "Overall survival is derived from days_to_death (deceased) or days_to_last_followup (living) and converted to months (30.4375 d).",
        ],
    )


def load_tcga_gbm_u133a(dataset_record) -> Cohort:
    """TCGA-GBM Affymetrix HT_HG-U133A array (larger cohort, log2 scale as distributed by Xena)."""
    path = _fetch("tcga_gbm_u133a")
    expr = pd.read_csv(path, sep="\t", index_col=0)
    expr.index.name = "gene"
    expr = expr[~expr.index.duplicated(keep="first")]
    clin = harmonise_clinical(load_clinical_raw(), expr.columns)
    prov = {"expression_file": manifest_entry("tcga_gbm_u133a"), "clinical_file": manifest_entry("tcga_gbm_clinical")}
    return Cohort(
        dataset=dataset_record,
        expression=expr.astype(np.float32),
        clinical=clin,
        expression_scale="log2 RMA-normalised array intensity (Xena HT_HG-U133A track)",
        provenance=prov,
        notes=[
            "Array intensities, not RNA-seq: dynamic range is compressed and low-abundance transcripts are poorly quantified.",
            "This is the platform on which the Verhaak 2010 subtypes were originally defined, so subtype labels and expression are not independent.",
        ],
    )
