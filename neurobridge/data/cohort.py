"""A harmonised bulk-expression cohort: expression matrix + clinical table + provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from neurobridge.schemas.datasets import DatasetRecord

# Harmonised clinical column names used across every cohort.
CLINICAL_COLUMNS = {
    "os_months": "Overall survival time (months)",
    "os_event": "Death observed (1) or censored (0)",
    "age": "Age at diagnosis (years)",
    "sex": "Sex (Female / Male)",
    "subtype": "Transcriptional subtype (Verhaak 2010 labels where available)",
    "idh_status": "IDH mutation status (Mutant / Wildtype)",
    "g_cimp": "G-CIMP methylation status",
    "mgmt_status": "MGMT promoter methylation (methylated / unmethylated)",
    "sample_type": "Primary / Recurrent / Normal",
    "histology": "Histological diagnosis",
    "grade": "WHO grade",
    "radiotherapy": "Radiotherapy received (1/0)",
    "chemotherapy": "Temozolomide / alkylating chemotherapy received (1/0)",
}


@dataclass
class Cohort:
    dataset: DatasetRecord
    expression: pd.DataFrame  # genes x samples, log2(x+1) scale
    clinical: pd.DataFrame  # samples x harmonised clinical columns
    expression_scale: str = "log2(x+1)"
    provenance: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def samples(self) -> pd.Index:
        return self.expression.columns

    @property
    def genes(self) -> pd.Index:
        return self.expression.index

    def subset(self, sample_ids) -> "Cohort":
        ids = [s for s in sample_ids if s in self.expression.columns]
        return Cohort(
            dataset=self.dataset,
            expression=self.expression.loc[:, ids],
            clinical=self.clinical.loc[ids],
            expression_scale=self.expression_scale,
            provenance=dict(self.provenance),
            notes=list(self.notes),
        )

    def copy(self) -> Cohort:
        """Investigation-private copy. Clinical data, provenance and notes are copied (investigations add columns
        such as scores and projected subtypes). The expression matrix is shared, not copied, to keep memory
        bounded with several concurrent users: NeuroBridge treats it as read-only, and
        tests/test_isolation.py checks that a run leaves the cached object unchanged."""
        return Cohort(
            dataset=self.dataset,
            expression=self.expression,
            clinical=self.clinical.copy(deep=True),
            expression_scale=self.expression_scale,
            provenance=dict(self.provenance),
            notes=list(self.notes),
        )

    def primary_tumours(self) -> "Cohort":
        mask = self.clinical["sample_type"].eq("Primary")
        return self.subset(self.clinical.index[mask])

    def gene_vector(self, gene: str) -> Optional[pd.Series]:
        if gene in self.expression.index:
            return self.expression.loc[gene]
        return None

    def summary(self) -> dict:
        c = self.clinical
        return {
            "dataset_id": self.dataset.dataset_id,
            "n_samples": int(self.expression.shape[1]),
            "n_genes": int(self.expression.shape[0]),
            "n_primary": int(c["sample_type"].eq("Primary").sum()) if "sample_type" in c else None,
            "n_with_survival": int(c[["os_months", "os_event"]].dropna().shape[0]) if "os_months" in c else 0,
            "n_events": int(np.nansum(c["os_event"].values)) if "os_event" in c else 0,
            "subtypes": c["subtype"].value_counts(dropna=False).to_dict() if "subtype" in c else {},
            "idh": c["idh_status"].value_counts(dropna=False).to_dict() if "idh_status" in c else {},
            "scale": self.expression_scale,
        }


def empty_clinical(index) -> pd.DataFrame:
    df = pd.DataFrame(index=pd.Index(index, name="sample_id"))
    for col in CLINICAL_COLUMNS:
        df[col] = np.nan
    return df
