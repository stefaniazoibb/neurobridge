"""Typed description of a dataset known to NeuroBridge."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DataModality(str, Enum):
    BULK_RNASEQ = "bulk RNA-seq"
    MICROARRAY = "expression microarray"
    SCRNASEQ = "single-cell RNA-seq"
    SPATIAL = "spatial transcriptomics"
    CLINICAL = "clinical metadata"
    SURVIVAL = "survival"
    MUTATION = "somatic mutation"
    CNA = "copy number"
    METHYLATION = "DNA methylation"


class AccessMode(str, Enum):
    PUBLIC_DIRECT = "public, direct download"
    PUBLIC_API = "public, REST API"
    LISTED_ONLY = "listed, not ingested"
    DEMO = "demonstration data (not real patients)"


class IngestionStatus(str, Enum):
    CACHED = "cached locally"
    DOWNLOADABLE = "downloadable on demand"
    NOT_INGESTED = "not ingested"
    UNAVAILABLE = "unavailable"


class DatasetRecord(BaseModel):
    dataset_id: str
    name: str
    source: str
    source_url: str
    publication: Optional[str] = Field(None, description="PMID or DOI of the primary publication")
    publication_title: Optional[str] = None
    modalities: list[DataModality]
    organism: str = "Homo sapiens"
    disease: str = "glioblastoma"
    n_samples: Optional[int] = None
    n_patients: Optional[int] = None
    n_cells: Optional[int] = None
    n_genes: Optional[int] = None
    platform: Optional[str] = None
    expression_units: Optional[str] = None
    clinical_variables: list[str] = Field(default_factory=list)
    survival_available: bool = False
    access: AccessMode
    ingestion: IngestionStatus
    preprocessing: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    version: Optional[str] = None
    role: Optional[str] = Field(None, description="discovery / validation / cell-type reference / listed")
    is_demo: bool = False
    files: list[str] = Field(default_factory=list, description="Download URLs")
    notes: Optional[str] = None
    live_metadata: dict = Field(default_factory=dict, description="Metadata fetched live from the source API")
