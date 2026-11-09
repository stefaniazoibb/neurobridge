"""Typed results produced by the analysis layer. Every result carries its method and limitations."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AnalysisKind(str, Enum):
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class AnalysisRecord(BaseModel):
    """Provenance header attached to every analysis output."""

    analysis_id: str
    name: str
    kind: AnalysisKind
    dataset_ids: list[str]
    method: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None
    software: dict[str, str] = Field(default_factory=dict)
    started_at: str
    finished_at: Optional[str] = None
    inputs: list[str] = Field(default_factory=list, description="Input references (file hashes, dataset versions)")
    outputs: list[str] = Field(default_factory=list, description="Output references (figure/table ids)")
    n_used: Optional[int] = None
    n_excluded: Optional[int] = None
    limitations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EffectSizeRow(BaseModel):
    gene: str
    log2_fc: float
    hedges_g: float
    hedges_g_ci_low: float
    hedges_g_ci_high: float
    mean_a: float
    mean_b: float
    p_value: float
    q_value: float
    test: str


class DEResult(BaseModel):
    record: AnalysisRecord
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    rows: list[EffectSizeRow]
    n_tested: int
    n_significant: int
    alpha: float = 0.05
    correction: str = "Benjamini-Hochberg"

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([r.model_dump() for r in self.rows])


class EnrichmentResult(BaseModel):
    record: AnalysisRecord
    library: str
    method: str
    terms: list[dict[str, Any]]
    n_input_genes: int
    background: Optional[str] = None


class SignatureScoreResult(BaseModel):
    record: AnalysisRecord
    signature: str
    genes_requested: list[str]
    genes_found: list[str]
    genes_missing: list[str]
    scores: dict[str, float]
    method: str


class GroupComparisonResult(BaseModel):
    record: AnalysisRecord
    variable: str
    grouping: str
    group_stats: list[dict[str, Any]]
    omnibus_test: str
    omnibus_statistic: float
    omnibus_p: float
    pairwise: list[dict[str, Any]] = Field(default_factory=list)
    effect_size_name: Optional[str] = None
    effect_size: Optional[float] = None


class CorrelationResult(BaseModel):
    record: AnalysisRecord
    genes: list[str]
    method: str
    matrix: list[list[float]]
    p_values: list[list[float]]
    q_values: list[list[float]]
    n: int


class KMGroup(BaseModel):
    label: str
    n: int
    events: int
    median_survival: Optional[float]
    median_ci_low: Optional[float] = None
    median_ci_high: Optional[float] = None
    timeline: list[float]
    survival: list[float]
    ci_low: list[float]
    ci_high: list[float]
    at_risk: list[int] = Field(default_factory=list)


class KMResult(BaseModel):
    record: AnalysisRecord
    stratifier: str
    time_unit: str
    groups: list[KMGroup]
    logrank_statistic: Optional[float]
    logrank_p: Optional[float]
    n_total: int
    n_events: int


class CoxRow(BaseModel):
    covariate: str
    coef: float
    hr: float
    hr_ci_low: float
    hr_ci_high: float
    se: float
    z: float
    p: float


class CoxResult(BaseModel):
    record: AnalysisRecord
    model: str
    rows: list[CoxRow]
    n: int
    events: int
    concordance: Optional[float]
    log_likelihood: Optional[float]
    aic: Optional[float] = None
    ph_test: dict[str, Any] = Field(default_factory=dict, description="Schoenfeld-residual based proportional-hazards test per covariate")
    ph_assumption_ok: Optional[bool] = None
    covariates_adjusted: list[str] = Field(default_factory=list)


class SurvivalBundle(BaseModel):
    cohort_id: str
    km: Optional[KMResult] = None
    cox_univariable: Optional[CoxResult] = None
    cox_multivariable: Optional[CoxResult] = None
    sensitivity: list[dict[str, Any]] = Field(default_factory=list)


class ValidationCohortResult(BaseModel):
    cohort_id: str
    role: str
    n: int
    events: Optional[int] = None
    estimate_name: str
    estimate: float
    ci_low: float
    ci_high: float
    p: float
    direction_consistent: Optional[bool] = None
    significant: bool
    notes: list[str] = Field(default_factory=list)


class ValidationSummary(BaseModel):
    record: AnalysisRecord
    hypothesis: str
    discovery_cohort: str
    validation_cohorts: list[str]
    per_cohort: list[ValidationCohortResult]
    pooled_estimate: Optional[float] = None
    pooled_ci_low: Optional[float] = None
    pooled_ci_high: Optional[float] = None
    pooled_p: Optional[float] = None
    heterogeneity_i2: Optional[float] = None
    heterogeneity_q_p: Optional[float] = None
    n_cohorts_replicating: int = 0
    verdict: str
    verdict_rationale: list[str] = Field(default_factory=list)
