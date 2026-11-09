"""Evidence grading and literature citations."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EvidenceLevel(str, Enum):
    """Ordered from weakest to strongest. Never promote a level beyond what the analysis supports."""

    BIOLOGICAL_PLAUSIBILITY = "biological plausibility"
    STATISTICAL_ASSOCIATION = "statistical association"
    MECHANISTIC_EVIDENCE = "mechanistic evidence"
    CAUSAL_EVIDENCE = "causal evidence"
    INSUFFICIENT = "insufficient evidence"


class Citation(BaseModel):
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str
    authors: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    source: str = Field("Europe PMC", description="Service the record was retrieved from")
    url: Optional[str] = None
    verified: bool = Field(False, description="True only if the record was retrieved from a live bibliographic API")
    retrieved_at: Optional[str] = None
    cited_by: Optional[int] = None
    abstract: Optional[str] = None
    relevance: Optional[str] = Field(None, description="Why this citation is relevant")

    def short(self) -> str:
        first = (self.authors or "").split(",")[0].strip() or "Unknown"
        return f"{first} et al., {self.journal or ''} {self.year or ''}".strip()


class Claim(BaseModel):
    text: str
    level: EvidenceLevel
    supported_by_analyses: list[str] = Field(default_factory=list, description="analysis_id references")
    supported_by_citations: list[str] = Field(default_factory=list, description="PMIDs")
    is_interpretation: bool = Field(False, description="True when the sentence is interpretation rather than a measured result")
    caveats: list[str] = Field(default_factory=list)
