from typing import Annotated, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


def merge_dict(a: dict, b: dict) -> dict:
    """Reducer that merges two dicts — used for discovery_tracks fan-out."""
    if not a:
        return b
    if not b:
        return a
    return {**a, **b}


class CompanyProfile(BaseModel):
    name: str
    url: str
    core_value_proposition: str = ""
    target_audience: List[str] = Field(default_factory=list)
    key_features: List[str] = Field(default_factory=list)
    pricing_model: str = ""
    # data-quality metadata
    remediation_attempts: int = 0
    stale_fields: List[str] = Field(default_factory=list)
    conflict_fields: List[str] = Field(default_factory=list)


class CompetitorComparison(BaseModel):
    name: str
    url: str
    overlapping_features: List[str] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)
    target_advantages: List[str] = Field(default_factory=list)
    pricing_delta: str = ""


class SWOTAnalysis(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    executive_summary: str = ""
    target_company: str = ""
    competitor_comparisons: List[CompetitorComparison] = Field(default_factory=list)
    swot: SWOTAnalysis = Field(default_factory=SWOTAnalysis)


class IntelligenceState(TypedDict):
    target_company_url: str
    target_profile: Optional[CompanyProfile]
    # Stage 3: parallel discovery tracks (merged by reducer)
    discovery_tracks: Annotated[Dict[str, List[str]], merge_dict]
    triage_fallback_attempts: int
    competitor_urls: List[str]
    competitors_data: Dict[str, CompanyProfile]
    final_report: Optional[FinalReport]
