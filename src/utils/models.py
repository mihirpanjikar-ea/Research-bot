from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class CompanyProfile(BaseModel):
    name: str
    url: str
    core_value_proposition: str = ""
    target_audience: List[str] = Field(default_factory=list)
    key_features: List[str] = Field(default_factory=list)
    pricing_model: str = ""


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
    competitor_urls: List[str]
    competitors_data: Dict[str, CompanyProfile]
    final_report: Optional[FinalReport]
