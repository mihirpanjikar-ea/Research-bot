from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langsmith import traceable
from pydantic import BaseModel, Field


@traceable(name="reducer.discovery_tracks", run_type="chain")
def merge_dict(a: dict, b: dict) -> dict:
    """Reducer that merges two dicts — used for discovery_tracks fan-out."""
    if not a:
        return b
    if not b:
        return a
    return {**a, **b}


@traceable(name="reducer.competitors_data", run_type="chain")
def merge_competitors(a: dict, b: dict) -> dict:
    """Reducer that merges competitor profile dicts — used for parallel extraction fan-in."""
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
    # Reflects data completeness after Stage 4 quality gate
    data_confidence: Literal["high", "medium", "low", "known_unknown"] = "medium"


class SWOTAnalysis(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)
    threats: List[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    executive_summary: str = ""
    report_version: int = 1
    target_company: str = ""
    competitor_comparisons: List[CompetitorComparison] = Field(default_factory=list)
    swot: SWOTAnalysis = Field(default_factory=SWOTAnalysis)
    # data-quality transparency fields surfaced by synthesis
    stale_data_warnings: List[str] = Field(default_factory=list)
    known_unknowns: List[str] = Field(default_factory=list)
    synthesis_model_used: str = ""


class IntelligenceState(TypedDict):
    target_company_url: str
    target_profile: Optional[CompanyProfile]
    # parallel discovery tracks (merged by reducer)
    discovery_tracks: Annotated[Dict[str, List[str]], merge_dict]
    triage_fallback_attempts: int
    competitor_urls: List[str]
    # parallel extraction fan-in (merged by reducer)
    competitors_data: Annotated[Dict[str, CompanyProfile], merge_competitors]
    final_report: Optional[FinalReport]
    # HITL + rolling summarisation
    historical_context: List[BaseMessage]        # plain list — replaced on each write
    remediation_context: Annotated[List[BaseMessage], add_messages]  # accumulated per cycle
    user_feedback: Optional[str]
    hitl_iteration_count: int
