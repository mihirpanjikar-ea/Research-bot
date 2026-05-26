import datetime
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_exa import ExaSearchResults
from pydantic import BaseModel, Field

from .config import (
    MAX_REMEDIATION_ATTEMPTS,
    RECENCY_CUTOFF_DAYS,
    STALE_SIGNAL_THRESHOLD_MONTHS,
    extraction_llm,
)
from .models import CompanyProfile, IntelligenceState


# ---------------------------------------------------------------------------
# Local Pydantic models (only used in this module)
# ---------------------------------------------------------------------------

class TrackAExtraction(BaseModel):
    """Structured output for official-site extraction (Track A)."""
    name: str = Field(default="", description="Official company name")
    core_value_proposition: str = Field(
        default="", description="One-sentence summary of what the company offers and to whom"
    )
    target_audience: List[str] = Field(
        default_factory=list, description="Buyer segments or user personas explicitly targeted"
    )
    key_features: List[str] = Field(default_factory=list)
    pricing_model: Optional[str] = None
    stale_fields: List[str] = Field(
        default_factory=list,
        description="Fields considered stale (older than the recency threshold)",
    )


class TrackBExtraction(BaseModel):
    """Structured output for forum/review-site extraction (Track B)."""
    key_features: List[str] = Field(default_factory=list)
    pricing_model: Optional[str] = None


class _MissingData(BaseModel):
    """Structured output for quality-gate gap-fill searches."""
    key_features: List[str] = Field(default_factory=list)
    pricing_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Core dual-track extraction
# ---------------------------------------------------------------------------

def extract_competitor_data(url: str) -> CompanyProfile:
    """
    Dual-track extraction for one competitor URL.

    Track A — official site (get_contents on pricing/features/about subpages):
        soft stale-flag pass; LLM marks fields older than STALE_SIGNAL_THRESHOLD_MONTHS.

    Track B — forums and review sites (ExaSearchResults with recency filter):
        harder recency signal; keeps pricing/features from fresh community sources.

    Conflict resolution:
        If Track A and B disagree on pricing, Track A wins but the field is
        recorded in conflict_fields for downstream transparency.

    Feature merging:
        Union of Track A and Track B key_features (deduped).
    """
    search_url = url if "://" in url else f"https://{url}"
    cutoff_date = (
        datetime.datetime.now() - datetime.timedelta(days=RECENCY_CUTOFF_DAYS)
    ).strftime("%Y-%m-%d")

    exa = ExaSearchResults()

    # --- Track B: forums/reviews (recency-filtered) ---
    forum_query = f'"{url}" pricing OR features OR reviews'
    forum_content = ""
    try:
        forum_res = exa.invoke({
            "query": forum_query,
            "num_results": 3,
            "type": "auto",
            "start_published_date": cutoff_date,
            "text_contents_options": True,
        })
        for res in forum_res.results:
            forum_content += (
                f"URL: {res.url}\nContent:\n{getattr(res, 'text', '')[:1000]}...\n\n"
            )
    except Exception as e:
        print(f"Track B error for {url}: {e}")

    # --- Track A: official site subpages ---
    site_content = ""
    try:
        site_res = exa.client.get_contents(
            [search_url],
            text={"max_characters": 3000},
            subpages=3,
            subpage_target=["pricing", "features", "about"],
        )
        for res in site_res.results:
            site_content += (
                f"URL: {res.url}\nContent:\n{getattr(res, 'text', '')[:1000]}...\n\n"
            )
    except Exception as e:
        print(f"Track A error for {url}: {e}")

    # --- Extract Track B ---
    prompt_b = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a business analyst. Extract key features and pricing from "
            "the given forum and review content.",
        ),
        ("user", "Company URL: {url}\n\nContext:\n{context}"),
    ])
    chain_b = prompt_b | extraction_llm.with_structured_output(TrackBExtraction)
    try:
        extracted_b = chain_b.invoke({"url": url, "context": forum_content or "No content."})
        # Some LLM chains may return plain dicts; ensure we have a TrackBExtraction instance
        if isinstance(extracted_b, dict):
            extracted_b = TrackBExtraction.model_validate(extracted_b)
    except Exception as e:
        print(f"Track B extraction failed for {url}: {e}")
        extracted_b = TrackBExtraction()

    # --- Extract Track A with stale flagging ---
    prompt_a = ChatPromptTemplate.from_messages([
        (
            "system",
            f"You are a business analyst. From the company's official web content, extract: "
            f"official company name, a one-sentence core value proposition, target audience "
            f"segments, key features, and pricing. Flag fields as stale if they appear older "
            f"than {STALE_SIGNAL_THRESHOLD_MONTHS} months.",
        ),
        ("user", "Company URL: {url}\n\nContext:\n{context}"),
    ])
    chain_a = prompt_a | extraction_llm.with_structured_output(TrackAExtraction)
    try:
        extracted_a = chain_a.invoke({"url": url, "context": site_content or "No content."})
        # Normalize plain-dict outputs into the pydantic model for attribute access
        if isinstance(extracted_a, dict):
            extracted_a = TrackAExtraction.model_validate(extracted_a)
    except Exception as e:
        print(f"Track A extraction failed for {url}: {e}")
        extracted_a = TrackAExtraction()

    # --- Conflict resolution ---
    conflict_fields: List[str] = []
    if (
        extracted_a.pricing_model
        and extracted_b.pricing_model
        and extracted_a.pricing_model != extracted_b.pricing_model
    ):
        conflict_fields.append("pricing_model")

    merged_features = list(set(extracted_a.key_features + extracted_b.key_features))
    # Track A pricing takes precedence; Track B fills gap when A is absent.
    final_pricing = extracted_a.pricing_model or extracted_b.pricing_model or ""

    return CompanyProfile(
        name=extracted_a.name or url,
        url=url,
        core_value_proposition=extracted_a.core_value_proposition,
        target_audience=extracted_a.target_audience,
        key_features=merged_features,
        pricing_model=final_pricing,
        stale_fields=extracted_a.stale_fields,
        conflict_fields=conflict_fields,
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def extract_competitor_node(state: dict) -> dict:
    """
    Single-URL extraction node — invoked once per competitor via the Send API.

    Receives {"url": "<competitor_url>"} dispatched by route_triage.
    Each parallel invocation is independent; results are merged into
    competitors_data by the merge_competitors reducer.
    """
    url = state.get("url", "")
    if not url:
        return {"competitors_data": {}}

    print(f"--- Extracting competitor: {url} ---")
    try:
        profile = extract_competitor_data(url)
    except Exception as e:
        print(f"Failed to extract {url}: {e}")
        return {"competitors_data": {}}

    return {"competitors_data": {url: profile}}


def quality_gate_node(state: IntelligenceState) -> dict:
    """
    Bounded remediation loop for missing competitor data fields.

    For each competitor with missing key_features or pricing_model:
        1. Ask the LLM to craft a targeted Exa query.
        2. Search Exa with a recency filter.
        3. Extract the missing fields from results.
        4. Repeat up to MAX_REMEDIATION_ATTEMPTS times.

    If fields remain missing after all attempts, they are set to
    "Known Unknown" so downstream synthesis can surface the gap explicitly.
    """
    competitors_dict = state.get("competitors_data") or {}
    updated: dict = {}
    cutoff_date = (
        datetime.datetime.now() - datetime.timedelta(days=RECENCY_CUTOFF_DAYS)
    ).strftime("%Y-%m-%d")

    query_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Generate a concise Exa search query to find specific missing "
            "competitive intelligence about the given company.",
        ),
        ("user", "Company: {name} ({url})\nMissing: {missing}\n\nGenerate one search query."),
    ])

    extract_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a business analyst. Extract only the requested missing "
            "information from the provided context.",
        ),
        ("user", "Company: {name}\nMissing: {missing}\n\nContext:\n{context}"),
    ])

    query_chain = query_prompt | extraction_llm
    extract_chain = extract_prompt | extraction_llm.with_structured_output(_MissingData)

    for url, profile in competitors_dict.items():
        while profile.remediation_attempts < MAX_REMEDIATION_ATTEMPTS:
            missing = []
            if not profile.key_features:
                missing.append("key features")
            if not profile.pricing_model:
                missing.append("pricing")
            if not missing:
                break

            print(
                f"--- Quality gate: remediating {url} "
                f"(attempt {profile.remediation_attempts + 1}) "
                f"- missing: {missing} ---"
            )

            try:
                query_res = query_chain.invoke({
                    "name": profile.name,
                    "url": profile.url,
                    "missing": ", ".join(missing),
                })
                search_res = ExaSearchResults().invoke({
                    "query": query_res.content,
                    "num_results": 3,
                    "type": "auto",
                    "start_published_date": cutoff_date,
                    "text_contents_options": True,
                })
                context = "\n\n".join(
                    f"URL: {res.url}\nContent:\n{getattr(res, 'text', '')[:1000]}..."
                    for res in search_res.results
                )
                missing_data = extract_chain.invoke({
                    "name": profile.name,
                    "missing": ", ".join(missing),
                    "context": context,
                })
                if not profile.key_features and missing_data.key_features:
                    profile.key_features = missing_data.key_features
                if not profile.pricing_model and missing_data.pricing_model:
                    profile.pricing_model = missing_data.pricing_model
            except Exception as e:
                print(f"Remediation error for {url}: {e}")

            profile.remediation_attempts += 1

        # Known Unknown fallback — explicit gap marker for synthesis
        if not profile.key_features:
            profile.key_features = ["Known Unknown"]
        if not profile.pricing_model:
            profile.pricing_model = "Known Unknown"

        updated[url] = profile

    return {"competitors_data": updated}
