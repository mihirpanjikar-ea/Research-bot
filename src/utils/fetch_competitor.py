import datetime
from typing import List
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from pydantic import BaseModel, Field

from .config import (
    MAX_PARALLEL_COMPETITORS,
    MAX_TRIAGE_FALLBACK_ATTEMPTS,
    RECENCY_CUTOFF_DAYS,
    exa_find_similar,
    exa_search,
    extraction_llm,
)
from .models import IntelligenceState


# ---------------------------------------------------------------------------
# Local Pydantic models (only used in this module)
# ---------------------------------------------------------------------------

class CompetitorSeed(BaseModel):
    competitor_urls: List[str] = Field(
        description="List of official root domains for direct business competitors."
    )


class TriageDecision(BaseModel):
    direct_competitor_urls: List[str] = Field(
        description=(
            "URLs confirmed as direct business competitors sharing the same "
            "target customer and core use-case."
        )
    )


# ---------------------------------------------------------------------------
# Noise domains: directories, aggregators, social media
# ---------------------------------------------------------------------------

_NOISE_DOMAINS = [
    "g2.com", "capterra.com", "getapp.com", "trustradius.com",
    "linkedin.com", "crunchbase.com", "facebook.com", "twitter.com",
    "youtube.com", "medium.com", "reddit.com", "producthunt.com",
    "news.ycombinator.com", "techcrunch.com", "forbes.com",
]


def _to_root_url(url: str) -> str:
    """
    Strip path, query, and fragment — return a clean https://bare-domain root.
    Ensures extract_competitors always scrapes the competitor homepage, never
    a specific article or subpage that might mention the target company.
    """
    if "://" not in url:
        url = "https://" + url
    parsed = urlparse(url)
    netloc = parsed.netloc
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return f"https://{netloc}" if netloc else url


# ---------------------------------------------------------------------------
# Discovery nodes
# ---------------------------------------------------------------------------

def track_a_discovery_node(state: IntelligenceState) -> dict:
    """Track A: Exa Similarity Search — finds URLs structurally similar to the target."""
    url = state.get("target_company_url", "")
    if not url:
        return {"discovery_tracks": {"track_a": []}}

    search_url = url if "://" in url else f"https://{url}"
    print(f"--- Track A (similarity) for: {search_url} ---")

    try:
        response = exa_find_similar.invoke({
            "url": search_url,
            "num_results": MAX_PARALLEL_COMPETITORS * 2,
            "exclude_source_domain": True,
        })
        urls = [r.url for r in response.results]
    except Exception as e:
        print(f"Track A error: {e}")
        urls = []

    return {"discovery_tracks": {"track_a": urls}}


def track_b_discovery_node(state: IntelligenceState) -> dict:
    """
    Track B: Exa Keyword Search — intent-based queries for alternatives/comparisons.

    Raw Exa results are listicle/comparison articles. An LLM extraction pass
    pulls the actual competitor domains out of the article bodies so we return
    company URLs, not article URLs.
    """
    profile = state.get("target_profile")
    target_name = (
        profile.name if profile and profile.name
        else state.get("target_company_url", "")
    )
    if not target_name:
        return {"discovery_tracks": {"track_b": []}}

    query = (
        f'"{target_name} alternative" OR '
        f'"vs {target_name}" OR '
        f'"best {target_name} competitors"'
    )
    cutoff_date = (
        datetime.datetime.now() - datetime.timedelta(days=RECENCY_CUTOFF_DAYS)
    ).strftime("%Y-%m-%d")

    print(f"--- Track B (keyword) for: {target_name} ---")

    urls: List[str] = []
    try:
        response = exa_search.invoke({
            "query": query,
            "num_results": MAX_PARALLEL_COMPETITORS * 2,
            "type": "auto",
            "start_published_date": cutoff_date,
            "text_contents_options": True,
        })

        article_corpus = "\n\n---\n\n".join(
            f"Article: {res.url}\n{getattr(res, 'text', '')[:2000]}"
            for res in response.results
        )

        if article_corpus.strip():
            extract_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a market researcher. From the supplied alternative/comparison "
                    "articles, list the official website root domains (e.g. example.com) of "
                    "direct business competitors to the target. Exclude the target itself, "
                    "news sites, review aggregators, and directories such as g2.com or capterra.com.",
                ),
                ("user", "Target: {name}\n\nArticles:\n{corpus}"),
            ])
            chain = extract_prompt | extraction_llm.with_structured_output(CompetitorSeed)
            extracted = chain.invoke({"name": target_name, "corpus": article_corpus})
            urls = (extracted.get("competitor_urls") or []) if isinstance(extracted, dict) else (extracted.competitor_urls or [])
    except Exception as e:
        print(f"Track B error: {e}")

    return {"discovery_tracks": {"track_b": urls}}


def track_c_discovery_node(state: IntelligenceState) -> dict:
    """
    Track C: LLM Seed Discovery — uses model knowledge as a corroboration signal.

    Design rule: Track C alone is never sufficient. Its output is filtered out
    by the triage node unless corroborated by Track A or B. This guards against
    stale LLM training data acting as a primary discovery signal.
    """
    profile = state.get("target_profile")
    target_name = (
        profile.name if profile and profile.name
        else state.get("target_company_url", "")
    )
    if not target_name:
        return {"discovery_tracks": {"track_c": []}}

    print(f"--- Track C (LLM seed) for: {target_name} ---")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a market research expert. Identify the top 8 direct business "
            "competitors for the given company. Provide only their official website "
            "root domains (e.g. example.com).",
        ),
        ("user", "Company: {name}"),
    ])

    try:
        chain = prompt | extraction_llm.with_structured_output(CompetitorSeed)
        res = chain.invoke({"name": target_name})
        urls = res.get("competitor_urls", []) if isinstance(res, dict) else (res.competitor_urls or [])
    except Exception as e:
        print(f"Track C error: {e}")
        urls = []

    return {"discovery_tracks": {"track_c": urls}}


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

def triage_node(state: IntelligenceState) -> dict:
    """
    Unions all discovery tracks, scores by cross-track confidence, filters
    noise domains and Track-C-only results, then runs an LLM triage pass to
    remove tangential companies. Caps the final list at MAX_PARALLEL_COMPETITORS.
    """
    tracks = state.get("discovery_tracks", {})
    target_url = state.get("target_company_url", "").lower()

    def normalize(url: str) -> str:
        parsed = urlparse(url)
        netloc = parsed.netloc or url.split("/")[0]
        netloc = netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc

    # Build a scored map: normalized_domain -> {original_url, tracks seen in}
    url_to_meta: dict = {}
    for track_id, track_urls in tracks.items():
        for url in (track_urls or []):
            norm = normalize(url)
            if not norm:
                continue
            # Skip exact domain match or subdomain of target (e.g. blog.target.com)
            if norm == target_url or norm.endswith("." + target_url):
                continue
            if norm not in url_to_meta:
                url_to_meta[norm] = {"original_url": url, "tracks": set()}
            url_to_meta[norm]["tracks"].add(track_id)

    # Filter noise + Track-C-only guard
    scored = [
        {
            "url": meta["original_url"],
            "norm": norm,
            "score": len(meta["tracks"]),
            "tracks": meta["tracks"],
        }
        for norm, meta in url_to_meta.items()
        if not any(noise in norm for noise in _NOISE_DOMAINS)
        and meta["tracks"] != {"track_c"}   # must be corroborated
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)

    if not scored:
        print("--- Triage: no candidates survived filtering ---")
        return {"competitor_urls": []}

    # LLM triage pass: remove tangential companies that survived the blocklist
    profile = state.get("target_profile")
    target_desc = (
        f"{profile.name}: {profile.core_value_proposition}"
        if profile and profile.name
        else state.get("target_company_url", "unknown")
    )
    # Use clean bare domains as candidates — easier for the LLM to evaluate and
    # avoids confusion when a Track A URL is a subpage mentioning the target.
    candidates_text = "\n".join(c["norm"] for c in scored)

    triage_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a market analyst. From the candidate list, return only direct business "
            "competitors to the target — same target customer, same core use-case. Remove "
            "adjacent tools, enterprise suites, resellers, and tangential software.",
        ),
        ("user", "Target: {target}\n\nCandidates (one per line):\n{candidates}"),
    ])

    try:
        chain = triage_prompt | extraction_llm.with_structured_output(TriageDecision)
        decision = chain.invoke({"target": target_desc, "candidates": candidates_text})
        if isinstance(decision, dict):
            selected = decision.get("direct_competitor_urls", [])[:MAX_PARALLEL_COMPETITORS]
        else:
            selected = decision.direct_competitor_urls[:MAX_PARALLEL_COMPETITORS]
    except Exception as e:
        print(f"LLM triage failed, falling back to score-ranked list: {e}")
        selected = [c["norm"] for c in scored[:MAX_PARALLEL_COMPETITORS]]

    # Normalize to https://domain roots and re-filter target (belt-and-suspenders).
    # This also fixes Track A returning subpage URLs (e.g. competitor.com/press/compare-us)
    # which would cause extract_competitors to scrape an article about the target instead
    # of the competitor's own homepage.
    top_urls = []
    for url in selected:
        root = _to_root_url(url)
        norm_check = normalize(root)
        if norm_check == target_url or norm_check.endswith("." + target_url):
            continue
        top_urls.append(root)

    print(f"--- Triage selected {len(top_urls)} competitor(s): {top_urls} ---")
    return {"competitor_urls": top_urls}


def triage_fallback_node(state: IntelligenceState) -> dict:
    """
    Broad fallback search when triage yielded zero results.
    Writes into discovery_tracks (not competitor_urls directly) so the next
    triage re-run picks it up through the normal union/score/filter pipeline.
    """
    profile = state.get("target_profile")
    target_name = (
        profile.name if profile and profile.name
        else state.get("target_company_url", "")
    )
    attempts = state.get("triage_fallback_attempts", 0)
    print(f"--- Triage fallback attempt {attempts + 1} for: {target_name} ---")

    try:
        response = exa_search.invoke({
            "query": f"list of competitors for {target_name}",
            "num_results": MAX_PARALLEL_COMPETITORS * 2,
            "type": "auto",
        })
        urls = [res.url for res in response.results]
    except Exception as e:
        print(f"Triage fallback error: {e}")
        urls = []

    return {
        "discovery_tracks": {"track_fallback": urls},
        "triage_fallback_attempts": attempts + 1,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_triage(state: IntelligenceState) -> str | list[Send]:
    """
    Three-way routing after triage:

    1. No URLs + attempts remaining  -> "triage_fallback" (retry loop)
    2. No URLs + attempts exhausted  -> "quality_gate"   (skip extraction gracefully)
    3. URLs found                    -> [Send("extract_competitor", {"url": u}) ...]
                                        (parallel fan-out, one Send per competitor)
    """
    urls = state.get("competitor_urls", [])
    attempts = state.get("triage_fallback_attempts", 0)
    if not urls and attempts < MAX_TRIAGE_FALLBACK_ATTEMPTS:
        return "triage_fallback"
    if not urls:
        print("--- Triage: no competitors found after fallback, skipping extraction ---")
        return "quality_gate"
    print(f"--- Triage: dispatching {len(urls)} competitor(s) in parallel ---")
    return [Send("extract_competitor", {"url": url}) for url in urls]
