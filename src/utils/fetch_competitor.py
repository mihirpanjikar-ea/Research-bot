import datetime
from typing import List
from urllib.parse import urlparse

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send
from langsmith import traceable
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
from .observability import log_event, log_swallowed_exception


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
    log_event("discovery.track_a.start", url=search_url)

    try:
        response = exa_find_similar.invoke({
            "url": search_url,
            "num_results": MAX_PARALLEL_COMPETITORS * 2,
            "exclude_source_domain": True,
        })
        urls = [r.url for r in response.results]
    except Exception as e:
        log_swallowed_exception("discovery.track_a", e, url=search_url)
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

    log_event("discovery.track_b.start", target=target_name)

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
        log_swallowed_exception("discovery.track_b", e, target=target_name)

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

    log_event("discovery.track_c.start", target=target_name)

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
        log_swallowed_exception("discovery.track_c", e, target=target_name)
        urls = []

    return {"discovery_tracks": {"track_c": urls}}


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

def _normalize_domain(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc or url.split("/")[0]
    netloc = netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


@traceable(name="triage.score_and_filter", run_type="chain")
def _score_and_filter_candidates(
    tracks: dict, target_url: str
) -> List[dict]:
    """
    Pre-LLM triage step: union discovery tracks, drop the target itself,
    filter noise domains, enforce Track-C corroboration, score by cross-track
    overlap, sort. Wrapped in @traceable so the dropped candidates and the
    surviving scored list are visible in LangSmith.
    """
    # Build scored map: normalized_domain -> {original_url, tracks seen in}
    url_to_meta: dict = {}
    for track_id, track_urls in tracks.items():
        for url in (track_urls or []):
            norm = _normalize_domain(url)
            if not norm:
                continue
            # Skip exact domain match or subdomain of target (e.g. blog.target.com)
            if norm == target_url or norm.endswith("." + target_url):
                continue
            if norm not in url_to_meta:
                url_to_meta[norm] = {"original_url": url, "tracks": set()}
            url_to_meta[norm]["tracks"].add(track_id)

    scored = [
        {
            "url": meta["original_url"],
            "norm": norm,
            "score": len(meta["tracks"]),
            "tracks": sorted(meta["tracks"]),  # set is not JSON-serialisable for LangSmith
        }
        for norm, meta in url_to_meta.items()
        if not any(noise in norm for noise in _NOISE_DOMAINS)
        and meta["tracks"] != {"track_c"}   # must be corroborated
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def triage_node(state: IntelligenceState) -> dict:
    """
    Unions all discovery tracks, scores by cross-track confidence, filters
    noise domains and Track-C-only results, then runs an LLM triage pass to
    remove tangential companies. Caps the final list at MAX_PARALLEL_COMPETITORS.
    """
    tracks = state.get("discovery_tracks", {})
    target_url = state.get("target_company_url", "").lower()

    scored = _score_and_filter_candidates(tracks, target_url)

    if not scored:
        log_event("triage.no_candidates")
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
        log_swallowed_exception("triage.llm", e, fallback="score_ranked")
        selected = [c["norm"] for c in scored[:MAX_PARALLEL_COMPETITORS]]

    # Normalize to https://domain roots and re-filter target (belt-and-suspenders).
    # This also fixes Track A returning subpage URLs (e.g. competitor.com/press/compare-us)
    # which would cause extract_competitors to scrape an article about the target instead
    # of the competitor's own homepage.
    top_urls = []
    for url in selected:
        root = _to_root_url(url)
        norm_check = _normalize_domain(root)
        if norm_check == target_url or norm_check.endswith("." + target_url):
            continue
        top_urls.append(root)

    log_event("triage.selected", count=len(top_urls), urls=top_urls)
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
    log_event("triage.fallback.start", attempt=attempts + 1, target=target_name)

    try:
        response = exa_search.invoke({
            "query": f"list of competitors for {target_name}",
            "num_results": MAX_PARALLEL_COMPETITORS * 2,
            "type": "auto",
        })
        urls = [res.url for res in response.results]
    except Exception as e:
        log_swallowed_exception("triage.fallback", e, target=target_name)
        urls = []

    return {
        "discovery_tracks": {"track_fallback": urls},
        "triage_fallback_attempts": attempts + 1,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@traceable(name="route_triage", run_type="chain")
def _route_triage_decision(urls: list, attempts: int) -> dict:
    """
    Pure-Python routing decision, traced so the branch (fallback / skip /
    fan-out) and the number of Sends dispatched are visible in LangSmith.
    Returns a metadata dict alongside the routing string for trace clarity.
    """
    if not urls and attempts < MAX_TRIAGE_FALLBACK_ATTEMPTS:
        return {"branch": "triage_fallback", "send_count": 0}
    if not urls:
        return {"branch": "quality_gate", "send_count": 0}
    return {"branch": "extract_competitor", "send_count": len(urls)}


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
    decision = _route_triage_decision(urls, attempts)
    branch = decision["branch"]

    if branch == "triage_fallback":
        return "triage_fallback"
    if branch == "quality_gate":
        log_event("route_triage.skip_extraction")
        return "quality_gate"
    log_event("route_triage.dispatch", count=len(urls))
    return [Send("extract_competitor", {"url": url}) for url in urls]
