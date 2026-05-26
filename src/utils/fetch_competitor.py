from langchain_exa import ExaFindSimilarResults

from .models import IntelligenceState

find_similar_tool = ExaFindSimilarResults()

MAX_COMPETITORS = 3


def fetch_competitors(state: IntelligenceState) -> dict:
    """Track A only: find_similar returns raw competitor URLs. No triage, no fallback."""
    url = state["target_company_url"]
    search_url = url if "://" in url else f"https://{url}"
    print(f"--- Discovering competitors (Track A) for: {search_url} ---")

    response = find_similar_tool.invoke({
        "url": search_url,
        "num_results": MAX_COMPETITORS,
        "exclude_source_domain": True,
    })

    competitor_urls = [r.url for r in response.results]
    return {"competitor_urls": competitor_urls}
