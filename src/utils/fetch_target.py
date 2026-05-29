from langchain_core.prompts import ChatPromptTemplate

from .config import exa_search, extraction_llm
from .models import CompanyProfile, IntelligenceState


def fetch_target_company(state: IntelligenceState) -> dict:
    """Scrape the target's homepage + about/pricing subpages, extract a CompanyProfile."""
    url = state["target_company_url"]
    print(f"--- Scraping target: {url} ---")

    scrape_url = url if "://" in url else f"https://{url}"

    response = exa_search.client.get_contents(
        [scrape_url],
        text={"max_characters": 3000},
        subpages=5,
        subpage_target=["about", "pricing"],
    )

    if not response.results:
        content = ""
    else:
        content = "\n\n---\n\n".join(
            f"URL: {r.url}\nTitle: {getattr(r, 'title', '')}\nContent:\n{getattr(r, 'text', '')[:2000]}"
            for r in response.results
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional business analyst. Extract structured company information from the provided web content."),
        ("user", "Company URL: {url}\n\nContext:\n{context}\n\nExtract the company name, URL, core value proposition, target audience, key features, and pricing details."),
    ])

    chain = prompt | extraction_llm.with_structured_output(CompanyProfile)
    profile = chain.invoke({"url": url, "context": content})

    return {"target_profile": profile}
