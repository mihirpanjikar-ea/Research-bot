from langchain_core.prompts import ChatPromptTemplate
from langchain_exa import ExaSearchResults
from langchain_openai import ChatOpenAI

from .models import CompanyProfile, IntelligenceState

search_tool = ExaSearchResults()
extraction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def _extract_one(url: str) -> CompanyProfile:
    scrape_url = url if "://" in url else f"https://{url}"
    response = search_tool.client.get_contents(
        [scrape_url],
        text={"max_characters": 3000},
        subpages=3,
        subpage_target=["about", "pricing", "features"],
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
    return chain.invoke({"url": url, "context": content})


def extract_competitors(state: IntelligenceState) -> dict:
    """Sequentially scrape + extract each competitor URL into a CompanyProfile."""
    urls = state.get("competitor_urls", [])
    competitors_data: dict = {}

    for url in urls:
        print(f"--- Extracting competitor: {url} ---")
        try:
            profile = _extract_one(url)
        except Exception as e:
            print(f"Failed to extract {url}: {e}")
            continue
        competitors_data[url] = profile

    return {"competitors_data": competitors_data}
