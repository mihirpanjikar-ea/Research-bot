from langchain_exa import ExaFindSimilarResults
from .models import ResearchState

find_similar_tool = ExaFindSimilarResults()

def fetch_competitors(state: ResearchState) -> dict:
    """Finds similar companies using the LangChain ExaFindSimilarResults tool."""
    print("--- Finding competitors via langchain-exa ---")
    
    # As a LangChain Runnable, the tool accepts a dictionary of arguments
    response = find_similar_tool.invoke({
        "url": state["target_url"],
        "num_results": 3,
        "text_contents_options": {"max_characters": 3000}
    })
    
    competitor_data = ""
    # The langchain-exa tool returns the Exa response object
    for idx, result in enumerate(response.results):
        competitor_data += f"\n\nCompetitor {idx + 1}: {result.url}\nContent: {result.text[:2000]}...\n"
        
    return {"competitor_info": competitor_data}