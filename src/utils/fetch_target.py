from langchain_exa import ExaSearchResults
from .models import ResearchState

search_tool = ExaSearchResults()

def fetch_target_company(state: ResearchState) -> dict:
    """Fetches the target company's website."""
    print(f"--- Fetching data for target: {state['target_url']} ---")
    
    # langchain-exa tools expose the raw Exa client via `.client`.
    # Since we want to scrape an exact URL rather than "search" for it, 
    # using the underlying client's get_contents is the cleanest approach here.
    response = search_tool.client.get_contents(
        [state["target_url"]], 
        text={"max_characters": 3000}
    )
    
    if not response.results:
        content = "Could not retrieve content for the target URL."
    else:
        content = response.results[0].text
        
    return {"target_info": content}