
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from utils.models import ResearchState

def analyze_and_compare(state: ResearchState) -> dict:
    """Uses OpenAI to find similarities and differences."""
    print("--- Analyzing target vs competitors ---")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    system_prompt = (
        "You are an expert business analyst. You will be provided with information about a "
        "target company and a few of its competitors. Your job is to analyze this data and "
        "provide a structured comparison. Highlight the core similarities (what industry they are in, "
        "shared features) and the key differences (unique value propositions, differing target audiences, "
        "or distinct features)."
    )
    
    user_prompt = f"""
    Target Company URL: {state['target_url']}
    Target Company Information:
    {state['target_info']}
    
    Competitors Information:
    {state['competitor_info']}
    
    Please provide the final competitive analysis.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    return {"final_analysis": response.content}