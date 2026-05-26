from langgraph.graph import StateGraph, START, END

from .models import ResearchState
from .fetch_target import fetch_target_company
from .fetch_competitor import fetch_competitors
from .analysis_compare import analyze_and_compare

def graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("get_target", fetch_target_company)
    workflow.add_node("get_competitors", fetch_competitors)
    workflow.add_node("analyze", analyze_and_compare)

    workflow.add_edge(START, "get_target")
    workflow.add_edge("get_target", "get_competitors")
    workflow.add_edge("get_competitors", "analyze")
    workflow.add_edge("analyze", END)
    
    app = workflow.compile()
    
    return app

