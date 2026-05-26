from langgraph.graph import StateGraph, START, END

from .analysis_compare import analyze_and_compare
from .domain_anchoring import domain_anchoring_node
from .extract_competitor import extract_competitors
from .fetch_competitor import fetch_competitors
from .fetch_target import fetch_target_company
from .models import IntelligenceState
from .sanitization import sanitization_node


def graph():
    workflow = StateGraph(IntelligenceState)

    workflow.add_node("sanitize", sanitization_node)
    workflow.add_node("anchor", domain_anchoring_node)
    workflow.add_node("scrape_target", fetch_target_company)
    workflow.add_node("discover_competitors", fetch_competitors)
    workflow.add_node("extract_competitors", extract_competitors)
    workflow.add_node("synthesize", analyze_and_compare)

    workflow.add_edge(START, "sanitize")
    workflow.add_edge("sanitize", "anchor")
    workflow.add_edge("anchor", "scrape_target")
    workflow.add_edge("scrape_target", "discover_competitors")
    workflow.add_edge("discover_competitors", "extract_competitors")
    workflow.add_edge("extract_competitors", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()
