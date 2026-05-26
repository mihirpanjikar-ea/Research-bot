from langgraph.graph import StateGraph, START, END

from .analysis_compare import analyze_and_compare
from .domain_anchoring import domain_anchoring_node
from .extract_competitor import extract_competitors, quality_gate_node
from .fetch_competitor import (
    route_triage,
    track_a_discovery_node,
    track_b_discovery_node,
    track_c_discovery_node,
    triage_fallback_node,
    triage_node,
)
from .fetch_target import fetch_target_company
from .models import IntelligenceState
from .sanitization import sanitization_node


def graph():
    workflow = StateGraph(IntelligenceState)

    # --- Nodes ---
    workflow.add_node("sanitize", sanitization_node)
    workflow.add_node("anchor", domain_anchoring_node)
    workflow.add_node("scrape_target", fetch_target_company)

    # Stage 3: parallel discovery
    workflow.add_node("discovery_a", track_a_discovery_node)
    workflow.add_node("discovery_b", track_b_discovery_node)
    workflow.add_node("discovery_c", track_c_discovery_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("triage_fallback", triage_fallback_node)

    workflow.add_node("extract_competitors", extract_competitors)
    # quality gate (bounded remediation + Known Unknown fallback)
    workflow.add_node("quality_gate", quality_gate_node)
    workflow.add_node("synthesize", analyze_and_compare)

    # --- Edges ---
    workflow.add_edge(START, "sanitize")
    workflow.add_edge("sanitize", "anchor")
    workflow.add_edge("anchor", "scrape_target")

    # Fan-out: scrape_target -> [discovery_a, discovery_b, discovery_c] in parallel
    workflow.add_conditional_edges(
        "scrape_target",
        lambda _: ["discovery_a", "discovery_b", "discovery_c"],
        ["discovery_a", "discovery_b", "discovery_c"],
    )

    # Fan-in: all three tracks converge at triage (merge_dict reducer accumulates them)
    workflow.add_edge("discovery_a", "triage")
    workflow.add_edge("discovery_b", "triage")
    workflow.add_edge("discovery_c", "triage")

    # Triage conditional: fallback loop or proceed
    workflow.add_conditional_edges(
        "triage",
        route_triage,
        {"fallback": "triage_fallback", "next": "extract_competitors"},
    )
    workflow.add_edge("triage_fallback", "triage")

    workflow.add_edge("extract_competitors", "quality_gate")
    workflow.add_edge("quality_gate", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()
