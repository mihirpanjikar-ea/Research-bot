from langgraph.graph import StateGraph, START, END

from .analysis_compare import analyze_and_compare
from .domain_anchoring import domain_anchoring_node
from .extract_competitor import extract_competitor_node, quality_gate_node
from .fetch_competitor import (
    route_triage,
    track_a_discovery_node,
    track_b_discovery_node,
    track_c_discovery_node,
    triage_fallback_node,
    triage_node,
)
from .fetch_target import fetch_target_company
from .hitl import (
    hitl_breakpoint_node,
    iterative_research_node,
    rolling_summarization_node,
    route_hitl,
)
from .models import IntelligenceState
from .sanitization import sanitization_node


def graph(checkpointer=None):
    workflow = StateGraph(IntelligenceState)

    # --- Nodes ---
    workflow.add_node("sanitize", sanitization_node)
    workflow.add_node("anchor", domain_anchoring_node)
    workflow.add_node("scrape_target", fetch_target_company)

    # Parallel discovery 
    workflow.add_node("discovery_a", track_a_discovery_node)
    workflow.add_node("discovery_b", track_b_discovery_node)
    workflow.add_node("discovery_c", track_c_discovery_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("triage_fallback", triage_fallback_node)

    # Parallel competitor extraction via Send API 
    workflow.add_node("extract_competitor", extract_competitor_node)
    # Bounded remediation + Known Unknown fallback
    workflow.add_node("quality_gate", quality_gate_node)

    # rolling summarisation -> synthesis -> HITL loop
    workflow.add_node("summarize", rolling_summarization_node)
    workflow.add_node("synthesize", analyze_and_compare)
    workflow.add_node("hitl_breakpoint", hitl_breakpoint_node)
    workflow.add_node("research", iterative_research_node)

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

    # Fan-in: all three discovery tracks converge at triage
    workflow.add_edge("discovery_a", "triage")
    workflow.add_edge("discovery_b", "triage")
    workflow.add_edge("discovery_c", "triage")

    # Triage routing: fallback loop, graceful skip, or parallel Send fan-out
    workflow.add_conditional_edges(
        "triage",
        route_triage,
        ["triage_fallback", "extract_competitor", "quality_gate"],
    )
    workflow.add_edge("triage_fallback", "triage")

    # Fan-in: all parallel extract_competitor results merge into quality_gate
    workflow.add_edge("extract_competitor", "quality_gate")

    # rolling summarisation -> synthesis -> HITL breakpoint
    workflow.add_edge("quality_gate", "summarize")
    workflow.add_edge("summarize", "synthesize")
    workflow.add_edge("synthesize", "hitl_breakpoint")

    # HITL routing: approved/limit-hit -> END; feedback -> research
    workflow.add_conditional_edges(
        "hitl_breakpoint",
        route_hitl,
        {"research": "research", "end": END},
    )

    # HITL loop: research findings -> summarise -> re-synthesise
    workflow.add_edge("research", "summarize")

    # interrupt_before is NOT used — interrupt() inside hitl_breakpoint_node
    # owns the pause natively; a checkpointer is still required for state persistence.
    return workflow.compile(checkpointer=checkpointer)
