from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .config import MAX_HISTORY_CHARS, MAX_HITL_ITERATIONS, exa_search, extraction_llm
from .models import IntelligenceState


# ---------------------------------------------------------------------------
# Rolling summarisation
# ---------------------------------------------------------------------------

def rolling_summarization_node(state: IntelligenceState) -> dict:
    """
    Compress historical_context to a single summary message when it
    exceeds MAX_HISTORY_CHARS. Fires before every synthesis pass so both the
    initial run and every HITL re-synthesis start within budget.

    historical_context is a plain List[BaseMessage] with no reducer, so
    returning a single-entry list replaces the prior messages wholesale —
    which is the intended replacement semantics for rolling summarisation.
    """
    messages = state.get("historical_context") or []
    if not messages:
        return {}

    history_text = "\n\n".join(
        str(m.content) if hasattr(m, "content") else str(m) for m in messages
    )

    if len(history_text) <= MAX_HISTORY_CHARS:
        return {}

    print(
        f"--- Rolling summarisation: compressing {len(messages)} message(s) "
        f"({len(history_text)} chars > {MAX_HISTORY_CHARS} limit) ---"
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI assistant specialised in information compression. "
            "Summarise the following historical context while preserving all key "
            "facts, competitor names, and specific data points. "
            "Keep the summary concise but comprehensive.",
        ),
        ("user", "Context to summarise:\n{history}"),
    ])

    chain = prompt | extraction_llm
    try:
        summary = chain.invoke({"history": history_text})
        return {
            "historical_context": [
                AIMessage(content=f"Summary of previous context: {summary.content}")
            ]
        }
    except Exception as e:
        print(f"Rolling summarisation error: {e}")
        return {}


# ---------------------------------------------------------------------------
# HITL breakpoint + routing
# ---------------------------------------------------------------------------

def hitl_breakpoint_node(state: IntelligenceState) -> dict:
    """
    HITL breakpoint using LangGraph's interrupt() primitive.

    interrupt() suspends the graph mid-node and surfaces a payload to the
    caller. When the caller resumes with Command(resume=<feedback>), execution
    continues here and the feedback string is returned by interrupt(). The
    node then writes it to user_feedback so route_hitl can act on it.

    No interrupt_before compile flag needed — interrupt() owns the pause.
    """
    report = state.get("final_report")
    competitors_data = state.get("competitors_data") or {}

    # Build the payload surfaced to the reviewer
    payload = {
        "report_version": report.report_version if report else 1,
        "stale_data_warnings": report.stale_data_warnings if report else [],
        "known_unknowns": report.known_unknowns if report else [],
        "conflict_fields": {
            url: profile.conflict_fields
            for url, profile in competitors_data.items()
            if profile.conflict_fields
        },
    }

    # Pause here — feedback is whatever the caller passes to Command(resume=...)
    feedback = interrupt(payload)

    return {"user_feedback": feedback}


def route_hitl(state: IntelligenceState) -> str:
    """
    Three-way routing from the HITL breakpoint:

    1. No feedback / "APPROVED"        -> "end"      (finalise)
    2. Iteration limit reached         -> "end"      (self-terminate)
    3. Actionable feedback provided    -> "research" (iterate)
    """
    feedback = (state.get("user_feedback") or "").strip()
    if not feedback or feedback.upper() == "APPROVED":
        print("--- HITL: approved, finalising report ---")
        return "end"
    if state.get("hitl_iteration_count", 0) >= MAX_HITL_ITERATIONS:
        print(f"--- HITL: iteration limit ({MAX_HITL_ITERATIONS}) reached, finalising ---")
        return "end"
    return "research"


# ---------------------------------------------------------------------------
# Iterative research node
# ---------------------------------------------------------------------------

class _SearchQueries(BaseModel):
    """Structured output: up to 3 targeted Exa queries derived from feedback."""
    queries: List[str] = Field(default_factory=list)


def iterative_research_node(state: IntelligenceState) -> dict:
    """
    Translates user feedback into targeted Exa search queries, runs each
    query, and appends the findings to remediation_context.

    remediation_context uses the add_messages reducer, so each cycle's
    findings accumulate rather than replace prior research. synthesis will
    consume the full accumulated context on the next pass.

    Clears user_feedback and increments hitl_iteration_count before returning
    so the next interrupt cycle requires fresh input.
    """
    feedback = state.get("user_feedback") or ""
    if not feedback.strip() or feedback.strip().upper() == "APPROVED":
        return {"user_feedback": None}

    iteration = state.get("hitl_iteration_count", 0)
    print(
        f"--- HITL research: iteration {iteration + 1} | "
        f"feedback: {feedback[:80]}{'...' if len(feedback) > 80 else ''} ---"
    )

    query_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI research assistant. Extract up to 3 precise Exa search "
            "queries from the user feedback to find missing competitive intelligence. "
            "Return only queries directly answerable by web search.",
        ),
        ("user", "User Feedback:\n{feedback}"),
    ])
    query_chain = query_prompt | extraction_llm.with_structured_output(_SearchQueries)

    try:
        search_queries_result = query_chain.invoke({"feedback": feedback})
        if isinstance(search_queries_result, dict):
            search_queries = _SearchQueries(**search_queries_result)
        else:
            search_queries = search_queries_result
    except Exception as e:
        print(f"Query generation failed: {e}")
        search_queries = _SearchQueries()

    new_context_parts: List[str] = []
    for query in search_queries.queries:
        try:
            res = exa_search.invoke({
                "query": query,
                "num_results": 2,
                "type": "auto",
                "text_contents_options": True,
            })
            for item in res.results:
                new_context_parts.append(
                    f"URL: {item.url}\nContent:\n{getattr(item, 'text', '')[:1000]}..."
                )
        except Exception as e:
            print(f"Research search error for '{query}': {e}")

    full_context = (
        "\n\n---\n\n".join(new_context_parts)
        if new_context_parts
        else "No results found."
    )

    return {
        "remediation_context": [
            HumanMessage(
                content=(
                    f"HITL Feedback (iteration {iteration + 1}): {feedback}\n\n"
                    f"Research Findings:\n{full_context}"
                )
            )
        ],
        "hitl_iteration_count": iteration + 1,
        "user_feedback": None,
    }
