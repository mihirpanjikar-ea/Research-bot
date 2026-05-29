from typing import List

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from .config import synthesis_llm
from .models import FinalReport, IntelligenceState


def analyze_and_compare(state: IntelligenceState) -> dict:
    """
    Synthesis pass: produce a structured FinalReport from target + competitor profiles.

    injects data-quality metadata (stale warnings, known unknowns,
    conflict fields) into the prompt; stamps quality fields onto the
    report; tracks which model executed via include_raw=True.

    reads historical_context (prior draft summaries) and
    remediation_context (HITL research findings) and incorporates
    them into the synthesis prompt. Appends a draft summary entry
    back to historical_context for rolling summarisation. Sets
    report_version from hitl_iteration_count.
    """
    print("--- Synthesising final report ---")

    target = state.get("target_profile")
    competitors_dict = state.get("competitors_data") or {}
    competitors = list(competitors_dict.values())

    # ------------------------------------------------------------------
    # historical and remediation context
    # ------------------------------------------------------------------
    messages = state.get("historical_context") or []
    history = "\n\n".join(
        str(m.content) if hasattr(m, "content") else str(m) for m in messages
    )

    remediation_messages = state.get("remediation_context") or []
    remediation_text = "\n\n".join(
        str(m.content) if hasattr(m, "content") else str(m) for m in remediation_messages
    )

    # ------------------------------------------------------------------
    # data-quality context
    # ------------------------------------------------------------------
    stale_warnings: List[str] = []
    known_unknowns: List[str] = []

    if target:
        if target.stale_fields:
            stale_warnings.append(
                f"Target ({target.name}) stale fields: {', '.join(target.stale_fields)}"
            )
        if target.pricing_model == "Known Unknown":
            known_unknowns.append(f"Target ({target.name}) pricing is unknown.")
        if not target.key_features or target.key_features == ["Known Unknown"]:
            known_unknowns.append(f"Target ({target.name}) features are unknown.")

    for comp in competitors:
        if comp.stale_fields:
            stale_warnings.append(
                f"Competitor ({comp.name}) stale fields: {', '.join(comp.stale_fields)}"
            )
        if comp.conflict_fields:
            stale_warnings.append(
                f"Competitor ({comp.name}) conflicting fields (Track A wins): "
                f"{', '.join(comp.conflict_fields)}"
            )
        if comp.pricing_model == "Known Unknown":
            known_unknowns.append(f"Competitor ({comp.name}) pricing is unknown.")
        if not comp.key_features or comp.key_features == ["Known Unknown"]:
            known_unknowns.append(f"Competitor ({comp.name}) features are unknown.")

    stale_warnings_text = (
        "\n".join(stale_warnings) if stale_warnings else "None identified."
    )
    known_unknowns_text = (
        "\n".join(known_unknowns) if known_unknowns else "None identified."
    )

    # ------------------------------------------------------------------
    # Synthesis prompt
    # ------------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a senior strategic analyst. Generate a competitive intelligence "
            "report containing: an executive summary, a SWOT analysis for the target, "
            "and a comparison entry for each competitor (overlapping features, "
            "differentiators, target advantages, pricing delta, data_confidence). "
            "Set data_confidence to 'known_unknown' where pricing or features are "
            "unknown, 'low' where stale or conflicting data was found, and 'high' "
            "otherwise. Incorporate feedback from prior review cycles and additional "
            "research context when present. Explicitly call out Known Unknowns and "
            "Stale Data. Be objective and concise.",
        ),
        (
            "user",
            "Target Company Profile:\n{target_profile}\n\n"
            "Competitor Profiles:\n{competitors_data}\n\n"
            "Stale Data Warnings:\n{stale_warnings_text}\n\n"
            "Known Unknowns:\n{known_unknowns_text}\n\n"
            "Historical Context (prior drafts):\n{history}\n\n"
            "Additional Research Context (HITL findings):\n{remediation_text}\n",
        ),
    ])

    chain = prompt | synthesis_llm.with_structured_output(FinalReport, include_raw=True)

    target_input = target.model_dump() if target else "No target profile available"
    comps_input = [c.model_dump() for c in competitors]

    response = chain.invoke({
        "target_profile": target_input,
        "competitors_data": comps_input,
        "stale_warnings_text": stale_warnings_text,
        "known_unknowns_text": known_unknowns_text,
        "history": history or "None.",
        "remediation_text": remediation_text or "None.",
    })

    report: FinalReport = response["parsed"]
    raw_msg = response["raw"]

    if report is None:
        raise RuntimeError(
            "Synthesis LLM failed to produce a valid FinalReport — structured output "
            f"returned None. Raw response: "
            f"{raw_msg.content[:500] if hasattr(raw_msg, 'content') else raw_msg}"
        )

    # ------------------------------------------------------------------
    # stamp data-quality fields
    # ------------------------------------------------------------------
    report.stale_data_warnings = stale_warnings
    report.known_unknowns = known_unknowns
    report.synthesis_model_used = raw_msg.response_metadata.get("model_name", "gpt-4o")

    # ------------------------------------------------------------------
    # stamp version; append draft entry to historical_context
    # ------------------------------------------------------------------
    report.report_version = state.get("hitl_iteration_count", 0) + 1

    history_entry = AIMessage(
        content=f"Draft v{report.report_version}: {report.executive_summary}"
    )

    print(f"--- Report v{report.report_version} synthesised using {report.synthesis_model_used} ---")
    return {
        "final_report": report,
        "historical_context": messages + [history_entry],
    }
