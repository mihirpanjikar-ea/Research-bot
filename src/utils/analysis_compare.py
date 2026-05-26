from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import FinalReport, IntelligenceState

synthesis_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def analyze_and_compare(state: IntelligenceState) -> dict:
    """Single synthesis pass: produce a structured FinalReport from target + competitor profiles."""
    print("--- Synthesising final report ---")

    target = state.get("target_profile")
    competitors_dict = state.get("competitors_data") or {}
    competitors = list(competitors_dict.values())

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a senior strategic analyst. Generate a competitive intelligence report containing: "
         "an executive summary, a SWOT analysis for the target, and a comparison entry for each competitor "
         "(overlapping features, differentiators, target advantages, pricing delta). "
         "Be objective and concise."),
        ("user",
         "Target Company Profile:\n{target_profile}\n\n"
         "Competitor Profiles:\n{competitors_data}\n"),
    ])

    chain = prompt | synthesis_llm.with_structured_output(FinalReport)

    target_input = target.model_dump() if target else "No target profile available"
    comps_input = [c.model_dump() for c in competitors]

    report = chain.invoke({
        "target_profile": target_input,
        "competitors_data": comps_input,
    })

    return {"final_report": report}
