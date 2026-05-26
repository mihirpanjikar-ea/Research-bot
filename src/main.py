from dotenv import load_dotenv

load_dotenv()

from utils.graph_builder import graph
from utils.models import FinalReport


def _print_report(report: FinalReport) -> None:
    print()
    print("=" * 70)
    print(f"  Competitive Intelligence Report")
    print(f"  Target: {report.target_company}")
    print("=" * 70)

    print("\nExecutive Summary")
    print("-" * 70)
    print(report.executive_summary)

    print("\nSWOT")
    print("-" * 70)
    print(f"  Strengths:     {report.swot.strengths}")
    print(f"  Weaknesses:    {report.swot.weaknesses}")
    print(f"  Opportunities: {report.swot.opportunities}")
    print(f"  Threats:       {report.swot.threats}")

    if report.competitor_comparisons:
        print("\nCompetitor Comparisons")
        print("-" * 70)
        for comp in report.competitor_comparisons:
            print(f"  - {comp.name}  ({comp.url})")
            print(f"      Overlapping:       {comp.overlapping_features}")
            print(f"      Differentiators:   {comp.differentiators}")
            print(f"      Target advantages: {comp.target_advantages}")
            print(f"      Pricing delta:     {comp.pricing_delta}")
    print()


if __name__ == "__main__":
    app = graph()

    target_website = input("Enter the URL of the company you want to analyze: ")

    initial_state = {
        "target_company_url": target_website,
        "target_profile": None,
        "discovery_tracks": {},
        "triage_fallback_attempts": 0,
        "competitor_urls": [],
        "competitors_data": {},
        "final_report": None,
    }

    result = app.invoke(initial_state)

    report = result.get("final_report")
    if report is None:
        print("\nNo report produced.")
    else:
        _print_report(report)
