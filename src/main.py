import sqlite3
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from utils.config import SQLITE_DB_PATH
from utils.graph_builder import graph
from utils.models import FinalReport


# ---------------------------------------------------------------------------
# Report display helpers
# ---------------------------------------------------------------------------

def _print_report_body(report: FinalReport) -> None:
    """Shared body: SWOT, competitor comparisons, data-quality warnings."""
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
            print(f"      Data confidence:   {comp.data_confidence}")

    if report.stale_data_warnings:
        print("\nStale Data Warnings")
        print("-" * 70)
        for w in report.stale_data_warnings:
            print(f"  ! {w}")

    if report.known_unknowns:
        print("\nKnown Unknowns")
        print("-" * 70)
        for ku in report.known_unknowns:
            print(f"  ? {ku}")

    print(f"\n[Synthesis model: {report.synthesis_model_used}]")


def _print_draft(report: FinalReport, competitors_data: dict) -> None:
    """
    Print a HITL draft. Surfaces stale_data_warnings, known_unknowns, and
    conflict_fields (Track A vs B disagreements) so the reviewer has full
    context before submitting feedback.
    """
    print()
    print("=" * 70)
    print(f"  DRAFT v{report.report_version} — Competitive Intelligence Report")
    print(f"  Target: {report.target_company}")
    print("=" * 70)
    _print_report_body(report)

    conflict_items = [
        f"{profile.name}: {', '.join(profile.conflict_fields)}"
        for profile in competitors_data.values()
        if profile.conflict_fields
    ]
    if conflict_items:
        print("\nConflict Fields (Track A vs B — Track A pricing used)")
        print("-" * 70)
        for item in conflict_items:
            print(f"  ~ {item}")


def _print_final(report: FinalReport) -> None:
    """Print the final approved report."""
    print()
    print("=" * 70)
    print(f"  FINAL v{report.report_version} — Competitive Intelligence Report")
    print(f"  Target: {report.target_company}")
    print("=" * 70)
    _print_report_body(report)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    target_website = input("Enter the URL of the company you want to analyze: ")

    # SqliteSaver persists every node checkpoint to disk — a process restart with
    # the same thread_id resumes from the last completed node rather than scratch.
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    try:
        checkpointer = SqliteSaver(conn)
        app = graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        initial_state = {
            "target_company_url": target_website,
            "target_profile": None,
            "discovery_tracks": {},
            "triage_fallback_attempts": 0,
            "competitor_urls": [],
            "competitors_data": {},
            "final_report": None,
            "historical_context": [],
            "remediation_context": [],
            "user_feedback": None,
            "hitl_iteration_count": 0,
        }

        # First run — graph halts inside hitl_breakpoint_node via interrupt()
        try:
            app.invoke(initial_state, config)
        except ValueError as e:
            print(f"Input rejected: {e}")
            sys.exit(1)

        # HITL review loop — each iteration resumes the interrupted node via Command
        while True:
            snap = app.get_state(config)

            if not snap.next:
                # Graph completed (approved or iteration limit reached)
                break

            # Surface the draft report with all data-quality metadata
            report = snap.values.get("final_report")
            competitors_data = snap.values.get("competitors_data") or {}

            if report:
                _print_draft(report, competitors_data)

            print()
            feedback = input("Feedback (or 'APPROVED' to finalise): ").strip()

            # Resume the interrupted node — feedback becomes the return value of
            # interrupt() inside hitl_breakpoint_node, which writes it to user_feedback
            app.invoke(Command(resume=feedback), config)

        # Print the final approved report
        final_report = app.get_state(config).values.get("final_report")
        if final_report is None:
            print("\nNo report produced.")
        else:
            _print_final(final_report)
    finally:
        conn.close()
