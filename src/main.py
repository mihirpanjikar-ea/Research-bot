import sqlite3
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from utils.config import SQLITE_DB_PATH
from utils.graph_builder import graph
from utils.report_display import _print_draft, _print_final


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
            try:
                app.invoke(Command(resume=feedback), config)
            except Exception as e:
                print(f"Error during research iteration: {e}")
                break

        # Print the final approved report
        final_report = app.get_state(config).values.get("final_report")
        if final_report is None:
            print("\nNo report produced.")
        else:
            _print_final(final_report)
    finally:
        conn.close()
