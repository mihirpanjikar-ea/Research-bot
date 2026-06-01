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
        thread_id = str(uuid.uuid4())
        base_config = {"configurable": {"thread_id": thread_id}}

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

        # First run — graph halts inside hitl_breakpoint_node via interrupt().
        # Tag/metadata flow into LangSmith so the trace is searchable.
        initial_config = {
            **base_config,
            "tags": ["research_bot", "initial_pass"],
            "metadata": {
                "thread_id": thread_id,
                "target_url": target_website,
                "phase": "initial",
            },
        }
        try:
            app.invoke(initial_state, initial_config)
        except ValueError as e:
            print(f"Input rejected: {e}")
            sys.exit(1)

        # HITL review loop — each iteration resumes the interrupted node via Command
        hitl_pass = 0
        while True:
            snap = app.get_state(base_config)

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
            hitl_pass += 1

            # Resume the interrupted node — feedback becomes the return value of
            # interrupt() inside hitl_breakpoint_node, which writes it to user_feedback.
            # Attach feedback + iteration as run metadata so the LangSmith trace
            # shows the input that bridged this pair of invocations.
            resume_config = {
                **base_config,
                "tags": ["research_bot", "hitl_resume", f"hitl_pass_{hitl_pass}"],
                "metadata": {
                    "thread_id": thread_id,
                    "target_url": target_website,
                    "phase": "hitl_resume",
                    "hitl_pass": hitl_pass,
                    "hitl_feedback": feedback,
                },
            }
            try:
                app.invoke(Command(resume=feedback), resume_config)
            except Exception as e:
                print(f"Error during research iteration: {e}")
                break

        # Print the final approved report
        final_report = app.get_state(base_config).values.get("final_report")
        if final_report is None:
            print("\nNo report produced.")
        else:
            _print_final(final_report)
    finally:
        conn.close()
