"""
Thin observability shim — every node-level status print is mirrored into
the current LangSmith run tree as a tag + metadata so the same events that
appear in the terminal are also searchable in the trace UI.

Falls back gracefully (no-op on the LangSmith side) when called outside a
traced context, e.g. during CLI startup before the graph runs.
"""

from typing import Any

from langsmith.run_helpers import get_current_run_tree


def log_event(message: str, **metadata: Any) -> None:
    """
    Print to stdout AND annotate the current LangSmith run tree.

    - The terminal still sees the human-readable line (unchanged behaviour).
    - LangSmith receives an `event:<message>` tag plus any structured kwargs
      as metadata on the active node run, so the same signal is searchable
      and visible in the trace UI without polluting prompt context.
    """
    if metadata:
        print(f"{message} | {metadata}")
    else:
        print(message)

    rt = get_current_run_tree()
    if rt is None:
        return

    try:
        rt.add_tags([f"event:{message[:60]}"])
        if metadata:
            rt.add_metadata(metadata)
    except Exception:
        # Never let observability break the graph
        pass


def log_swallowed_exception(where: str, exc: BaseException, **metadata: Any) -> None:
    """
    Special-case helper for try/except blocks that fall back to defaults.
    Surfaces the exception in LangSmith so silent failures are searchable
    by tag (`swallowed_exception`) and the message is preserved as metadata.
    """
    print(f"{where} error: {exc}")

    rt = get_current_run_tree()
    if rt is None:
        return

    try:
        rt.add_tags(["swallowed_exception", f"swallowed_at:{where[:50]}"])
        rt.add_metadata({
            "swallowed_exception": {
                "where": where,
                "type": type(exc).__name__,
                "message": str(exc),
                **metadata,
            },
        })
    except Exception:
        pass
