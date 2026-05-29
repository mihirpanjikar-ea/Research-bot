from .models import FinalReport


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
