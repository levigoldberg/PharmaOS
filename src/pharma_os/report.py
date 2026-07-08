"""Report generation for PharmaOS workflow runs."""

from __future__ import annotations

from pharma_os.schemas import FinalReport


def build_report(run_id: str) -> FinalReport:
    """Build a minimal report for a run id."""

    return FinalReport(
        report_id=f"report-{run_id}",
        run_id=run_id,
        title=f"PharmaOS report for {run_id}",
        summary="No persisted workflow details are available yet.",
        confidence=0.0,
        validation_status="warning",
        provenance="cli.report.placeholder",
    )
