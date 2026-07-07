"""Report generation for PharmaOS workflow runs."""

from __future__ import annotations

from pharma_os.schemas import Report


def build_report(run_id: str) -> Report:
    """Build a minimal report for a run id."""

    return Report(
        run_id=run_id,
        title=f"PharmaOS report for {run_id}",
        content="No persisted workflow details are available yet.",
    )
