"""Report generation from Scientific Memory."""

from __future__ import annotations

from pharma_os.memory import MemoryStore
from pharma_os.schemas import FinalReport
from pharma_os.validators import aggregate_validation_status


def build_report(run_id: str, memory: MemoryStore | None = None) -> FinalReport:
    """Build and persist a report for a run ID."""

    store = memory or MemoryStore()
    bundle = store.get_run_bundle(run_id)
    if bundle.run is None:
        return FinalReport(
            report_id=f"report-{run_id}",
            run_id=run_id,
            title=f"PharmaOS report for {run_id}",
            summary="No persisted workflow details are available yet.",
            confidence=0.0,
            validation_status="warning",
            provenance="pharma_os.report.placeholder",
        )

    validation_status = aggregate_validation_status(bundle.validation_results)
    latest_gate = bundle.human_gates[-1] if bundle.human_gates else None
    if latest_gate is not None and validation_status == "passed":
        validation_status = "needs_human_review"
    summary = _summary(bundle)
    confidence = _confidence(validation_status, bool(latest_gate), len(bundle.confidence_flags))
    report = FinalReport(
        report_id=f"report-{run_id}",
        run_id=run_id,
        title=f"PharmaOS report for {run_id}",
        summary=summary,
        claims=bundle.claims,
        sources=bundle.sources,
        validation_results=bundle.validation_results,
        confidence_flags=bundle.confidence_flags,
        human_gate=latest_gate,
        confidence=confidence,
        validation_status=validation_status,
        provenance="pharma_os.report.build_report",
    )
    return store.save_report(report)


def _summary(bundle: object) -> str:
    run = bundle.run
    if run is None:
        return "No persisted workflow details are available yet."
    parts = [
        f"Workflow {run.workflow_name} completed with status {run.status}.",
        f"Scientific Memory contains {len(bundle.sources)} sources and {len(bundle.claims)} claims.",
    ]
    if bundle.validation_results:
        parts.append(
            f"{len(bundle.validation_results)} validation results were recorded."
        )
    if bundle.confidence_flags:
        parts.append(f"{len(bundle.confidence_flags)} confidence flags require review.")
    if bundle.human_gates:
        parts.append("A human review gate is open.")
    return " ".join(parts)


def _confidence(validation_status: str, has_gate: bool, flag_count: int) -> float:
    if validation_status == "failed":
        return 0.25
    if has_gate:
        return 0.4
    if validation_status == "warning":
        return 0.65
    return max(0.5, 0.9 - min(flag_count, 5) * 0.05)
