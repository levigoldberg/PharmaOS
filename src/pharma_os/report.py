"""Report generation from Scientific Memory."""

from __future__ import annotations

from pharma_os.execution_modes import reused_artifacts_from_output, summarize_execution_modes
from pharma_os.memory import MemoryStore
from pharma_os.review_flags import canonical_review_flags, review_flag_summary
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
    execution_mode_summary = summarize_execution_modes(
        bundle.agent_traces,
        reused_artifacts=reused_artifacts_from_output(bundle.output_json),
    )
    if execution_mode_summary.requested_reasoning_steps == 0 and isinstance(bundle.output_json, dict):
        output_summary = bundle.output_json.get("execution_mode_summary")
        if isinstance(output_summary, dict):
            from pharma_os.schemas import ExecutionModeSummary

            execution_mode_summary = ExecutionModeSummary.model_validate(output_summary)
    review_flags = canonical_review_flags(
        run_id=run_id,
        validation_results=bundle.validation_results,
        human_gate=latest_gate,
        confidence_flags=bundle.confidence_flags,
    )
    summary = _summary(bundle, execution_mode_summary.summary, review_flag_summary(review_flags))
    confidence = _confidence(validation_status, bool(latest_gate), review_flags)
    report = FinalReport(
        report_id=f"report-{run_id}",
        run_id=run_id,
        title=f"PharmaOS report for {run_id}",
        summary=summary,
        claims=bundle.claims,
        sources=bundle.sources,
        validation_results=bundle.validation_results,
        confidence_flags=review_flags,
        human_gate=latest_gate,
        confidence=confidence,
        validation_status=validation_status,
        provenance="pharma_os.report.build_report",
        execution_mode_summary=execution_mode_summary,
    )
    return store.save_report(report)


def _summary(bundle: object, execution_mode_summary: str, review_summary: str) -> str:
    run = bundle.run
    if run is None:
        return "No persisted workflow details are available yet."
    parts = [
        f"Workflow {run.workflow_name} completed with status {run.status}.",
        f"Scientific Memory contains {len(bundle.sources)} sources and {len(bundle.claims)} claims.",
        execution_mode_summary,
    ]
    if bundle.validation_results:
        parts.append(
            f"{len(bundle.validation_results)} validation results were recorded."
        )
    parts.append(review_summary)
    return " ".join(parts)


def _confidence(validation_status: str, has_gate: bool, review_flags: tuple[object, ...]) -> float:
    if validation_status == "failed":
        return 0.25
    severities = [getattr(flag, "severity", "") for flag in review_flags]
    if "critical" in severities:
        return 0.3
    if "high" in severities:
        return 0.45
    if has_gate:
        return 0.55
    if validation_status == "warning":
        return 0.65
    return max(0.5, 0.9 - min(len(review_flags), 5) * 0.05)
