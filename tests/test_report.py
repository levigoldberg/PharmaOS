from __future__ import annotations

from datetime import datetime, timezone

from pharma_os.memory import MemoryStore
from pharma_os.report import build_report
from pharma_os.schemas import AgentRunTrace, EvidenceClaim, SourceMetadata, WorkflowRun


def test_report_reads_persisted_run() -> None:
    store = MemoryStore(":memory:")
    run = WorkflowRun(
        run_id="RUN",
        workflow_name="trial_intelligence",
        status="completed",
        started_at=datetime.now(timezone.utc),
        input_provenance="test",
        validation_status="passed",
    )
    source = SourceMetadata(
        source_id="ctgov:NCT01234567",
        title="Trial",
        url="https://clinicaltrials.gov/study/NCT01234567",
        provenance="test",
    )
    claim = EvidenceClaim(
        claim_id="claim-1",
        claim_text="NCT01234567 has status RECRUITING.",
        source_ids=(source.source_id,),
        provenance="test",
        confidence=0.9,
        confidence_level="high",
    )
    store.save_run(run)
    store.save_sources(run.run_id, (source,))
    store.save_claims(run.run_id, (claim,))
    store.save_agent_trace(
        AgentRunTrace(
            trace_id="trace-live",
            run_id=run.run_id,
            agent_name="LiveAgent",
            input_summary="live",
            output_id="OUT-LIVE",
            output_type="Fixture",
            output_summary="live output",
            provenance="test.live",
            execution_mode="live_agent",
        )
    )
    store.save_agent_trace(
        AgentRunTrace(
            trace_id="trace-fallback",
            run_id=run.run_id,
            agent_name="FallbackAgent",
            input_summary="fallback",
            output_id="OUT-FALLBACK",
            output_type="Fixture",
            output_summary="fallback output",
            provenance="test.fallback",
            execution_mode="deterministic_fallback",
        )
    )

    report = build_report("RUN", memory=store)

    assert report.run_id == "RUN"
    assert report.sources[0].source_id == source.source_id
    assert report.claims[0].claim_id == claim.claim_id
    assert report.execution_mode_summary.requested_reasoning_steps == 2
    assert report.execution_mode_summary.live_ai_calls_completed == 1
    assert report.execution_mode_summary.deterministic_fallbacks_used == 1
    assert "2 reasoning steps requested" in report.summary


def test_report_unknown_run_is_placeholder() -> None:
    report = build_report("MISSING", memory=MemoryStore(":memory:"))

    assert report.validation_status == "warning"
    assert "No persisted workflow" in report.summary
