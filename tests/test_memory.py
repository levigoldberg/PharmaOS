from __future__ import annotations

from datetime import datetime, timezone

from pharma_os.memory import MemoryStore
from pharma_os.schemas import AgentRunTrace, EvidenceClaim, SourceMetadata, WorkflowRun


def test_memory_round_trips_run_bundle() -> None:
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
        source_type="clinical_trial_registry",
    )
    claim = EvidenceClaim(
        claim_id="claim-1",
        claim_text="NCT01234567 has status RECRUITING.",
        source_ids=(source.source_id,),
        provenance="test",
        confidence=0.9,
        confidence_level="high",
    )

    trace = AgentRunTrace(
        trace_id="trace-1",
        run_id=run.run_id,
        agent_name="test_agent",
        input_summary="Test input.",
        output_id="output-1",
        output_type="TestOutput",
        output_summary="Test output.",
        source_ids=(source.source_id,),
        confidence=0.8,
        rationale_summary="Validated fixture output.",
        provenance="test",
    )

    store.save_run(
        run,
        input_payload={"disease": "glioblastoma"},
        output_payload={"output_id": "output-1"},
        trace_metadata={"trace_id": "trace-1"},
    )
    store.save_sources(run.run_id, (source,))
    store.save_claims(run.run_id, (claim,))
    store.save_agent_trace(trace)
    bundle = store.get_run_bundle("RUN")

    assert bundle.run is not None
    assert bundle.run.workflow_name == "trial_intelligence"
    assert bundle.input_json == {"disease": "glioblastoma"}
    assert bundle.output_json == {"output_id": "output-1"}
    assert bundle.trace_metadata_json == {"trace_id": "trace-1"}
    assert bundle.sources[0].source_id == source.source_id
    assert bundle.claims[0].claim_id == claim.claim_id
    assert bundle.agent_traces[0].trace_id == "trace-1"
