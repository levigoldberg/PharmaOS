from __future__ import annotations

from datetime import datetime, timezone

from pharma_os.memory import MemoryStore
from pharma_os.schemas import EvidenceClaim, SourceMetadata, WorkflowRun


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

    store.save_run(run)
    store.save_sources(run.run_id, (source,))
    store.save_claims(run.run_id, (claim,))
    bundle = store.get_run_bundle("RUN")

    assert bundle.run is not None
    assert bundle.run.workflow_name == "trial_intelligence"
    assert bundle.sources[0].source_id == source.source_id
    assert bundle.claims[0].claim_id == claim.claim_id
