from __future__ import annotations

from datetime import datetime, timezone

from pharma_os.html_report import build_run_html, write_run_html
from pharma_os.memory import MemoryStore
from pharma_os.schemas import AgentRunTrace, WorkflowRun


def test_html_generation_for_saved_run(tmp_path) -> None:
    store = MemoryStore(":memory:")
    run = WorkflowRun(
        run_id="RUNHTML",
        workflow_name="trial_intelligence",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        input_provenance="test",
        validation_status="passed",
    )
    store.save_run(run, input_payload={"disease": "glioblastoma"}, output_payload={"output_id": "OUT"}, trace_metadata={"mode": "test"})
    store.save_agent_trace(
        AgentRunTrace(
            trace_id="TRACE1",
            run_id="RUNHTML",
            agent_name="FixtureAgent",
            output_type="FixtureOutput",
            provenance="test",
            execution_mode="deterministic_fallback",
            model="gpt-test",
            model_route="control_tower",
            retry_count=2,
            retry_attempts=3,
            retry_exhausted=True,
            fallback_cause="rate_limit",
            final_retry_reason="429",
        )
    )

    html = build_run_html("RUNHTML", memory=store)
    assert "Run Metadata" in html
    assert "Input JSON" in html
    assert "Trace Metadata" in html
    assert "glioblastoma" in html
    assert "Runtime Routes And Retries" in html
    assert "control_tower" in html
    assert "gpt-test" in html
    assert "rate_limit" in html

    output_path = write_run_html("RUNHTML", tmp_path / "run.html", memory=store)
    assert output_path.exists()
    assert "Raw Bundle JSON" in output_path.read_text(encoding="utf-8")
