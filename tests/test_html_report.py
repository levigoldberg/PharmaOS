from __future__ import annotations

from datetime import datetime, timezone

from pharma_os.html_report import build_run_html, write_run_html
from pharma_os.memory import MemoryStore
from pharma_os.schemas import WorkflowRun


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

    html = build_run_html("RUNHTML", memory=store)
    assert "Run Metadata" in html
    assert "Input JSON" in html
    assert "Trace Metadata" in html
    assert "glioblastoma" in html

    output_path = write_run_html("RUNHTML", tmp_path / "run.html", memory=store)
    assert output_path.exists()
    assert "Raw Bundle JSON" in output_path.read_text(encoding="utf-8")
