"""Smoke tests for the PharmaOS CLI."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

import pharma_os.orchestrator as orchestrator_module
from pharma_os.cli import main
from pharma_os.schemas import WorkflowRun


class FakeWorkflowOutput(BaseModel):
    run_id: str
    output_id: str
    validation_status: str = "passed"


def test_top_level_help_includes_orchestrate(capsys):
    """The source CLI exposes the Control Tower orchestration command."""

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "orchestrate" in capsys.readouterr().out


def test_unknown_workflow_does_not_complete_successfully(capsys, tmp_path):
    """Unknown workflows return a clear CLI error instead of a completed placeholder."""

    exit_code = main(["run", "demo-workflow", "--db-path", str(tmp_path / "memory.sqlite")])

    assert exit_code == 2
    assert "Unknown workflow: demo-workflow" in capsys.readouterr().out


def test_run_command_writes_default_json_and_html(capsys, tmp_path, monkeypatch):
    """Direct workflow runs write viewable artifacts under outputs/ by default."""

    monkeypatch.chdir(tmp_path)

    def fake_runner(input_data, *, memory):
        run_id = "direct-run-123"
        output = FakeWorkflowOutput(run_id=run_id, output_id="direct-output-123")
        memory.save_run(
            WorkflowRun(
                run_id=run_id,
                workflow_name="clinical_outcome_prediction",
                status="completed",
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                input_provenance="fake",
                validation_status="passed",
                metadata={"nct_id": input_data.nct_id},
            ),
            input_payload=input_data,
            output_payload=output,
        )
        return output

    monkeypatch.setitem(orchestrator_module._WORKFLOW_RUNNERS, "clinical_outcome_prediction", fake_runner)

    exit_code = main(
        [
            "run",
            "clinical_outcome_prediction",
            "--nct-id",
            "NCT12345678",
            "--db-path",
            str(tmp_path / "memory.sqlite"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Run completed" in stdout
    assert "json:" in stdout
    output_dir = tmp_path / "outputs" / "clinical_outcome_prediction_direct-run-123"
    json_path = output_dir / "clinical_outcome_prediction_direct-run-123.json"
    html_path = output_dir / "clinical_outcome_prediction_direct-run-123.html"
    assert json_path.exists()
    assert html_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "direct-run-123"
    assert payload["output_id"] == "direct-output-123"
    assert "PharmaOS Run direct-run-123" in html_path.read_text(encoding="utf-8")


def test_report_command_outputs_report(capsys, tmp_path, monkeypatch):
    """The report command emits a report for the requested run id."""

    monkeypatch.chdir(tmp_path)

    exit_code = main(
        ["report", "--run-id", "RUN123", "--db-path", str(tmp_path / "memory.sqlite")]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Report generated" in stdout
    json_path = tmp_path / "outputs" / "report_RUN123" / "report_RUN123.json"
    output = json.loads(json_path.read_text(encoding="utf-8"))
    assert output["run_id"] == "RUN123"
    assert output["title"] == "PharmaOS report for RUN123"
    assert output["validation_status"] == "warning"
    assert json_path.exists()
    assert (tmp_path / "outputs" / "report_RUN123" / "report_RUN123.html").exists()
