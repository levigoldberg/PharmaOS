"""Smoke tests for the PharmaOS CLI."""

from __future__ import annotations

import json

from pharma_os.cli import main


def test_run_command_outputs_completed_workflow(capsys, tmp_path):
    """The run command emits a completed workflow run."""

    exit_code = main(["run", "demo-workflow", "--db-path", str(tmp_path / "memory.sqlite")])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["workflow_name"] == "demo-workflow"
    assert output["status"] == "completed"
    assert output["run_id"]


def test_report_command_outputs_report(capsys, tmp_path):
    """The report command emits a report for the requested run id."""

    exit_code = main(
        ["report", "--run-id", "RUN123", "--db-path", str(tmp_path / "memory.sqlite")]
    )

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["run_id"] == "RUN123"
    assert output["title"] == "PharmaOS report for RUN123"
    assert output["validation_status"] == "warning"
