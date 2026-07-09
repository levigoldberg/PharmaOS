"""Smoke tests for the PharmaOS CLI."""

from __future__ import annotations

import json

from pharma_os.cli import main


def test_unknown_workflow_does_not_complete_successfully(capsys, tmp_path):
    """Unknown workflows return a clear CLI error instead of a completed placeholder."""

    exit_code = main(["run", "demo-workflow", "--db-path", str(tmp_path / "memory.sqlite")])

    assert exit_code == 2
    assert "Unknown workflow: demo-workflow" in capsys.readouterr().out


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
