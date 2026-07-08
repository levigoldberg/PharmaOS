from __future__ import annotations

import json
from pathlib import Path

from pharma_os.agents.clinical_trial_intelligence import deterministic_trial_intelligence_output
from pharma_os.cli import main
from pharma_os.memory import MemoryStore
from pharma_os.schemas import ClinicalTrialIntelligenceInput
from pharma_os.tools.clinicaltrials import ClinicalTrialsGovClient
from pharma_os.workflows.trial_intelligence import run_trial_intelligence_workflow


FIXTURE = Path(__file__).parent / "fixtures" / "clinicaltrials_search_glioblastoma.json"


def _mock_runner(input_data: ClinicalTrialIntelligenceInput, run_id: str):
    import httpx

    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    client = ClinicalTrialsGovClient(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    )
    search_result = client.search_trials(input_data)
    output = deterministic_trial_intelligence_output(
        run_id=run_id,
        input_data=input_data,
        search_result=search_result,
    )
    return output, {"trace_id": "test-trace"}


def test_trial_intelligence_workflow_persists_bundle() -> None:
    store = MemoryStore(":memory:")
    output = run_trial_intelligence_workflow(
        ClinicalTrialIntelligenceInput(disease="glioblastoma", target="EGFR", limit=10),
        memory=store,
        agent_runner=_mock_runner,
    )

    bundle = store.get_run_bundle(output.run_id)

    assert bundle.run is not None
    assert bundle.sources
    assert bundle.claims
    assert bundle.validation_results
    assert bundle.confidence_flags
    assert bundle.agent_outputs
    assert bundle.agent_outputs[0].agent_name == "agent3_trial_landscape_component"
    assert bundle.reports


def test_cli_report_reads_prior_persisted_run(tmp_path, monkeypatch, capsys) -> None:
    from pharma_os.workflows import trial_intelligence

    monkeypatch.setattr(trial_intelligence, "_default_agent_runner", _mock_runner)
    db_path = tmp_path / "memory.sqlite"
    run_json = tmp_path / "run.json"
    report_json = tmp_path / "report.json"

    exit_code = main(
        [
            "run",
            "trial_intelligence",
            "--disease",
            "glioblastoma",
            "--target",
            "EGFR",
            "--db-path",
            str(db_path),
            "--output-json",
            str(run_json),
        ]
    )

    assert exit_code == 0
    run_output = json.loads(run_json.read_text(encoding="utf-8"))
    run_id = run_output["run_id"]

    exit_code = main(
        [
            "report",
            "--run-id",
            run_id,
            "--db-path",
            str(db_path),
            "--output-json",
            str(report_json),
        ]
    )

    assert exit_code == 0
    report_output = json.loads(report_json.read_text(encoding="utf-8"))
    assert report_output["run_id"] == run_id
    assert report_output["sources"]


def test_default_trial_intelligence_runner_uses_agent3_landscape_component(monkeypatch) -> None:
    from pharma_os.agents import clinical_trial_intelligence as cti_agent

    original_search = ClinicalTrialsGovClient.search_trials

    def fake_search(self, input_data):
        import httpx

        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        client = ClinicalTrialsGovClient(
            client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
        )
        return original_search(client, input_data)

    monkeypatch.setattr(cti_agent.ClinicalTrialsGovClient, "search_trials", fake_search)

    output, trace = cti_agent.run_clinical_trial_intelligence_agent(
        ClinicalTrialIntelligenceInput(disease="glioblastoma", target="EGFR", limit=10),
        run_id="test-run",
    )

    assert output.landscape_summary == "Found 2 ClinicalTrials.gov records for glioblastoma."
    assert trace["mode"] == "agent3_trial_landscape_component"
