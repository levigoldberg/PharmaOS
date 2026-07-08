from __future__ import annotations

import json

from pharma_os import schemas
from pharma_os.agents import clinical_outcome_prediction as cop_agent
from pharma_os.cli import main
from pharma_os.memory import MemoryStore
from pharma_os.schemas import (
    AssetIdentityOutput,
    ClinicalOutcomePredictionInput,
    ClinicalTrialsSearchResult,
    ClinicalTrialRecord,
    LabelExpansionClinicalRationale,
    PoSOutput,
    SafetyContext,
    SourceMetadata,
    TrialEndpoint,
    TrialIntervention,
    TrialLocation,
    TrialSponsor,
)
from pharma_os.workflows.clinical_outcome_prediction import run_clinical_outcome_prediction_workflow


def _trial() -> ClinicalTrialRecord:
    return ClinicalTrialRecord(
        nct_id="NCT12345678",
        brief_title="Example phase 2 glioblastoma trial",
        official_title="A Phase 2 Study of Examplemab in Glioblastoma",
        overall_status="RECRUITING",
        phases=("PHASE2",),
        study_type="INTERVENTIONAL",
        conditions=("Glioblastoma",),
        interventions=(TrialIntervention(name="Examplemab", type="DRUG", description="monoclonal antibody"),),
        lead_sponsor=TrialSponsor(name="Example Bio", sponsor_class="INDUSTRY"),
        enrollment_count=42,
        enrollment_type="ESTIMATED",
        start_date="2025-01",
        primary_completion_date="2027-01",
        completion_date="2027-06",
        primary_endpoints=(TrialEndpoint(measure="Progression-free survival", endpoint_type="primary"),),
        secondary_endpoints=(TrialEndpoint(measure="Overall survival", endpoint_type="secondary"),),
        locations=(TrialLocation(facility="Example Site", city="Boston", state="MA", country="United States"),),
        eligibility_criteria="Adults with recurrent glioblastoma.",
        source_id="ctgov:NCT12345678",
    )


def _comparator_trial() -> ClinicalTrialRecord:
    return ClinicalTrialRecord(
        nct_id="NCT87654321",
        brief_title="Comparator glioblastoma trial",
        overall_status="COMPLETED",
        phases=("PHASE2",),
        conditions=("Glioblastoma",),
        interventions=(TrialIntervention(name="Comparator", type="DRUG"),),
        lead_sponsor=TrialSponsor(name="Comparator Bio", sponsor_class="INDUSTRY"),
        enrollment_count=60,
        primary_endpoints=(TrialEndpoint(measure="Progression-free survival", endpoint_type="primary"),),
        source_id="ctgov:NCT87654321",
    )


def _source(source_id: str) -> SourceMetadata:
    return SourceMetadata(
        source_id=source_id,
        title=source_id,
        provenance="test",
        source_type="fixture",
    )


def _install_agent_fixtures(monkeypatch) -> None:
    monkeypatch.setattr(cop_agent.ClinicalTrialsGovClient, "fetch_trial", lambda self, nct_id: _trial())
    monkeypatch.setattr(
        cop_agent.ClinicalTrialsGovClient,
        "search_trials",
        lambda self, input_data: ClinicalTrialsSearchResult(
            query=input_data,
            trials=(_trial(), _comparator_trial()),
            sources=(_source("ctgov:NCT12345678"), _source("ctgov:NCT87654321")),
            api_url="https://clinicaltrials.gov/api/v2/studies?fixture=true",
        ),
    )
    monkeypatch.setattr(
        cop_agent,
        "resolve_asset_identity",
        lambda trial: (
            AssetIdentityOutput(
                nct_id=trial.nct_id,
                asset_name="Examplemab",
                raw_intervention_names=("Examplemab",),
                sponsor="Example Bio",
                normalized_indication="glioblastoma",
                therapeutic_area="oncology",
                modality="antibody",
                rule_ids=("modality_antibody", "indication_oncology"),
                source_ids=("ctgov:NCT12345678",),
                confidence=0.9,
            ),
            (_source("ctgov:NCT12345678"),),
        ),
    )
    monkeypatch.setattr(
        cop_agent,
        "lookup_pos",
        lambda trial, asset, workbook_path=None: (
            PoSOutput(
                probability_of_success=0.344,
                current_phase="Phase II",
                disease_area="Oncology",
                workbook_path="fixture.xlsx",
                lookup_key="Disease Area|Oncology|Phase II",
                benchmark_row={"Key": "Disease Area|Oncology|Phase II", "Phase LOA": 0.344},
                source_ids=("pos_workbook:fixture",),
                confidence=0.9,
            ),
            _source("pos_workbook:fixture"),
        ),
    )
    monkeypatch.setattr(
        cop_agent,
        "_label_context",
        lambda asset, trial, client=None: (
            SafetyContext(
                label_available=True,
                summary="Warnings and adverse reactions fixture.",
                source_ids=("openfda_label:examplemab",),
                confidence=0.7,
            ),
            LabelExpansionClinicalRationale(
                rationale="Registry condition and public label context support clinical rationale structuring.",
                source_ids=("ctgov:NCT12345678", "openfda_label:examplemab"),
                confidence=0.6,
            ),
            (_source("openfda_label:examplemab"),),
        ),
    )


def test_clinical_outcome_prediction_workflow_persists_bundle(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)
    store = MemoryStore(":memory:")

    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        memory=store,
    )

    bundle = store.get_run_bundle(output.run_id)
    assert bundle.run is not None
    assert output.trial_identity.nct_id == "NCT12345678"
    assert output.asset_identity.asset_name == "Examplemab"
    assert output.historical_pos_estimate.probability_of_success == 0.344
    assert output.approval_likelihood_proxy.probability == 0.344
    assert output.comparator_benchmarking.matched_public_trials_count == 1
    assert output.failure_mode_classification.likely_failure_modes
    assert any(flag.status == "not_implemented" for flag in output.source_availability.flags)
    assert bundle.sources
    assert bundle.claims
    assert bundle.validation_results
    assert bundle.agent_outputs
    assert bundle.reports


def test_clinical_outcome_prediction_cli_route(monkeypatch, tmp_path) -> None:
    _install_agent_fixtures(monkeypatch)
    db_path = tmp_path / "memory.sqlite"
    output_path = tmp_path / "clinical_outcome_prediction.json"

    exit_code = main(
        [
            "run",
            "clinical_outcome_prediction",
            "--nct-id",
            "NCT12345678",
            "--db-path",
            str(db_path),
            "--output-json",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["input"]["nct_id"] == "NCT12345678"
    assert payload["historical_pos_estimate"]["probability_of_success"] == 0.344
    assert payload["approval_likelihood_proxy"]["probability"] == 0.344
    assert any(
        flag["source_name"] == "TrialBench/HINT/TOP-style models" and flag["status"] == "not_implemented"
        for flag in payload["source_availability"]["flags"]
    )


def test_clinical_outcome_prediction_schema_rejects_agent4_fields() -> None:
    fields = schemas.ClinicalOutcomePredictionOutput.model_fields
    assert "rnpv" not in fields
    assert "commercial_model" not in fields
    assert "patent_exclusivity" not in fields
