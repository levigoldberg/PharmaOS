from __future__ import annotations

import json

from pharma_os.cli import main
from pharma_os.memory import MemoryStore
from pharma_os.schemas import (
    AssetIdentityOutput,
    ClinicalTrialRecord,
    DueDiligenceInput,
    PatentExclusivityOutput,
    PoSOutput,
    PricingOutput,
    SourceMetadata,
    TrialIntervention,
    TrialSponsor,
)
from pharma_os.workflows import due_diligence
from pharma_os.workflows.due_diligence import run_due_diligence_workflow


def _trial() -> ClinicalTrialRecord:
    return ClinicalTrialRecord(
        nct_id="NCT12345678",
        brief_title="Example glioblastoma trial",
        overall_status="RECRUITING",
        phases=("PHASE2",),
        conditions=("Glioblastoma",),
        interventions=(TrialIntervention(name="Examplemab", type="DRUG", description="monoclonal antibody"),),
        lead_sponsor=TrialSponsor(name="Example Bio", sponsor_class="INDUSTRY"),
        enrollment_count=42,
        source_id="ctgov:NCT12345678",
    )


def _source(source_id: str) -> SourceMetadata:
    return SourceMetadata(
        source_id=source_id,
        title=source_id,
        provenance="test",
        source_type="fixture",
    )


def test_due_diligence_workflow_persists_bundle(monkeypatch) -> None:
    monkeypatch.setattr(due_diligence.ClinicalTrialsGovClient, "fetch_trial", lambda self, nct_id: _trial())
    monkeypatch.setattr(
        due_diligence,
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
                source_ids=("ctgov:NCT12345678",),
                confidence=0.9,
            ),
            (_source("ctgov:NCT12345678"),),
        ),
    )
    monkeypatch.setattr(
        due_diligence,
        "search_patent_exclusivity",
        lambda asset, loe_year_override=None: (
            PatentExclusivityOutput(
                asset_name="Examplemab",
                searched_terms=("Examplemab",),
                estimated_loe_year=loe_year_override,
                source_ids=("lens:examplemab",),
                confidence=0.8,
            ),
            (_source("lens:examplemab"),),
        ),
    )
    monkeypatch.setattr(
        due_diligence,
        "lookup_pos",
        lambda trial, asset, workbook_path=None: (
            PoSOutput(
                probability_of_success=0.344,
                current_phase="Phase II",
                disease_area="Oncology",
                workbook_path="fixture.xlsx",
                lookup_key="Disease Area|Oncology|Phase II",
                source_ids=("pos_workbook:fixture",),
                confidence=0.9,
            ),
            _source("pos_workbook:fixture"),
        ),
    )
    monkeypatch.setattr(
        due_diligence,
        "lookup_pricing",
        lambda asset, wac_data_path=None: (
            PricingOutput(
                annual_wac=100000.0,
                wac_value=100000.0,
                wac_unit_basis="annual",
                matched_product="Examplemab vial",
                dosing_summary="Use once monthly.",
                source_ids=("wac:fixture", "openfda_label:examplemab"),
                confidence=0.8,
            ),
            (_source("wac:fixture"), _source("openfda_label:examplemab")),
        ),
    )

    store = MemoryStore(":memory:")
    output = run_due_diligence_workflow(
        DueDiligenceInput(
            nct_id="NCT12345678",
            annual_patients=1000,
            peak_penetration=0.2,
            gross_to_net=0.15,
            operating_margin=0.35,
            discount_rate=0.1,
            development_cost=50_000_000,
            launch_year=2029,
            loe_year=2040,
        ),
        memory=store,
    )

    bundle = store.get_run_bundle(output.run_id)
    assert bundle.run is not None
    assert output.asset_identity.asset_name == "Examplemab"
    assert output.pos.probability_of_success == 0.344
    assert output.pricing.annual_wac == 100000.0
    assert output.commercial_model.calculable
    assert output.rnpv.calculable
    assert bundle.sources
    assert bundle.claims
    assert bundle.validation_results
    assert bundle.agent_outputs
    assert bundle.reports


def test_due_diligence_cli_persists_report(monkeypatch, tmp_path) -> None:
    def fake_run(input_data, memory=None):
        store = memory or MemoryStore(":memory:")
        output = run_due_diligence_workflow(input_data, memory=store)
        return output

    monkeypatch.setattr(due_diligence.ClinicalTrialsGovClient, "fetch_trial", lambda self, nct_id: _trial())
    monkeypatch.setattr(
        due_diligence,
        "resolve_asset_identity",
        lambda trial: (
            AssetIdentityOutput(nct_id=trial.nct_id, asset_name="Examplemab", source_ids=("ctgov:NCT12345678",), confidence=0.9),
            (_source("ctgov:NCT12345678"),),
        ),
    )
    monkeypatch.setattr(
        due_diligence,
        "search_patent_exclusivity",
        lambda asset, loe_year_override=None: (
            PatentExclusivityOutput(asset_name="Examplemab", estimated_loe_year=loe_year_override, source_ids=("lens:examplemab",), confidence=0.8),
            (_source("lens:examplemab"),),
        ),
    )
    monkeypatch.setattr(
        due_diligence,
        "lookup_pos",
        lambda trial, asset, workbook_path=None: (
            PoSOutput(probability_of_success=0.344, current_phase="Phase II", disease_area="Oncology", source_ids=("pos_workbook:fixture",), confidence=0.9),
            _source("pos_workbook:fixture"),
        ),
    )
    monkeypatch.setattr(
        due_diligence,
        "lookup_pricing",
        lambda asset, wac_data_path=None: (
            PricingOutput(annual_wac=100000.0, wac_value=100000.0, dosing_summary="Use once monthly.", source_ids=("wac:fixture", "openfda_label:examplemab"), confidence=0.8),
            (_source("wac:fixture"), _source("openfda_label:examplemab")),
        ),
    )
    db_path = tmp_path / "memory.sqlite"
    output_path = tmp_path / "due.json"
    exit_code = main(
        [
            "run",
            "due_diligence",
            "--nct-id",
            "NCT12345678",
            "--annual-patients",
            "1000",
            "--peak-penetration",
            "0.2",
            "--gross-to-net",
            "0.15",
            "--operating-margin",
            "0.35",
            "--discount-rate",
            "0.1",
            "--development-cost",
            "50000000",
            "--launch-year",
            "2029",
            "--loe-year",
            "2040",
            "--db-path",
            str(db_path),
            "--output-json",
            str(output_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    report_path = tmp_path / "report.json"
    exit_code = main(["report", "--run-id", payload["run_id"], "--db-path", str(db_path), "--output-json", str(report_path)])
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_id"] == payload["run_id"]
    assert report["sources"]
