from __future__ import annotations

import json
from datetime import datetime, timezone

from pharma_os.agent_runtime import StructuredAgentResult
from pharma_os import schemas
from pharma_os.agents import clinical_outcome_prediction as cop_agent
from pharma_os.cli import main
from pharma_os.memory import MemoryStore
from pharma_os.schemas import (
    AssetIdentityOutput,
    AgentRunTrace,
    AgentStepTrace,
    ClinicalOutcomePredictionInput,
    ClinicalTrialsSearchResult,
    ClinicalTrialRecord,
    EndpointRiskAssessment,
    EvidenceClaim,
    LabelExpansionClinicalRationale,
    PoSOutput,
    SafetyContext,
    SourceMetadata,
    TrialEndpoint,
    TrialIntervention,
    TrialLocation,
    TrialSponsor,
)
from pharma_os.validators import validate_clinical_outcome_constraints, validate_source_coverage
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


def _weak_comparator_trial() -> ClinicalTrialRecord:
    return ClinicalTrialRecord(
        nct_id="NCT11111111",
        brief_title="Weak comparator glioblastoma trial",
        overall_status="RECRUITING",
        phases=("PHASE1",),
        conditions=("Glioblastoma",),
        interventions=(TrialIntervention(name="Weak Comparator", type="DRUG"),),
        lead_sponsor=TrialSponsor(name="Weak Bio", sponsor_class="INDUSTRY"),
        enrollment_count=20,
        primary_endpoints=(TrialEndpoint(measure="Biomarker response", endpoint_type="primary"),),
        source_id="ctgov:NCT11111111",
    )


def _excluded_comparator_trial() -> ClinicalTrialRecord:
    return ClinicalTrialRecord(
        nct_id="NCT22222222",
        brief_title="Excluded melanoma trial",
        overall_status="COMPLETED",
        phases=("PHASE1",),
        conditions=("Melanoma",),
        interventions=(TrialIntervention(name="Excluded Comparator", type="DEVICE"),),
        lead_sponsor=TrialSponsor(name="Excluded Bio", sponsor_class="INDUSTRY"),
        enrollment_count=100,
        primary_endpoints=(TrialEndpoint(measure="Overall response rate", endpoint_type="primary"),),
        source_id="ctgov:NCT22222222",
    )


def _trial_missing_enrollment() -> ClinicalTrialRecord:
    trial = _trial()
    return trial.model_copy(update={"enrollment_count": None, "start_date": None, "primary_completion_date": None, "completion_date": None})


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
    assert output.comparator_benchmarking.landscape_summary == "Found 2 ClinicalTrials.gov records for Glioblastoma."
    assert output.comparator_benchmarking.status_summary is not None
    assert output.comparator_benchmarking.risk_flags
    assert output.failure_mode_classification.likely_failure_modes
    assert any(flag.status == "not_implemented" for flag in output.source_availability.flags)
    assert bundle.sources
    assert bundle.claims
    assert bundle.validation_results
    assert bundle.agent_outputs
    assert bundle.agent_traces
    trace_agents = {trace.agent_name for trace in bundle.agent_traces}
    assert "ClinicalOutcomeManagerAgent" in trace_agents
    assert "EndpointRiskAgent" in trace_agents
    assert "ComparatorRelevanceAgent" in trace_agents
    assert "EnrollmentFeasibilityAgent" in trace_agents
    assert "SafetyContextAgent" in trace_agents
    assert "FailureModeSynthesisAgent" in trace_agents
    assert bundle.reports


def test_clinical_outcome_prediction_cli_route(monkeypatch, tmp_path) -> None:
    _install_agent_fixtures(monkeypatch)
    db_path = tmp_path / "memory.sqlite"
    output_path = tmp_path / "clinical_outcome_prediction.json"
    html_path = tmp_path / "clinical_outcome_prediction.html"

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
            "--output-html",
            str(html_path),
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
    html = html_path.read_text(encoding="utf-8")
    assert "Agent Traces" in html
    assert "EndpointRiskAgent" in html
    assert "ComparatorRelevanceAgent" in html


def test_clinical_outcome_prediction_offline_fallback(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)
    monkeypatch.setenv("PHARMA_OS_AGENTS_DISABLED", "true")
    store = MemoryStore(":memory:")

    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        memory=store,
    )

    bundle = store.get_run_bundle(output.run_id)
    assert output.endpoint_risk_assessment.rationale == "Endpoint risk is based on primary endpoint count and endpoint wording in the registry record."
    assert bundle.agent_traces
    assert all(trace.provenance == "pharma_os.agent_runtime.offline" for trace in bundle.agent_traces)


def test_ambiguous_asset_identity_triggers_adjudication(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)
    ambiguous_trial = _trial().model_copy(
        update={
            "interventions": (
                TrialIntervention(name="Examplemab", type="DRUG"),
                TrialIntervention(name="Partnerdrug", type="DRUG"),
            )
        }
    )
    monkeypatch.setattr(cop_agent.ClinicalTrialsGovClient, "fetch_trial", lambda self, nct_id: ambiguous_trial)
    store = MemoryStore(":memory:")

    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        memory=store,
    )

    bundle = store.get_run_bundle(output.run_id)
    assert "AssetIdentityAdjudicatorAgent" in {trace.agent_name for trace in bundle.agent_traces}
    assert output.asset_identity.confidence <= 0.55
    assert any(flag.section == "asset_identity" for flag in output.asset_identity.missing_data_flags)


def test_endpoint_risk_agent_output_validation(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)
    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        memory=MemoryStore(":memory:"),
    )

    assert output.endpoint_risk_assessment.risk_level in {"low", "medium", "high", "unknown"}
    assert output.endpoint_risk_assessment.source_ids
    assert output.endpoint_risk_assessment.rationale


def test_mock_agent_runtime_endpoint_output_is_used(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)

    def fake_runtime(**kwargs):
        output = kwargs["offline_output"]
        if kwargs["agent_name"] == "EndpointRiskAgent":
            output = EndpointRiskAssessment(
                risk_level="high",
                risk_factors=("mock endpoint hierarchy ambiguity",),
                rationale="Mock agent output flagged endpoint hierarchy ambiguity.",
                source_ids=("ctgov:NCT12345678",),
                confidence=0.6,
            )
        now = datetime.now(timezone.utc)
        trace = AgentRunTrace(
            trace_id=f"trace-{kwargs['agent_name']}",
            run_id=kwargs["run_id"],
            agent_name=kwargs["agent_name"],
            input_summary=kwargs["input_summary"],
            output_id=getattr(output, "output_id", None),
            output_type=output.__class__.__name__,
            output_summary=f"{kwargs['agent_name']} mocked output.",
            steps=(
                AgentStepTrace(
                    run_id=kwargs["run_id"],
                    agent_name=kwargs["agent_name"],
                    step_id=f"step-{kwargs['agent_name']}",
                    input_summary=kwargs["input_summary"],
                    output_summary=f"{kwargs['agent_name']} mocked output.",
                    source_ids=kwargs["source_ids"],
                    confidence=kwargs["confidence"],
                    started_at=now,
                    completed_at=now,
                    provenance="test.mock_agent_runtime",
                ),
            ),
            source_ids=kwargs["source_ids"],
            confidence=kwargs["confidence"],
            rationale_summary=kwargs["rationale_summary"],
            started_at=now,
            completed_at=now,
            provenance="test.mock_agent_runtime",
        )
        return StructuredAgentResult(output=output, trace=trace)

    monkeypatch.setattr(cop_agent, "run_structured_agent", fake_runtime)

    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        memory=MemoryStore(":memory:"),
    )

    assert output.endpoint_risk_assessment.risk_level == "high"
    assert output.endpoint_risk_assessment.rationale == "Mock agent output flagged endpoint hierarchy ambiguity."


def test_comparator_relevance_includes_relevant_weak_and_excluded(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)
    monkeypatch.setattr(
        cop_agent.ClinicalTrialsGovClient,
        "search_trials",
        lambda self, input_data: ClinicalTrialsSearchResult(
            query=input_data,
            trials=(_trial(), _comparator_trial(), _weak_comparator_trial(), _excluded_comparator_trial()),
            sources=(
                _source("ctgov:NCT12345678"),
                _source("ctgov:NCT87654321"),
                _source("ctgov:NCT11111111"),
                _source("ctgov:NCT22222222"),
            ),
            api_url="https://clinicaltrials.gov/api/v2/studies?fixture=true",
        ),
    )

    result = cop_agent.run_clinical_outcome_prediction_agent_result(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        run_id="test-run",
    )

    relevance = next(payload for payload in result.subagent_payloads if payload.__class__.__name__ == "ComparatorRelevanceOutput")
    categories = {item.relevance for item in relevance.trial_relevance}
    assert categories == {"relevant", "weak", "excluded"}
    assert "relevant" in result.output.comparator_benchmarking.benchmark_summary
    assert "excluded" in result.output.comparator_benchmarking.benchmark_summary


def test_enrollment_feasibility_handles_missing_enrollment_and_duration(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)
    monkeypatch.setattr(cop_agent.ClinicalTrialsGovClient, "fetch_trial", lambda self, nct_id: _trial_missing_enrollment())

    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        memory=MemoryStore(":memory:"),
    )

    assert output.enrollment_duration_risk.enrollment_count is None
    assert output.enrollment_duration_risk.planned_duration_months is None
    assert {flag.field for flag in output.enrollment_duration_risk.missing_data_flags} >= {"enrollment_count", "planned_duration_months"}
    assert any(mode.category in {"enrollment", "missing_data"} for mode in output.failure_mode_classification.likely_failure_modes)


def test_safety_context_does_not_invent_adverse_event_rates(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)
    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        memory=MemoryStore(":memory:"),
    )
    unsafe_output = output.model_copy(
        update={
            "safety_context": output.safety_context.model_copy(
                update={"summary": "Adverse event rate is 50 percent based on agent estimate."}
            )
        }
    )

    results = validate_clinical_outcome_constraints(run_id="test-run", output=unsafe_output)
    assert results[0].status == "failed"
    assert "invented_safety_rate" in results[0].message


def test_failure_mode_synthesis_with_missing_data(monkeypatch) -> None:
    _install_agent_fixtures(monkeypatch)
    monkeypatch.setattr(cop_agent.ClinicalTrialsGovClient, "fetch_trial", lambda self, nct_id: _trial_missing_enrollment())

    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        memory=MemoryStore(":memory:"),
    )

    categories = {mode.category for mode in output.failure_mode_classification.likely_failure_modes}
    assert categories & {"enrollment", "safety", "missing_data"}
    assert output.human_gate is not None


def test_unsourced_agent_generated_claims_fail_validation() -> None:
    result = validate_source_coverage(
        target_id="clinical-output-fixture",
        claims=(
            EvidenceClaim.model_construct(
                claim_id="claim-unsourced-agent",
                claim_text="Agent-generated clinical claim without provenance.",
                source_ids=(),
                provenance="test.agent",
                confidence=0.4,
                confidence_level="low",
            ),
        ),
        source_ids={"ctgov:NCT12345678"},
        run_id="test-run",
    )

    assert result.status == "failed"


def test_clinical_outcome_prediction_schema_rejects_agent4_fields() -> None:
    fields = schemas.ClinicalOutcomePredictionOutput.model_fields
    assert "rnpv" not in fields
    assert "commercial_model" not in fields
    assert "patent_exclusivity" not in fields
