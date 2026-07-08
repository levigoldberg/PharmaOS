from __future__ import annotations

import json
from datetime import datetime, timezone

from pharma_os.cli import main
from pharma_os.memory import MemoryStore
from pharma_os.schemas import (
    Agent3HandoffReference,
    Agent4HandoffReference,
    AgentOutput,
    AnalogCandidateRecord,
    ApprovalLikelihoodProxy,
    AssetIdentityOutput,
    AssetMemo,
    ClinicalEvidenceSummary,
    ClinicalOutcomePredictionInput,
    ClinicalOutcomePredictionOutput,
    ClinicalRiskSummary,
    ClinicalTrialRecord,
    ClinicalTrialsSearchResult,
    CommercialModelOutput,
    ComparatorBenchmarkBundle,
    CompetitiveLandscapeSummary,
    DueDiligenceInput,
    DueDiligenceOutput,
    EndpointRiskAssessment,
    EnrollmentDurationRisk,
    FailureModeClassification,
    HistoricalPoSEstimate,
    LabelExpansionClinicalRationale,
    PatentExclusivityOutput,
    PatentLOEReview,
    PoSOutput,
    PricingOutput,
    ProtocolDesignInput,
    RNPVOutput,
    SafetyContext,
    SafetyLabelSummary,
    SourceAvailabilityReport,
    SourceMetadata,
    TrialDesignFeatures,
    TrialEndpoint,
    TrialIdentity,
    TrialIntervention,
    TrialSponsor,
    WorkflowRun,
)
from pharma_os.tools.protocol_design import calculate_analog_benchmark, execute_ctgov_search_plan
from pharma_os.workflows import protocol_design
from pharma_os.workflows.protocol_design import run_protocol_design_workflow


def _source(source_id: str, source_type: str = "fixture") -> SourceMetadata:
    return SourceMetadata(source_id=source_id, title=source_id, provenance="test", source_type=source_type)


def _target_trial() -> ClinicalTrialRecord:
    return ClinicalTrialRecord(
        nct_id="NCT12345678",
        brief_title="Randomized Examplemab in glioblastoma",
        overall_status="RECRUITING",
        phases=("PHASE2",),
        study_type="INTERVENTIONAL",
        conditions=("Glioblastoma",),
        interventions=(TrialIntervention(name="Examplemab", type="DRUG"), TrialIntervention(name="Placebo", type="DRUG")),
        lead_sponsor=TrialSponsor(name="Example Bio", sponsor_class="INDUSTRY"),
        enrollment_count=80,
        start_date="2026-01",
        primary_completion_date="2028-01",
        results_available=False,
        primary_endpoints=(TrialEndpoint(measure="Progression-free survival", endpoint_type="primary"),),
        secondary_endpoints=(TrialEndpoint(measure="Overall response rate", endpoint_type="secondary"),),
        eligibility_criteria="Inclusion Criteria: diagnosis; measurable disease; biomarker testing. Exclusion Criteria: infection; cardiac disease.",
        minimum_age="18 Years",
        sex="ALL",
        source_id="ctgov:NCT12345678",
    )


def _analog(nct_id: str, *, enrollment: int | None = 100, status: str = "COMPLETED") -> ClinicalTrialRecord:
    return ClinicalTrialRecord(
        nct_id=nct_id,
        brief_title=f"Randomized analog {nct_id}",
        overall_status=status,
        phases=("PHASE2",),
        study_type="INTERVENTIONAL",
        conditions=("Glioblastoma",),
        interventions=(TrialIntervention(name="Analogmab", type="DRUG"), TrialIntervention(name="Placebo", type="DRUG")),
        lead_sponsor=TrialSponsor(name="Analog Bio", sponsor_class="INDUSTRY"),
        enrollment_count=enrollment,
        start_date="2020-01",
        primary_completion_date="2022-01",
        results_available=True,
        primary_endpoints=(TrialEndpoint(measure="Progression-free survival", endpoint_type="primary"),),
        secondary_endpoints=(TrialEndpoint(measure="Overall response rate", endpoint_type="secondary"),),
        eligibility_criteria="Inclusion Criteria: diagnosis; measurable disease; biomarker testing. Exclusion Criteria: infection; cardiac disease; prior therapy.",
        locations=(),
        source_id=f"ctgov:{nct_id}",
    )


def _agent3_output() -> ClinicalOutcomePredictionOutput:
    trial = _target_trial()
    return ClinicalOutcomePredictionOutput(
        output_id="clinical-outcome-prediction-output-agent3",
        run_id="agent3-run",
        input=ClinicalOutcomePredictionInput(nct_id=trial.nct_id),
        trial_identity=TrialIdentity(nct_id=trial.nct_id, phases=trial.phases, conditions=trial.conditions, sponsor="Example Bio", source_ids=(trial.source_id,)),
        asset_identity=AssetIdentityOutput(nct_id=trial.nct_id, asset_name="Examplemab", sponsor="Example Bio", normalized_indication="glioblastoma", modality="antibody", source_ids=(trial.source_id,), confidence=0.9),
        trial_design_features=TrialDesignFeatures(primary_endpoint_count=1, enrollment_count=80, source_ids=(trial.source_id,)),
        endpoint_risk_assessment=EndpointRiskAssessment(risk_level="medium", rationale="Fixture endpoint risk.", source_ids=(trial.source_id,), confidence=0.7),
        enrollment_duration_risk=EnrollmentDurationRisk(risk_level="low", rationale="Fixture enrollment risk.", source_ids=(trial.source_id,), confidence=0.8),
        comparator_benchmarking=ComparatorBenchmarkBundle(matched_public_trials_count=2, comparator_trial_ids=("NCT00000001",), benchmark_summary="Fixture comparator benchmark.", source_ids=(trial.source_id,), confidence=0.7),
        historical_pos_estimate=HistoricalPoSEstimate(probability_of_success=0.344, current_phase="Phase II", disease_area="Oncology", lookup_key="Disease Area|Oncology|Phase II", assumption_type="source_derived", source_ids=("pos_workbook:fixture",), confidence=0.9),
        approval_likelihood_proxy=ApprovalLikelihoodProxy(probability=0.344, basis="Fixture proxy.", assumption_type="source_derived", source_ids=("pos_workbook:fixture",), confidence=0.9),
        failure_mode_classification=FailureModeClassification(overall_risk_level="medium", source_ids=(trial.source_id,), confidence=0.7),
        safety_context=SafetyContext(label_available=True, summary="Fixture safety context.", source_ids=("openfda_label:examplemab",), confidence=0.7),
        label_expansion_clinical_rationale=LabelExpansionClinicalRationale(rationale="Fixture label rationale.", source_ids=(trial.source_id,), confidence=0.6),
        source_availability=SourceAvailabilityReport(),
        sources=(_source(trial.source_id, "clinical_trial_registry"), _source("pos_workbook:fixture", "pos_workbook"), _source("openfda_label:examplemab", "drug_label")),
        confidence=0.8,
        validation_status="passed",
    )


def _agent4_output() -> DueDiligenceOutput:
    trial = _target_trial()
    asset = AssetIdentityOutput(nct_id=trial.nct_id, asset_name="Examplemab", sponsor="Example Bio", normalized_indication="glioblastoma", therapeutic_area="oncology", modality="antibody", source_ids=(trial.source_id,), confidence=0.9)
    input_data = DueDiligenceInput(nct_id=trial.nct_id)
    return DueDiligenceOutput(
        output_id="due-diligence-output-agent4",
        run_id="agent4-run",
        input=input_data,
        target_trial=trial,
        trial=trial,
        asset_identity=asset,
        agent3_handoff=Agent3HandoffReference(agent3_run_id="agent3-run", agent3_output_id="clinical-outcome-prediction-output-agent3", nct_id=trial.nct_id, generated_or_reused="reused", retrieved_from_memory=True, confidence=0.8),
        clinical_risk_summary=ClinicalRiskSummary(nct_id=trial.nct_id, asset_name="Examplemab", indication="glioblastoma", phase="PHASE2", endpoint_risk_level="medium", enrollment_duration_risk_level="low", source_ids=(trial.source_id,), confidence=0.8),
        clinical_evidence=ClinicalEvidenceSummary(nct_id=trial.nct_id, ctgov_summary="Fixture CT.gov evidence.", pubmed_article_count=1, pubmed_titles=("Examplemab evidence",), source_ids=(trial.source_id, "pubmed:1"), confidence=0.8),
        competitive_landscape=CompetitiveLandscapeSummary(nct_id=trial.nct_id, matched_public_trials_count=2, benchmark_summary="Fixture landscape.", source_ids=(trial.source_id,), confidence=0.7),
        safety_label_summary=SafetyLabelSummary(asset_name="Examplemab", label_available=True, warnings_summary="Fixture warning.", source_ids=("openfda_label:examplemab",), confidence=0.7),
        patent_loe_review=PatentLOEReview(asset_name="Examplemab", review_summary="Fixture LOE review.", confidence=0.0),
        patent_exclusivity=PatentExclusivityOutput(asset_name="Examplemab", confidence=0.0),
        pos=PoSOutput(probability_of_success=0.344, current_phase="Phase II", disease_area="Oncology", source_ids=("pos_workbook:fixture",), confidence=0.9),
        pricing=PricingOutput(annual_wac=None, confidence=0.0),
        commercial_model=CommercialModelOutput(calculable=False, confidence=0.0),
        rnpv=RNPVOutput(calculable=False, confidence=0.0),
        asset_memo=AssetMemo(memo_id="memo-agent4", title="Fixture memo", summary="Fixture memo.", requires_human_review=True, confidence=0.5),
        sources=(_source(trial.source_id, "clinical_trial_registry"), _source("pubmed:1", "literature"), _source("openfda_label:examplemab", "drug_label")),
        confidence=0.6,
        validation_status="needs_human_review",
    )


def _handoffs() -> tuple[ClinicalOutcomePredictionOutput, Agent3HandoffReference, DueDiligenceOutput, Agent4HandoffReference]:
    agent3 = _agent3_output()
    agent4 = _agent4_output()
    return (
        agent3,
        Agent3HandoffReference(agent3_run_id=agent3.run_id, agent3_output_id=agent3.output_id, nct_id=agent3.input.nct_id, generated_or_reused="reused", retrieved_from_memory=True, source_ids=tuple(source.source_id for source in agent3.sources), confidence=agent3.confidence),
        agent4,
        Agent4HandoffReference(agent4_run_id=agent4.run_id, agent4_output_id=agent4.output_id, nct_id=agent4.input.nct_id, generated_or_reused="reused", retrieved_from_memory=True, source_ids=tuple(source.source_id for source in agent4.sources), confidence=agent4.confidence),
    )


def test_execute_ctgov_search_plan_dedupes_and_preserves_provenance() -> None:
    agent3, _, agent4, _ = _handoffs()
    plan = protocol_design.build_search_strategy(run_id="run", target_trial=_target_trial(), agent3_output=agent3, agent4_output=agent4)

    class FakeClient:
        def search_trials(self, input_data):
            analog = _analog("NCT00000001")
            return ClinicalTrialsSearchResult(
                query=input_data,
                trials=(analog, analog, _target_trial()),
                sources=(_source(analog.source_id, "clinical_trial_registry"), _source(_target_trial().source_id, "clinical_trial_registry")),
                api_url="https://clinicaltrials.gov/api/v2/studies",
            )

    candidates, sources, flags = execute_ctgov_search_plan(search_plan=plan, target_nct_id="NCT12345678", client=FakeClient())

    assert not flags
    assert [candidate.trial.nct_id for candidate in candidates] == ["NCT00000001"]
    assert candidates[0].query_ids
    assert any(source.source_id.startswith("ctgov_search:protocol_design:") for source in sources)


def test_calculate_analog_benchmark_handles_missing_values() -> None:
    agent3, _, agent4, _ = _handoffs()
    plan = protocol_design.build_search_strategy(run_id="run", target_trial=_target_trial(), agent3_output=agent3, agent4_output=agent4)
    candidates = (
        AnalogCandidateRecord(candidate_id="c1", trial=_analog("NCT00000001", enrollment=100), query_ids=("q1",), source_ids=("ctgov:NCT00000001",), provenance="test"),
        AnalogCandidateRecord(candidate_id="c2", trial=_analog("NCT00000002", enrollment=None), query_ids=("q1",), source_ids=("ctgov:NCT00000002",), provenance="test"),
    )
    selection = protocol_design.select_analog_trials(run_id="run", target_trial=_target_trial(), candidates=candidates, agent3_output=agent3, agent4_output=agent4, search_plan=plan, top_k=10)

    bundle = calculate_analog_benchmark(run_id="run", target_trial=_target_trial(), candidates=candidates, selection=selection, search_plan=plan)

    assert bundle.enrollment.observed_count == 1
    assert bundle.enrollment.missing_count == 1
    assert bundle.primary_endpoint_family_frequency
    assert any(flag.field == "enrollment" for flag in bundle.missing_data_flags)


def test_protocol_design_workflow_returns_brief_and_persists_bundle(monkeypatch) -> None:
    agent3, handoff3, agent4, handoff4 = _handoffs()
    monkeypatch.setattr(protocol_design, "_get_or_run_agent3", lambda input_data, memory: (agent3, handoff3))
    monkeypatch.setattr(protocol_design, "_get_or_run_agent4", lambda input_data, memory: (agent4, handoff4))

    class FakeClient:
        def search_trials(self, input_data):
            analogs = (_analog("NCT00000001", enrollment=120), _analog("NCT00000002", enrollment=80))
            return ClinicalTrialsSearchResult(
                query=input_data,
                trials=analogs,
                sources=tuple(_source(analog.source_id, "clinical_trial_registry") for analog in analogs),
                api_url="https://clinicaltrials.gov/api/v2/studies",
            )

    monkeypatch.setattr(protocol_design, "execute_ctgov_search_plan", lambda search_plan, target_nct_id: execute_ctgov_search_plan(search_plan=search_plan, target_nct_id=target_nct_id, client=FakeClient()))
    store = MemoryStore(":memory:")

    output = run_protocol_design_workflow(ProtocolDesignInput(nct_id="NCT12345678"), memory=store)

    assert output.protocol_design_brief.artifact_type == "draft_protocol_design_brief"
    assert output.protocol_design_brief.requires_human_review is True
    assert output.analog_benchmark_bundle.selected_analog_ids
    assert output.human_gate is not None
    assert output.human_gate.decision == "needs_human_review"
    assert output.validation_status == "needs_human_review"
    bundle = store.get_run_bundle(output.run_id)
    assert bundle.run is not None
    assert bundle.agent_outputs
    payload = json.loads(store._connection.execute("SELECT output_json FROM runs WHERE run_id = ?", (output.run_id,)).fetchone()["output_json"])
    assert payload["analog_benchmark_bundle"]["selected_analog_ids"]


def test_protocol_design_cli_command(monkeypatch, tmp_path) -> None:
    agent3, handoff3, agent4, handoff4 = _handoffs()
    monkeypatch.setattr(protocol_design, "_get_or_run_agent3", lambda input_data, memory: (agent3, handoff3))
    monkeypatch.setattr(protocol_design, "_get_or_run_agent4", lambda input_data, memory: (agent4, handoff4))

    class FakeClient:
        def search_trials(self, input_data):
            analog = _analog("NCT00000001", enrollment=120)
            return ClinicalTrialsSearchResult(
                query=input_data,
                trials=(analog,),
                sources=(_source(analog.source_id, "clinical_trial_registry"),),
                api_url="https://clinicaltrials.gov/api/v2/studies",
            )

    monkeypatch.setattr(protocol_design, "execute_ctgov_search_plan", lambda search_plan, target_nct_id: execute_ctgov_search_plan(search_plan=search_plan, target_nct_id=target_nct_id, client=FakeClient()))
    output_path = tmp_path / "protocol_design.json"

    exit_code = main(["run", "protocol_design", "--nct-id", "NCT12345678", "--db-path", str(tmp_path / "memory.sqlite"), "--output-json", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["input"]["nct_id"] == "NCT12345678"
    assert payload["protocol_design_brief"]["requires_human_review"] is True
    assert payload["human_gate"]["decision"] == "needs_human_review"


def _save_agent_output(store: MemoryStore, workflow_name: str, output: object, source_ids: tuple[str, ...]) -> None:
    run = WorkflowRun(
        run_id=getattr(output, "run_id"),
        workflow_name=workflow_name,
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        input_provenance="test",
        source_ids=source_ids,
        validation_status="passed",
        metadata={"nct_id": "NCT12345678"},
    )
    store.save_run(run, input_payload=getattr(output, "input"), output_payload=output)
    store.save_agent_output(
        AgentOutput(
            output_id=f"agent-output-{getattr(output, 'run_id')}",
            agent_name=workflow_name,
            run_id=getattr(output, "run_id"),
            provenance="test",
            sources=getattr(output, "sources"),
            claims=getattr(output, "claims", ()),
            confidence=getattr(output, "confidence"),
            validation_status="passed",
        ),
        payload=output,
    )
