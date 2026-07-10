from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from pharma_os.cli import main
from pharma_os.agent_runtime import AgentRuntimeConfig
from pharma_os.html_report import write_run_html
from pharma_os.memory import MemoryStore
from pharma_os.schemas import (
    Agent3HandoffReference,
    Agent4HandoffReference,
    AgentOutput,
    AnalogCandidateRecord,
    AnalogSearchPlanOutput,
    AnalogTrialSelectionOutput,
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
    CTGovSearchQuery,
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
    SelectedAnalogTrial,
    SourceAvailabilityReport,
    SourceMetadata,
    TrialArmGroup,
    TrialDesignFeatures,
    TrialEndpoint,
    TrialIdentity,
    TrialIntervention,
    TrialLocation,
    TrialSponsor,
    WorkflowRun,
)
from pharma_os.tools.protocol_design import calculate_analog_benchmark, execute_ctgov_search_plan
from pharma_os.validators import validate_protocol_design_constraints
from pharma_os.workflows import protocol_design
from pharma_os.workflows.protocol_design import run_protocol_design_workflow
from pharma_os.agents.protocol_design import run_protocol_design_manager_agent
from pharma_os.agents import protocol_design as protocol_design_agents


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


def _benchmark_for_trials(trials: tuple[ClinicalTrialRecord, ...]):
    plan = AnalogSearchPlanOutput(
        output_id="search-plan",
        target_nct_id="NCT12345678",
        queries=(
            CTGovSearchQuery(
                query_id="q1",
                condition="Glioblastoma",
                expected_analog_dimension="structured design fixture",
                rationale="Fixture query.",
            ),
        ),
        rationale="Fixture search plan.",
    )
    candidates = tuple(
        AnalogCandidateRecord(
            candidate_id=f"candidate-{trial.nct_id}",
            trial=trial,
            query_ids=("q1",),
            source_ids=(trial.source_id,),
            provenance="test",
        )
        for trial in trials
    )
    selection = AnalogTrialSelectionOutput(
        output_id="selection",
        target_nct_id="NCT12345678",
        selected_analogs=tuple(
            SelectedAnalogTrial(
                nct_id=trial.nct_id,
                match_score=0.9,
                match_confidence="high",
                reasoning="Fixture selected analog.",
                source_ids=(trial.source_id,),
            )
            for trial in trials
        ),
        confidence=0.9,
    )
    return calculate_analog_benchmark(
        run_id="run",
        target_trial=_target_trial(),
        candidates=candidates,
        selection=selection,
        search_plan=plan,
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


def test_analog_benchmark_prefers_structured_design_fields() -> None:
    structured_trial = _analog("NCT00000001").model_copy(
        update={
            "brief_title": "Dose exploration without design keywords",
            "allocation": "RANDOMIZED",
            "masking": "DOUBLE",
            "number_of_arms": 2,
            "arm_groups": (
                TrialArmGroup(label="Experimental", type="EXPERIMENTAL", intervention_names=("Drug: Analogmab",)),
                TrialArmGroup(label="Placebo Control", type="PLACEBO_COMPARATOR", intervention_names=("Drug: Placebo",)),
            ),
            "interventions": (
                TrialIntervention(name="Analogmab", type="DRUG", arm_group_labels=("Experimental",)),
                TrialIntervention(name="Comparator Without Placebo Keyword", type="DRUG", arm_group_labels=("Placebo Control",)),
            ),
        }
    )

    bundle = _benchmark_for_trials((structured_trial,))

    assert bundle.randomized_frequency[0].label == "randomized"
    assert bundle.blinding_frequency[0].label == "blinded_or_masked"
    assert bundle.arm_count_distribution[0].label == "2"
    assert bundle.comparator_categories[0].label == "placebo_control"


def test_analog_benchmark_uses_heuristics_when_structured_fields_missing() -> None:
    heuristic_trial = _analog("NCT00000001").model_copy(
        update={
            "brief_title": "Randomized open-label analog study",
            "allocation": None,
            "masking": None,
            "number_of_arms": None,
            "arm_groups": (),
            "interventions": (TrialIntervention(name="Analogmab", type="DRUG"), TrialIntervention(name="Placebo", type="DRUG")),
        }
    )

    bundle = _benchmark_for_trials((heuristic_trial,))

    assert bundle.randomized_frequency[0].label == "randomized"
    assert bundle.blinding_frequency[0].label == "open_label"
    assert bundle.arm_count_distribution[0].label == "2"
    assert bundle.comparator_categories[0].label == "placebo_control"


def test_next_study_intent_does_not_blindly_increment_phase() -> None:
    agent3, _, agent4, _ = _handoffs()
    target = _target_trial()

    intent = protocol_design_agents.build_next_study_intent(
        run_id="run",
        target_trial=target,
        agent3_output=agent3,
        agent4_output=agent4,
        source_ids=(target.source_id,),
        missing_data_flags=(),
    )

    assert intent.evidence_anchor_nct_id == target.nct_id
    assert intent.current_development_stage == "Phase II"
    assert intent.proposed_next_stage == "Phase IIb optimization study"
    assert "not an automatic phase increment" in intent.rationale
    assert intent.requires_human_review is True


def test_analog_search_and_selection_follow_next_study_intent_phase() -> None:
    agent3, _, agent4, _ = _handoffs()
    target = _target_trial()
    intent = protocol_design_agents.build_next_study_intent(
        run_id="run",
        target_trial=target,
        agent3_output=agent3,
        agent4_output=agent4,
        source_ids=(target.source_id,),
        missing_data_flags=(),
    ).model_copy(
        update={
            "proposed_next_stage": "Phase III pivotal study",
            "study_role": "pivotal efficacy confirmation",
            "development_objective": "Confirm clinical benefit in a registration-enabling population.",
        }
    )

    plan = protocol_design.build_search_strategy(
        run_id="run",
        target_trial=target,
        agent3_output=agent3,
        agent4_output=agent4,
        next_study_intent=intent,
    )
    assert plan.queries[0].phase == "PHASE3"
    assert "proposed_next_stage" in plan.expected_dimensions
    assert "current target trial phase" in plan.rationale

    phase3_candidate = AnalogCandidateRecord(
        candidate_id="c1",
        trial=_analog("NCT00000003").model_copy(update={"phases": ("PHASE3",)}),
        query_ids=("q1",),
        source_ids=("ctgov:NCT00000003",),
        provenance="test",
    )
    selection = protocol_design.select_analog_trials(
        run_id="run",
        target_trial=target,
        candidates=(phase3_candidate,),
        agent3_output=agent3,
        agent4_output=agent4,
        search_plan=plan,
        next_study_intent=intent,
    )

    assert selection.selected_analogs
    assert "proposed_next_stage" in selection.selected_analogs[0].matched_dimensions


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
    assert output.next_study_intent.proposed_next_stage == "Phase IIb optimization study"
    assert output.protocol_design_brief.next_study_intent == output.next_study_intent
    assert "Phase IIb optimization study" in output.protocol_design_brief.title
    assert output.analog_benchmark_bundle.selected_analog_ids
    assert output.human_gate is not None
    assert output.human_gate.decision == "needs_human_review"
    assert output.validation_status == "needs_human_review"
    assert output.human_readable_summary is not None
    assert output.human_readable_summary.module_name == "protocol_design"
    assert "Phase IIb optimization study" in output.human_readable_summary.plain_language_summary
    assert any(finding.title == "Next study intent" for finding in output.human_readable_summary.key_findings)
    assert "Agent 3 run" in output.human_readable_summary.handoff_summary
    assert "Agent 4 run" in output.human_readable_summary.handoff_summary
    bundle = store.get_run_bundle(output.run_id)
    assert bundle.run is not None
    assert bundle.agent_outputs
    assert bundle.agent_traces
    assert {trace.agent_name for trace in bundle.agent_traces} >= {
        "ProtocolDesignManagerAgent",
        "DevelopmentStrategyAgent",
        "AnalogSearchPlannerAgent",
        "AnalogSelectionAgent",
        "AnalogBenchmarkInterpreterAgent",
        "EndpointStrategyAgent",
        "PopulationEligibilityAgent",
        "ComparatorDesignAgent",
        "SafetyMonitoringAgent",
        "StatisticalSkeletonAgent",
        "RegulatoryCriticAgent",
        "ProtocolBriefWriterAgent",
        "Agent5HumanReadableSummaryAgent",
    }
    assert any(agent_output.agent_name == "ProtocolBriefWriterAgent" for agent_output in bundle.agent_outputs)
    assert any(agent_output.agent_name == "Agent5HumanReadableSummaryAgent" for agent_output in bundle.agent_outputs)
    assert bundle.input_json is not None
    assert bundle.input_json["cli_input"]["nct_id"] == "NCT12345678"
    assert bundle.input_json["expanded_pipeline_input"]["target_trial"]["nct_id"] == "NCT12345678"
    assert bundle.input_json["expanded_pipeline_input"]["next_study_intent"]["proposed_next_stage"] == "Phase IIb optimization study"
    assert bundle.input_json["expanded_pipeline_input"]["agent3_output"]["output_id"] == agent3.output_id
    assert bundle.input_json["expanded_pipeline_input"]["agent4_output"]["output_id"] == agent4.output_id
    payload = json.loads(store._connection.execute("SELECT output_json FROM runs WHERE run_id = ?", (output.run_id,)).fetchone()["output_json"])
    assert payload["next_study_intent"]["proposed_next_stage"] == "Phase IIb optimization study"
    assert payload["protocol_design_brief"]["next_study_intent"]["study_role"]
    assert payload["analog_benchmark_bundle"]["selected_analog_ids"]
    assert payload["human_readable_summary"]["module_name"] == "protocol_design"

    html_path = write_run_html(output.run_id, "/tmp/protocol-design-agent5-test.html", memory=store)
    html = html_path.read_text(encoding="utf-8")
    assert "Agent Traces" in html
    assert "ProtocolBriefWriterAgent" in html
    assert "Next Study Intent" in html
    assert "Phase IIb optimization study" in html
    assert "Human-Readable Module Summary" in html


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

    html_path = tmp_path / "protocol_design.html"

    exit_code = main([
        "run",
        "protocol_design",
        "--nct-id",
        "NCT12345678",
        "--db-path",
        str(tmp_path / "memory.sqlite"),
        "--output-json",
        str(output_path),
        "--output-html",
        str(html_path),
    ])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["input"]["nct_id"] == "NCT12345678"
    assert payload["protocol_design_brief"]["requires_human_review"] is True
    assert payload["human_gate"]["decision"] == "needs_human_review"
    assert "Agent Traces" in html_path.read_text(encoding="utf-8")


def test_protocol_design_manager_offline_fallback_and_section_source_ids() -> None:
    agent3, _, agent4, _ = _handoffs()
    target = _target_trial()

    def fake_execute(search_plan, target_nct_id):
        analogs = (
            AnalogCandidateRecord(candidate_id="c1", trial=_analog("NCT00000001", enrollment=120), query_ids=("q1",), source_ids=("ctgov:NCT00000001",), provenance="test"),
            AnalogCandidateRecord(candidate_id="c2", trial=_analog("NCT00000002", enrollment=80), query_ids=("q1",), source_ids=("ctgov:NCT00000002",), provenance="test"),
        )
        return analogs, (_source("ctgov:NCT00000001", "clinical_trial_registry"), _source("ctgov:NCT00000002", "clinical_trial_registry")), ()

    def fake_calculate(trial, candidates, selection, search_plan):
        return calculate_analog_benchmark(run_id="manager-run", target_trial=trial, candidates=candidates, selection=selection, search_plan=search_plan)

    result = run_protocol_design_manager_agent(
        run_id="manager-run",
        target_trial=target,
        agent3_output=agent3,
        agent4_output=agent4,
        source_ids=(target.source_id, "agent_output:clinical_outcome_prediction:fixture", "agent_output:due_diligence:fixture"),
        assumptions=(),
        missing_data_flags=(),
        claims=(),
        top_k=10,
        execute_search_plan=fake_execute,
        calculate_benchmark=fake_calculate,
        config=AgentRuntimeConfig(disabled=True),
    )

    assert result.search_plan.queries
    assert len(result.search_plan.queries) >= 3
    assert result.selection.selected_analogs
    assert result.selection.excluded_candidates
    assert all(item.reason for item in result.selection.excluded_candidates)
    assert result.benchmark_bundle.enrollment.median == 100.0
    assert result.section_outputs
    assert all(output.source_ids for output in result.section_outputs)
    assert all(trace.provenance == "pharma_os.agent_runtime.offline" for trace in result.traces)


def test_protocol_section_assembly_uses_stable_ids_when_live_titles_change() -> None:
    agent3, _, agent4, _ = _handoffs()
    target = _target_trial()
    benchmark = _benchmark_for_trials((_analog("NCT00000001", enrollment=120),))
    intent = protocol_design_agents.build_next_study_intent(
        run_id="manager-run",
        target_trial=target,
        agent3_output=agent3,
        agent4_output=agent4,
        source_ids=(target.source_id,),
        missing_data_flags=(),
    )
    interpretation = protocol_design_agents._fallback_benchmark_interpretation(
        "manager-run",
        target,
        intent,
        benchmark,
    )
    specs = protocol_design_agents._section_agent_specs(
        run_id="manager-run",
        target_trial=target,
        agent3_output=agent3,
        agent4_output=agent4,
        next_study_intent=intent,
        benchmark_bundle=benchmark,
        benchmark_interpretation=interpretation,
        source_ids=(target.source_id,),
    )
    outputs = []
    for _, _, fallback_output in specs:
        sections = tuple(
            section.model_copy(update={"title": "Executive Summary"})
            if section.section_id.endswith("executive-synopsis")
            else section
            for section in fallback_output.sections
        )
        outputs.append(fallback_output.model_copy(update={"sections": sections}))

    grouped = protocol_design_agents._sections_by_brief_field(tuple(outputs))

    assert grouped["strategy_sections"]["executive_synopsis"].title == "Executive Summary"
    assert grouped["strategy_sections"]["study_design"].title == "Study Design"


def test_protocol_section_agent_falls_back_when_live_output_omits_required_section(monkeypatch) -> None:
    agent3, _, agent4, _ = _handoffs()
    target = _target_trial()
    benchmark = _benchmark_for_trials((_analog("NCT00000001", enrollment=120),))
    intent = protocol_design_agents.build_next_study_intent(
        run_id="manager-run",
        target_trial=target,
        agent3_output=agent3,
        agent4_output=agent4,
        source_ids=(target.source_id,),
        missing_data_flags=(),
    )
    interpretation = protocol_design_agents._fallback_benchmark_interpretation(
        "manager-run",
        target,
        intent,
        benchmark,
    )
    comparator_spec = next(
        spec
        for spec in protocol_design_agents._section_agent_specs(
            run_id="manager-run",
            target_trial=target,
            agent3_output=agent3,
            agent4_output=agent4,
            next_study_intent=intent,
            benchmark_bundle=benchmark,
            benchmark_interpretation=interpretation,
            source_ids=(target.source_id,),
        )
        if spec[0] == "ComparatorDesignAgent"
    )
    agent_name, instructions, fallback_output = comparator_spec
    bad_output = fallback_output.model_copy(
        update={"sections": tuple(section for section in fallback_output.sections if not section.section_id.endswith("executive-synopsis"))}
    )
    monkeypatch.setattr(
        protocol_design_agents,
        "_run_typed_agent",
        lambda **_: SimpleNamespace(output=bad_output),
    )

    result = protocol_design_agents._run_section_agent(
        run_id="manager-run",
        agent_name=agent_name,
        instructions=instructions,
        fallback_output=fallback_output,
        target_trial=target,
        agent3_output=agent3,
        agent4_output=agent4,
        next_study_intent=intent,
        benchmark_bundle=benchmark,
        benchmark_interpretation=interpretation,
        source_ids=(target.source_id,),
        config=AgentRuntimeConfig(disabled=True),
    )

    assert result.output == fallback_output
    assert any(section.title == "Executive Synopsis" for section in result.output.sections)
    assert result.trace_metadata["fallback_reason"].startswith("section_or_agent_mismatch")


def test_protocol_design_llm_base_payload_is_compact() -> None:
    agent3, _, agent4, _ = _handoffs()
    long_trial = _target_trial().model_copy(
        update={
            "eligibility_criteria": "Inclusion Criteria: " + ("long eligibility text " * 500),
            "locations": tuple(
                TrialLocation(facility=f"Site {index}", city="City", state="State", country=f"Country {index % 40}")
                for index in range(125)
            ),
        }
    )

    payload = protocol_design_agents._base_payload(
        long_trial,
        agent3,
        agent4,
        ("ctgov:NCT12345678", "agent_output:clinical_outcome_prediction:fixture", "agent_output:due_diligence:fixture"),
    )

    assert "locations" not in payload["target_trial"]
    assert payload["target_trial"]["locations_summary"]["site_count"] == 125
    assert len(payload["target_trial"]["locations_summary"]["sample_locations"]) == 8
    assert len(payload["target_trial"]["locations_summary"]["countries"]) <= 30
    assert len(payload["target_trial"]["eligibility_criteria"]) < 2600
    assert payload["agent3_context"]["output_id"] == agent3.output_id
    assert payload["agent4_context"]["output_id"] == agent4.output_id


def test_protocol_design_workflow_dedupes_unhashable_missing_flags(monkeypatch) -> None:
    agent3, handoff3, agent4, handoff4 = _handoffs()
    monkeypatch.setattr(protocol_design, "_get_or_run_agent3", lambda input_data, memory: (agent3, handoff3))
    monkeypatch.setattr(protocol_design, "_get_or_run_agent4", lambda input_data, memory: (agent4, handoff4))

    class FakeClient:
        def search_trials(self, input_data):
            analog = _analog("NCT00000001", enrollment=None)
            return ClinicalTrialsSearchResult(
                query=input_data,
                trials=(analog,),
                sources=(_source(analog.source_id, "clinical_trial_registry"),),
                api_url="https://clinicaltrials.gov/api/v2/studies",
            )

    monkeypatch.setattr(
        protocol_design,
        "execute_ctgov_search_plan",
        lambda search_plan, target_nct_id: execute_ctgov_search_plan(
            search_plan=search_plan,
            target_nct_id=target_nct_id,
            client=FakeClient(),
        ),
    )

    output = run_protocol_design_workflow(ProtocolDesignInput(nct_id="NCT12345678"), memory=MemoryStore(":memory:"))

    flag_ids = [flag.flag_id for flag in output.analog_benchmark_bundle.missing_data_flags]
    assert flag_ids
    assert len(flag_ids) == len(set(flag_ids))
    assert any(flag.field == "enrollment" for flag in output.analog_benchmark_bundle.missing_data_flags)


def test_protocol_design_validator_blocks_over_final_language(monkeypatch) -> None:
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
    output = run_protocol_design_workflow(ProtocolDesignInput(nct_id="NCT12345678"), memory=MemoryStore(":memory:"))
    bad_study_design = output.protocol_design_brief.study_design.model_copy(
        update={"body": "This is the final design and the protocol approved path."}
    )
    bad_brief = output.protocol_design_brief.model_copy(update={"study_design": bad_study_design})
    bad_output = output.model_copy(update={"protocol_design_brief": bad_brief})

    results = validate_protocol_design_constraints(run_id="guardrail", output=bad_output)

    assert any(result.status == "failed" and result.validator == "protocol_design_source_boundary" for result in results)


def test_protocol_design_reuses_agent3_and_agent4_handoffs(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    agent3 = _agent3_output()
    agent4 = _agent4_output()
    _save_agent_output(store, "clinical_outcome_prediction", agent3, tuple(source.source_id for source in agent3.sources))
    _save_agent_output(store, "due_diligence", agent4, tuple(source.source_id for source in agent4.sources))
    monkeypatch.setattr(protocol_design, "run_clinical_outcome_prediction_workflow", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Agent 3 should be reused")))
    monkeypatch.setattr(protocol_design, "run_due_diligence_workflow", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Agent 4 should be reused")))

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

    output = run_protocol_design_workflow(ProtocolDesignInput(nct_id="NCT12345678"), memory=store)

    assert output.agent3_handoff.generated_or_reused == "reused"
    assert output.agent4_handoff.generated_or_reused == "reused"


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
