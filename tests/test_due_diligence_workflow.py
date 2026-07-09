from __future__ import annotations

import json
from datetime import datetime, timezone

from pharma_os.cli import main
from pharma_os.memory import MemoryStore
from pharma_os.agents import due_diligence as due_agent
from pharma_os.schemas import (
    AgentOutput,
    ApprovalLikelihoodProxy,
    AssetIdentityOutput,
    ClinicalOutcomePredictionInput,
    ClinicalOutcomePredictionOutput,
    ClinicalEvidenceSummary,
    ClinicalTrialRecord,
    ComparatorBenchmarkBundle,
    CompetitiveLandscapeSummary,
    DueDiligenceInput,
    DueDiligenceSynthesisOutput,
    EndpointRiskAssessment,
    EnrollmentDurationRisk,
    EvidenceClaim,
    FailureModeClassification,
    HistoricalPoSEstimate,
    LabelExpansionClinicalRationale,
    PatentExclusivityOutput,
    PoSOutput,
    PricingOutput,
    SafetyContext,
    SafetyLabelSummary,
    SourceMetadata,
    SourceAvailabilityReport,
    TrialIntervention,
    TrialDesignFeatures,
    TrialIdentity,
    TrialSponsor,
    WorkflowRun,
)
from pharma_os.workflows import due_diligence
from pharma_os.workflows.due_diligence import run_due_diligence_workflow
from pharma_os.validators import validate_due_diligence_constraints, validate_source_coverage


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


def _agent3_output(
    *,
    run_id: str = "agent3-run",
    asset_name: str = "Examplemab",
    phase: str = "PHASE2",
    source_id: str = "ctgov:NCT12345678",
    pos_source_id: str = "pos_workbook:fixture",
) -> ClinicalOutcomePredictionOutput:
    source = _source(source_id)
    return ClinicalOutcomePredictionOutput(
        output_id=f"clinical-outcome-prediction-output-{run_id}",
        run_id=run_id,
        input=ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        trial_identity=TrialIdentity(
            nct_id="NCT12345678",
            phases=(phase,),
            conditions=("Glioblastoma",),
            sponsor="Example Bio",
            source_ids=(source_id,),
        ),
        asset_identity=AssetIdentityOutput(
            nct_id="NCT12345678",
            asset_name=asset_name,
            sponsor="Example Bio",
            normalized_indication="glioblastoma",
            source_ids=(source_id,),
            confidence=0.9,
        ),
        trial_design_features=TrialDesignFeatures(primary_endpoint_count=0, source_ids=(source_id,)),
        endpoint_risk_assessment=EndpointRiskAssessment(risk_level="low", rationale="Fixture endpoint risk.", source_ids=(source_id,), confidence=0.8),
        enrollment_duration_risk=EnrollmentDurationRisk(risk_level="low", rationale="Fixture enrollment risk.", source_ids=(source_id,), confidence=0.8),
        comparator_benchmarking=ComparatorBenchmarkBundle(benchmark_summary="Fixture comparator benchmark.", source_ids=(source_id,), confidence=0.7),
        historical_pos_estimate=HistoricalPoSEstimate(
            probability_of_success=0.344,
            current_phase="Phase II",
            disease_area="Oncology",
            lookup_key="Disease Area|Oncology|Phase II",
            assumption_type="source_derived",
            source_ids=(pos_source_id,),
            confidence=0.9,
        ),
        approval_likelihood_proxy=ApprovalLikelihoodProxy(
            probability=0.344,
            basis="Fixture proxy.",
            assumption_type="source_derived",
            source_ids=(pos_source_id,),
            confidence=0.9,
        ),
        failure_mode_classification=FailureModeClassification(overall_risk_level="low", source_ids=(source_id,), confidence=0.7),
        safety_context=SafetyContext(label_available=False, confidence=0.0),
        label_expansion_clinical_rationale=LabelExpansionClinicalRationale(rationale="Fixture label rationale.", source_ids=(source_id,), confidence=0.5),
        source_availability=SourceAvailabilityReport(),
        sources=(source, _source(pos_source_id)),
        confidence=0.8,
        validation_status="passed",
    )


def _save_agent3_output(store: MemoryStore, output: ClinicalOutcomePredictionOutput) -> None:
    run = WorkflowRun(
        run_id=output.run_id,
        workflow_name="clinical_outcome_prediction",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        input_provenance="test.agent3",
        source_ids=tuple(source.source_id for source in output.sources),
        validation_status="passed",
        metadata={"nct_id": output.input.nct_id},
    )
    store.save_run(run, input_payload=output.input, output_payload=output)
    store.save_sources(output.run_id, output.sources)
    store.save_agent_output(
        AgentOutput(
            output_id=f"agent-output-{output.run_id}",
            agent_name="clinical_outcome_prediction_agent",
            run_id=output.run_id,
            provenance="test",
            claims=output.claims,
            sources=output.sources,
            confidence=output.confidence,
            validation_status="passed",
        ),
        payload=output,
    )


def _install_due_diligence_fixtures(monkeypatch, *, agent3_output: ClinicalOutcomePredictionOutput | None = None) -> None:
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
    monkeypatch.setattr(
        due_diligence,
        "build_clinical_evidence_summary",
        lambda run_id, trial, asset: (
            ClinicalEvidenceSummary(
                nct_id=trial.nct_id,
                ctgov_summary="Fixture CT.gov evidence.",
                pubmed_query='"NCT12345678" OR "Examplemab"',
                pubmed_article_count=1,
                pubmed_titles=("Examplemab glioblastoma evidence",),
                source_ids=("ctgov:NCT12345678", "pubmed:123"),
                confidence=0.8,
            ),
            (_source("pubmed:123"),),
            (
                EvidenceClaim(
                    claim_id=f"claim-{run_id}-pubmed-123",
                    claim_text="PubMed article 123 was retrieved for the due-diligence evidence query.",
                    source_ids=("pubmed:123",),
                    provenance="test.pubmed",
                    confidence=0.8,
                    confidence_level="medium",
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        due_diligence,
        "build_safety_label_summary",
        lambda asset: (
            SafetyLabelSummary(
                asset_name=asset.asset_name,
                label_available=True,
                warnings_summary="Fixture warnings.",
                adverse_reactions_summary="Fixture adverse reactions.",
                source_ids=("openfda_label:examplemab",),
                confidence=0.7,
            ),
            (_source("openfda_label:examplemab"),),
        ),
    )
    if agent3_output is not None:
        monkeypatch.setattr(due_diligence, "run_clinical_outcome_prediction_workflow", lambda input_data, memory=None: agent3_output)


def _due_input(**updates) -> DueDiligenceInput:
    payload = {
        "nct_id": "NCT12345678",
        "annual_patients": 1000,
        "peak_penetration": 0.2,
        "gross_to_net": 0.15,
        "operating_margin": 0.35,
        "discount_rate": 0.1,
        "development_cost": 50_000_000,
        "launch_year": 2029,
        "loe_year": 2040,
    }
    payload.update(updates)
    return DueDiligenceInput(**payload)


def test_due_diligence_workflow_persists_bundle(monkeypatch) -> None:
    _install_due_diligence_fixtures(monkeypatch, agent3_output=_agent3_output(run_id="generated-agent3"))

    store = MemoryStore(":memory:")
    output = run_due_diligence_workflow(_due_input(), memory=store)

    bundle = store.get_run_bundle(output.run_id)
    assert bundle.run is not None
    assert output.asset_identity.asset_name == "Examplemab"
    assert output.pos.probability_of_success == 0.344
    assert output.agent3_handoff.agent3_run_id == "generated-agent3"
    assert output.clinical_risk_summary.endpoint_risk_level == "low"
    assert output.clinical_evidence.pubmed_article_count == 1
    assert output.competitive_landscape.benchmark_summary == "Fixture comparator benchmark."
    assert output.safety_label_summary.label_available is True
    assert output.patent_loe_review.estimated_loe_year == 2040
    assert output.pricing.annual_wac == 100000.0
    assert output.commercial_model.calculable
    assert output.rnpv.calculable
    assert output.red_flags
    assert output.asset_memo.requires_human_review is True
    assert output.human_readable_summary is not None
    assert output.human_readable_summary.module_name == "due_diligence"
    assert "Agent 5" in output.human_readable_summary.handoff_summary
    assert bundle.sources
    assert bundle.claims
    assert bundle.validation_results
    assert bundle.agent_outputs
    assert bundle.agent_traces
    assert {trace.agent_name for trace in bundle.agent_traces} >= {
        "DueDiligenceManagerAgent",
        "ClinicalEvidenceSynthesisAgent",
        "CompetitiveLandscapeAgent",
        "SafetyDiligenceAgent",
        "IPLOECriticAgent",
        "CommercialAssumptionsCriticAgent",
        "DiligenceRedTeamAgent",
        "AssetMemoAgent",
        "Agent4HumanReadableSummaryAgent",
    }
    assert any(agent_output.agent_name == "AssetMemoAgent" for agent_output in bundle.agent_outputs)
    assert any(agent_output.agent_name == "Agent4HumanReadableSummaryAgent" for agent_output in bundle.agent_outputs)
    assert all(trace.provenance == "pharma_os.agent_runtime.offline" for trace in bundle.agent_traces)
    payload = json.loads(store._connection.execute("SELECT output_json FROM runs WHERE run_id = ?", (output.run_id,)).fetchone()["output_json"])
    assert payload["human_readable_summary"]["module_name"] == "due_diligence"
    assert bundle.reports


def test_due_diligence_synthesis_agent_uses_fallback_for_wrong_section(monkeypatch) -> None:
    fallback = DueDiligenceSynthesisOutput(
        output_id="ip-fallback",
        agent_name="IPLOECriticAgent",
        section="ip_loe",
        synthesis="Fallback IP/LOE review.",
        source_ids=("lens:fixture",),
        confidence=0.6,
    )
    wrong = DueDiligenceSynthesisOutput(
        output_id="wrong-section",
        agent_name="SafetyDiligenceAgent",
        section="safety",
        synthesis="Wrong section output.",
        source_ids=("label:fixture",),
        confidence=0.6,
    )

    def fake_runtime(**kwargs):
        return due_agent._synthesis_fallback_result(
            agent_name=kwargs["agent_name"],
            run_id=kwargs["run_id"],
            input_summary=kwargs["input_summary"],
            fallback_output=wrong,
            source_ids=kwargs["source_ids"],
            confidence=kwargs["confidence"],
            rationale_summary=kwargs["rationale_summary"],
            reason="test_wrong_section",
        )

    monkeypatch.setattr(due_agent, "_run_typed_agent", fake_runtime)

    result = due_agent._run_synthesis_agent(
        agent_name="IPLOECriticAgent",
        section="ip_loe",
        instructions="Review IP/LOE.",
        fallback_output=fallback,
        run_id="RUN",
        source_ids=("lens:fixture",),
        confidence=0.6,
        payload={},
        config=due_agent.AgentRuntimeConfig(disabled=False),
    )

    assert result.output.section == "ip_loe"
    assert result.output.agent_name == "IPLOECriticAgent"
    assert result.output.synthesis == "Fallback IP/LOE review."
    assert result.trace_metadata["fallback_reason"].startswith("section_or_agent_mismatch")


def test_due_diligence_synthesis_agent_uses_fallback_for_runtime_exception(monkeypatch) -> None:
    fallback = DueDiligenceSynthesisOutput(
        output_id="commercial-fallback",
        agent_name="CommercialAssumptionsCriticAgent",
        section="commercial_assumptions",
        synthesis="Fallback commercial review.",
        source_ids=("pricing:fixture",),
        confidence=0.5,
    )

    def fail_runtime(**kwargs):
        raise RuntimeError("live call failed")

    monkeypatch.setattr(due_agent, "_run_typed_agent", fail_runtime)

    result = due_agent._run_synthesis_agent(
        agent_name="CommercialAssumptionsCriticAgent",
        section="commercial_assumptions",
        instructions="Review commercial assumptions.",
        fallback_output=fallback,
        run_id="RUN",
        source_ids=("pricing:fixture",),
        confidence=0.5,
        payload={},
        config=due_agent.AgentRuntimeConfig(disabled=False),
    )

    assert result.output.section == "commercial_assumptions"
    assert result.output.agent_name == "CommercialAssumptionsCriticAgent"
    assert result.output.synthesis == "Fallback commercial review."
    assert result.trace_metadata["fallback_reason"] == "runtime_exception"


def test_due_diligence_cli_persists_report(monkeypatch, tmp_path) -> None:
    _install_due_diligence_fixtures(monkeypatch, agent3_output=_agent3_output(run_id="cli-agent3"))
    db_path = tmp_path / "memory.sqlite"
    output_path = tmp_path / "due.json"
    html_path = tmp_path / "due.html"
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
            "--refresh-agent3",
            "--db-path",
            str(db_path),
            "--output-json",
            str(output_path),
            "--output-html",
            str(html_path),
        ]
    )
    assert exit_code == 0
    assert html_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "Agent Traces" in html
    assert "AssetMemoAgent" in html
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["input"]["refresh_agent3"] is True
    assert payload["agent3_handoff"]["agent3_run_id"] == "cli-agent3"
    assert payload["clinical_risk_summary"]["nct_id"] == "NCT12345678"
    assert payload["clinical_evidence"]["pubmed_article_count"] == 1
    assert payload["asset_memo"]["requires_human_review"] is True
    report_path = tmp_path / "report.json"
    exit_code = main(["report", "--run-id", payload["run_id"], "--db-path", str(db_path), "--output-json", str(report_path)])
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_id"] == payload["run_id"]
    assert report["sources"]
    assert any("PubMed article" in claim["claim_text"] for claim in report["claims"])
    assert "Scientific Memory contains" in report["summary"]


def test_due_diligence_reuses_existing_agent3_output(monkeypatch) -> None:
    _install_due_diligence_fixtures(monkeypatch, agent3_output=_agent3_output(run_id="should-not-run"))
    store = MemoryStore(":memory:")
    _save_agent3_output(store, _agent3_output(run_id="existing-agent3"))

    output = run_due_diligence_workflow(_due_input(), memory=store)

    assert output.agent3_handoff.agent3_run_id == "existing-agent3"
    assert output.agent3_handoff.generated_or_reused == "reused"
    assert output.agent3_handoff.retrieved_from_memory is True
    assert output.clinical_risk_summary.nct_id == "NCT12345678"


def test_due_diligence_refresh_agent3_forces_fresh_run(monkeypatch) -> None:
    calls = {"count": 0}

    def fresh_agent3(input_data, memory=None):
        calls["count"] += 1
        return _agent3_output(run_id="fresh-agent3")

    _install_due_diligence_fixtures(monkeypatch)
    monkeypatch.setattr(due_diligence, "run_clinical_outcome_prediction_workflow", fresh_agent3)
    store = MemoryStore(":memory:")
    _save_agent3_output(store, _agent3_output(run_id="existing-agent3"))

    output = run_due_diligence_workflow(_due_input(refresh_agent3=True), memory=store)

    assert calls["count"] == 1
    assert output.agent3_handoff.agent3_run_id == "fresh-agent3"
    assert output.agent3_handoff.generated_or_reused == "generated"
    assert output.agent3_handoff.retrieved_from_memory is False


def test_due_diligence_generates_agent3_when_missing(monkeypatch) -> None:
    calls = {"count": 0}

    def generated_agent3(input_data, memory=None):
        calls["count"] += 1
        return _agent3_output(run_id="generated-agent3")

    _install_due_diligence_fixtures(monkeypatch)
    monkeypatch.setattr(due_diligence, "run_clinical_outcome_prediction_workflow", generated_agent3)

    output = run_due_diligence_workflow(_due_input(), memory=MemoryStore(":memory:"))

    assert calls["count"] == 1
    assert output.agent3_handoff.agent3_run_id == "generated-agent3"
    assert output.agent3_handoff.generated_or_reused == "generated"
    assert output.clinical_risk_summary.historical_pos == 0.344


def test_cross_agent_consistency_passes_when_fields_match(monkeypatch) -> None:
    _install_due_diligence_fixtures(monkeypatch, agent3_output=_agent3_output(run_id="matching-agent3"))

    output = run_due_diligence_workflow(_due_input(), memory=MemoryStore(":memory:"))

    cross_agent = [result for result in output.validation_results if result.validator == "cross_agent_consistency"]
    assert cross_agent
    assert all(result.status == "passed" for result in cross_agent)


def test_cross_agent_consistency_flags_mismatched_asset_phase_and_sources(monkeypatch) -> None:
    _install_due_diligence_fixtures(
        monkeypatch,
        agent3_output=_agent3_output(
            run_id="mismatched-agent3",
            asset_name="Othermab",
            phase="PHASE3",
            source_id="ctgov:NCT87654321",
            pos_source_id="pos_workbook:other",
        ),
    )

    output = run_due_diligence_workflow(_due_input(), memory=MemoryStore(":memory:"))

    failed = [result for result in output.validation_results if result.validator == "cross_agent_consistency" and result.status == "failed"]
    assert {result.validation_id.rsplit("-", 1)[-1] for result in failed} >= {"asset_name", "phase", "source_ids"}
    assert any(flag.provenance == "pharma_os.validators.generate_confidence_flags.validation" for flag in output.confidence_flags)
    assert output.human_gate is not None


def test_due_diligence_pubmed_synthesis_stays_with_metadata(monkeypatch) -> None:
    _install_due_diligence_fixtures(monkeypatch, agent3_output=_agent3_output(run_id="pubmed-agent3"))

    output = run_due_diligence_workflow(_due_input(), memory=MemoryStore(":memory:"))

    assert "Examplemab glioblastoma evidence" in output.clinical_evidence.ctgov_summary
    assert "full text" not in output.clinical_evidence.ctgov_summary.casefold()
    assert output.clinical_evidence.source_ids


def test_due_diligence_unsourced_claims_fail_validation() -> None:
    claim = EvidenceClaim(
        claim_id="claim-unsourced",
        claim_text="Unsourced diligence claim with 42 patients.",
        source_ids=("missing-source",),
        provenance="test",
        confidence=0.1,
        confidence_level="low",
    )

    result = validate_source_coverage(
        target_id="due-output",
        claims=(claim,),
        source_ids=set(),
        run_id="RUN",
    )

    assert result.status == "failed"


def test_due_diligence_missing_loe_and_commercial_inputs_create_flags_and_questions(monkeypatch) -> None:
    _install_due_diligence_fixtures(monkeypatch, agent3_output=_agent3_output(run_id="missing-agent3"))

    output = run_due_diligence_workflow(DueDiligenceInput(nct_id="NCT12345678"), memory=MemoryStore(":memory:"))

    assert any(flag.category in {"ip_loe", "rnpv"} for flag in output.red_flags)
    assert any("LOE" in question for question in output.asset_memo.review_questions)
    assert any("rNPV" in question for question in output.asset_memo.review_questions)


def test_due_diligence_guardrail_blocks_decision_language(monkeypatch) -> None:
    _install_due_diligence_fixtures(monkeypatch, agent3_output=_agent3_output(run_id="guardrail-agent3"))
    output = run_due_diligence_workflow(_due_input(), memory=MemoryStore(":memory:"))
    bad_memo = output.asset_memo.model_copy(
        update={"summary": "This is a recommended investment and go/no-go decision."}
    )
    bad_output = output.model_copy(update={"asset_memo": bad_memo})

    results = validate_due_diligence_constraints(run_id="RUN", output=bad_output)

    assert any(result.status == "failed" and result.validator == "due_diligence_guardrails" for result in results)
