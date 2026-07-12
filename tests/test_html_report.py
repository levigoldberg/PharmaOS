from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pharma_os.due_diligence_report import build_due_diligence_report_payload
from pharma_os.html_report import build_nct_report_html, build_run_html, write_nct_report, write_nct_report_if_persistent, write_run_html
from pharma_os.memory import MemoryStore
from pharma_os.schemas import AgentRunTrace, WorkflowRun


def test_html_generation_for_saved_run(tmp_path) -> None:
    store = MemoryStore(":memory:")
    run = WorkflowRun(
        run_id="RUNHTML",
        workflow_name="trial_intelligence",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        input_provenance="test",
        validation_status="passed",
    )
    store.save_run(run, input_payload={"disease": "glioblastoma"}, output_payload={"output_id": "OUT"}, trace_metadata={"mode": "test"})
    store.save_agent_trace(
        AgentRunTrace(
            trace_id="TRACE1",
            run_id="RUNHTML",
            agent_name="FixtureAgent",
            output_type="FixtureOutput",
            provenance="test",
            execution_mode="deterministic_fallback",
            model="gpt-test",
            model_route="control_tower",
            retry_count=2,
            retry_attempts=3,
            retry_exhausted=True,
            fallback_cause="rate_limit",
            final_retry_reason="429",
        )
    )

    html = build_run_html("RUNHTML", memory=store)
    assert "Run Metadata" in html
    assert "Input JSON" in html
    assert "Trace Metadata" in html
    assert "glioblastoma" in html
    assert "Runtime Routes And Retries" in html
    assert "control_tower" in html
    assert "gpt-test" in html
    assert "rate_limit" in html

    output_path = write_run_html("RUNHTML", tmp_path / "run.html", memory=store)
    assert output_path.exists()
    assert "Raw Bundle JSON" in output_path.read_text(encoding="utf-8")


def test_due_diligence_html_renders_panoptic_style_charts() -> None:
    store = MemoryStore(":memory:")
    output = {
        "output_id": "DUEOUT",
        "run_id": "DUERUN",
        "target_trial": {
            "nct_id": "NCT12345678",
            "brief_title": "Example trial",
            "overall_status": "RECRUITING",
            "phases": ["PHASE2"],
            "conditions": ["Atopic Dermatitis"],
        },
        "asset_identity": {
            "asset_name": "Example",
            "sponsor": "Example Bio",
            "normalized_indication": "atopic dermatitis",
        },
        "asset_memo": {
            "title": "Draft diligence memo",
            "summary": "Source-backed draft memo.",
            "review_questions": [],
        },
        "pricing": {"matched_product": "Dupixent", "annual_wac": 100000.0, "wac_value": 4000.0, "wac_unit_basis": "package", "confidence": 0.8},
        "pos": {"probability_of_success": 0.2},
        "commercial_model": {
            "calculable": True,
            "annual_patients": 1000000.0,
            "peak_penetration": 0.1,
            "gross_to_net": 0.2,
            "net_price": 80000.0,
            "peak_net_sales": 8000000000.0,
            "selected_population_measure": {"value": 1000000.0},
            "patient_funnel": {
                "starting_population": 1000000.0,
                "diagnosed_patients": 800000.0,
                "treated_or_managed_patients": 600000.0,
                "eligible_patients": 400000.0,
                "commercially_addressable_patients": 300000.0,
                "diagnosed_fraction": 0.8,
                "treated_fraction": 0.75,
                "eligibility_fraction": 0.6667,
                "commercially_addressable_fraction": 0.75,
            },
            "revenue_forecast": [
                {"year": 1, "treated_patients": 3000.0, "net_price": 80000.0, "net_revenue": 240000000.0},
                {"year": 2, "treated_patients": 9000.0, "net_price": 80000.0, "net_revenue": 720000000.0},
                {"year": 3, "treated_patients": 30000.0, "net_price": 80000.0, "net_revenue": 2400000000.0},
            ],
            "commercial_input_bundle_summary": {
                "us_population_denominator": {
                    "total_us_population": 340110990,
                    "adult_population": 267181678,
                    "source_id": "census:fixture",
                    "source_type": "structured_api",
                    "source_year": 2024,
                    "human_review_required": False,
                },
                "market_query_diagnostics": [
                    {"query": "atopic dermatitis prevalence United States", "status": "ok", "article_count": 3}
                ],
            },
            "cases": {
                "downside": {
                    "revenue_forecast": [
                        {"year": 1, "treated_patients": 1500.0, "net_price": 70000.0, "net_revenue": 105000000.0},
                        {"year": 2, "treated_patients": 4500.0, "net_price": 70000.0, "net_revenue": 315000000.0},
                    ]
                },
                "upside": {
                    "revenue_forecast": [
                        {"year": 1, "treated_patients": 5000.0, "net_price": 90000.0, "net_revenue": 450000000.0},
                        {"year": 2, "treated_patients": 15000.0, "net_price": 90000.0, "net_revenue": 1350000000.0},
                    ]
                },
            },
        },
        "rnpv": {
            "calculable": True,
            "rnpv": 100000000.0,
            "probability_of_success": 0.2,
            "launch_year": 2030,
            "loe_year": 2040,
            "discount_rate": 0.12,
            "operating_margin": 0.65,
            "development_cost": 250000000.0,
            "assumptions": [
                {"name": "tax_rate", "value": 0.21},
                {"name": "valuation_year", "value": 2026},
            ],
        },
        "red_flags": [],
        "missing_data_flags": [],
        "confidence": 0.75,
        "validation_status": "needs_human_review",
    }
    output["investment_report"] = build_due_diligence_report_payload(output)
    run = WorkflowRun(
        run_id="DUERUN",
        workflow_name="due_diligence",
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        input_provenance="test",
        validation_status="needs_human_review",
    )
    store.save_run(run, input_payload={"nct_id": "NCT12345678"}, output_payload=output)

    html = build_run_html("DUERUN", memory=store)

    assert "Panoptic-Style Investment Snapshot" in html
    assert "Revenue Forecast" in html
    assert "Patient Funnel" in html
    assert "rNPV Sensitivity" in html
    assert "Market Evidence Diagnostics" in html
    assert "atopic dermatitis prevalence United States" in html
    assert "chart-spec" in html


def test_cumulative_nct_report_uses_latest_outputs_in_lifecycle_order(tmp_path) -> None:
    store = MemoryStore(":memory:")
    nct_id = "NCT99999999"
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _save_cumulative_report_fixtures(store, nct_id, base_time)

    html = build_nct_report_html(nct_id, memory=store)

    assert "CurrentAsset" in html
    assert "OldAsset" not in html
    assert "Agent 3 - Clinical Outcome Prediction" in html
    assert "Agent 4 - Due Diligence" in html
    assert "Agent 5 - Protocol Design" in html
    assert html.index("Agent 3 - Clinical Outcome Prediction") < html.index("Agent 4 - Due Diligence")
    assert html.index("Agent 4 - Due Diligence") < html.index("Agent 5 - Protocol Design")
    assert "Raw Bundle JSON" not in html
    assert "Output JSON" not in html

    output_path = tmp_path / f"{nct_id}.html"
    output_path.write_text("old report", encoding="utf-8")
    written = write_nct_report(nct_id, output_path, memory=store)
    assert written == output_path
    assert "old report" not in output_path.read_text(encoding="utf-8")


def test_cumulative_nct_report_finds_nested_orchestration_payload_ncts() -> None:
    store = MemoryStore(":memory:")
    nct_id = "NCT77777777"
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _save_workflow_output(
        store,
        "NESTED_A3",
        "clinical_outcome_prediction",
        nct_id,
        base_time,
        _agent3_output(nct_id, "NestedAsset"),
        metadata={},
        input_payload={"cli_input": {"nct_id": nct_id}},
    )
    _save_workflow_output(
        store,
        "NESTED_A4",
        "due_diligence",
        nct_id,
        base_time + timedelta(minutes=1),
        _agent4_output(nct_id),
        validation_status="needs_human_review",
        metadata={},
        input_payload={"cli_input": {"nct_id": nct_id}},
    )
    _save_workflow_output(
        store,
        "NESTED_A5",
        "protocol_design",
        nct_id,
        base_time + timedelta(minutes=2),
        _agent5_output(nct_id),
        validation_status="needs_human_review",
        metadata={},
        input_payload={"cli_input": {"nct_id": nct_id}},
    )

    html = build_nct_report_html(nct_id, memory=store)

    assert "Agent 3 - Clinical Outcome Prediction" in html
    assert "Agent 4 - Due Diligence" in html
    assert "Agent 5 - Protocol Design" in html
    assert "NestedAsset" in html


def test_cumulative_nct_report_persistent_default_path(tmp_path) -> None:
    nct_id = "NCT88888888"
    store = MemoryStore(tmp_path / ".pharma_os" / "scientific_memory.sqlite")
    _save_cumulative_report_fixtures(store, nct_id, datetime(2026, 1, 1, tzinfo=timezone.utc))

    output_path = write_nct_report_if_persistent(nct_id, memory=store)

    assert output_path == tmp_path / "reports" / f"{nct_id}.html"
    assert output_path.exists()
    assert "PharmaOS Cumulative Report" in output_path.read_text(encoding="utf-8")
    assert write_nct_report_if_persistent(nct_id, memory=MemoryStore(":memory:")) is None

    explicit_store = MemoryStore(tmp_path / "memory.sqlite")
    _save_cumulative_report_fixtures(explicit_store, nct_id, datetime(2026, 1, 2, tzinfo=timezone.utc))
    explicit_path = write_nct_report_if_persistent(nct_id, memory=explicit_store)
    assert explicit_path == tmp_path / "reports" / f"{nct_id}.html"


def _save_cumulative_report_fixtures(store: MemoryStore, nct_id: str, base_time: datetime) -> None:
    _save_workflow_output(
        store,
        "OLD_A3",
        "clinical_outcome_prediction",
        nct_id,
        base_time,
        _agent3_output(nct_id, "OldAsset"),
        metadata={"artifact_lineage_status": "superseded", "nct_id": nct_id},
    )
    _save_workflow_output(
        store,
        "NEW_A3",
        "clinical_outcome_prediction",
        nct_id,
        base_time + timedelta(minutes=1),
        _agent3_output(nct_id, "CurrentAsset"),
    )
    _save_workflow_output(
        store,
        "A4",
        "due_diligence",
        nct_id,
        base_time + timedelta(minutes=2),
        _agent4_output(nct_id),
        validation_status="needs_human_review",
    )
    _save_workflow_output(
        store,
        "A5",
        "protocol_design",
        nct_id,
        base_time + timedelta(minutes=3),
        _agent5_output(nct_id),
        validation_status="needs_human_review",
    )


def _save_workflow_output(
    store: MemoryStore,
    run_id: str,
    workflow_name: str,
    nct_id: str,
    completed_at: datetime,
    output: dict,
    *,
    validation_status: str = "passed",
    metadata: dict | None = None,
    input_payload: dict | None = None,
) -> None:
    run = WorkflowRun(
        run_id=run_id,
        workflow_name=workflow_name,
        status="completed",
        started_at=completed_at,
        completed_at=completed_at,
        input_provenance="test",
        validation_status=validation_status,
        metadata={"nct_id": nct_id} if metadata is None else metadata,
    )
    output["run_id"] = run_id
    output["validation_status"] = validation_status
    store.save_run(run, input_payload=input_payload or {"nct_id": nct_id}, output_payload=output)


def _agent3_output(nct_id: str, asset_name: str) -> dict:
    return {
        "input": {"nct_id": nct_id},
        "trial_identity": {
            "nct_id": nct_id,
            "brief_title": "Current phase 2 trial",
            "overall_status": "ACTIVE_NOT_RECRUITING",
            "phases": ["PHASE2"],
            "conditions": ["Atopic Dermatitis"],
        },
        "asset_identity": {"asset_name": asset_name, "normalized_indication": "atopic dermatitis"},
        "endpoint_risk_assessment": {"risk_level": "low", "rationale": "Primary endpoint is standard for this setting.", "missing_data_flags": []},
        "enrollment_duration_risk": {"risk_level": "medium", "rationale": "Enrollment assumptions depend on site activation.", "missing_data_flags": []},
        "comparator_benchmarking": {"matched_public_trials_count": 2, "benchmark_summary": "Two relevant comparator trials were found."},
        "historical_pos_estimate": {"probability_of_success": 0.25, "lookup_key": "Disease Area|Autoimmune|Phase II"},
        "approval_likelihood_proxy": {"probability": 0.25, "basis": "historical phase/area proxy"},
        "failure_mode_classification": {"likely_failure_modes": []},
        "safety_context": {"summary": "No public label-derived safety profile was available."},
        "sources": [],
        "claims": [],
        "confidence_flags": [],
        "confidence": 0.5,
    }


def _agent4_output(nct_id: str) -> dict:
    return {
        "target_trial": {
            "nct_id": nct_id,
            "brief_title": "Current phase 2 trial",
            "overall_status": "ACTIVE_NOT_RECRUITING",
            "phases": ["PHASE2"],
            "primary_endpoints": [{"measure": "EASI", "time_frame": "Week 24"}],
        },
        "asset_identity": {"asset_name": "CurrentAsset", "normalized_indication": "atopic dermatitis"},
        "asset_memo": {
            "summary": "Phase 2 placebo-controlled study with unresolved safety and commercial evidence gaps.",
            "review_questions": ["Confirm whether the missing safety label changes the diligence conclusion."],
        },
        "clinical_risk_summary": {"endpoint_risk_level": "low", "enrollment_duration_risk_level": "medium", "historical_pos": 0.25},
        "clinical_evidence": {"pubmed_article_count": 0},
        "competitive_landscape": {"benchmark_summary": "Comparator search found two relevant Phase 2 trials."},
        "safety_label_summary": {"warnings_summary": "No label-derived warnings were available."},
        "patent_loe_review": {"estimated_loe_year": 2040, "review_summary": "LOE year is estimated at 2040; counsel review is required."},
        "pricing": {"matched_product": "DUPIXENT (pricing analog: approved AD biologic)", "annual_wac": 100000.0},
        "pos": {"probability_of_success": 0.25},
        "commercial_model": {"calculable": False, "peak_net_sales": None},
        "rnpv": {"calculable": False, "rnpv": None, "loe_year": 2040},
        "red_flags": [{"severity": "high", "reason": "Commercial model is non-calculable under current evidence and assumptions."}],
        "missing_data_flags": [{"severity": "medium", "reason": "Public openFDA safety label context is missing or unavailable."}],
        "human_gate": {"decision": "needs_human_review"},
        "sources": [],
        "claims": [],
        "confidence_flags": [],
        "confidence": 0.4,
    }


def _agent5_output(nct_id: str) -> dict:
    return {
        "input": {"nct_id": nct_id},
        "next_study_intent": {
            "indication": "atopic dermatitis",
            "proposed_next_stage": "analog-follow-on-derived study (Phase 2 interventional efficacy/safety signal study.)",
            "study_role": "Adult moderate-to-severe AD.",
            "key_clinical_question": "Which protocol assumptions are supported by analog evidence?",
        },
        "analog_benchmark_bundle": {
            "selected_analog_ids": ["NCT11111111"],
            "enrollment": {"median": 22, "unit": "participants"},
            "planned_duration_months": {"median": 19.7, "unit": "months"},
            "comparator_categories": [{"label": "placebo_control"}],
            "limitations": ["Fewer than five selected analog trials were available, limiting benchmark stability."],
            "confidence": 0.58,
        },
        "follow_on_trials": [],
        "analog_derived_design_decisions": [
            {"field_name": "proposed_enrollment", "proposed_value": "22 participants", "derivation_method": "median"},
            {"field_name": "comparator_control", "proposed_value": "placebo_control", "derivation_method": "frequency"},
        ],
        "protocol_design_brief": {
            "executive_synopsis": {"body": "Draft next-study strategy brief requiring human review before protocol use."},
            "strategic_rationale": {"body": "Next-study rationale was created after analog selection and benchmark synthesis. It did not drive analog retrieval."},
            "endpoint_strategy": {"body": "Endpoint strategy requires statistical review before hierarchy or powering assumptions are used."},
            "operational_feasibility_risks": {"body": "Operational risks should be reviewed against enrollment, duration, sites, and schedule burden."},
            "reviewer_critique": {
                "limitations": ["No clear same-asset/same-sponsor follow-on lineage was supported."],
                "statistical_questions": ["What is the primary estimand?"],
                "regulatory_questions": ["Is the comparator strategy adequately supported?"],
            },
            "human_review_questions": ["Is one analog enough to support design assumptions?"],
        },
        "missing_data_flags": [],
        "human_gate": {"decision": "needs_human_review"},
        "sources": [],
        "claims": [],
        "confidence_flags": [],
        "confidence": 0.58,
    }
