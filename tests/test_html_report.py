from __future__ import annotations

from datetime import datetime, timezone

from pharma_os.due_diligence_report import build_due_diligence_report_payload
from pharma_os.html_report import build_run_html, write_run_html
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
    assert "chart-spec" in html
