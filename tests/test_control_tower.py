from __future__ import annotations

from datetime import datetime, timezone

from pharma_os.control_tower import build_deterministic_execution_plan, validate_execution_plan
from pharma_os.memory import MemoryStore
from pharma_os.orchestrator import Orchestrator
from pharma_os.registry import WorkflowRegistry
from pharma_os.schemas import HumanGate, OrchestrationRequest, SourceMetadata, WorkflowRun


def test_control_tower_plans_minimum_clinical_risk_path() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk for this trial", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert [step.capability_name for step in plan.steps] == ["clinical_outcome_prediction"]
    assert [step.action for step in plan.steps] == ["run"]
    assert not plan.blocked


def test_control_tower_plans_diligence_with_dependency_only() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Build clinical stage due diligence", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert [step.capability_name for step in plan.steps] == ["clinical_outcome_prediction", "due_diligence"]
    assert [step.action for step in plan.steps] == ["run", "run"]


def test_control_tower_reuses_handoffs_for_next_study_design() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    _save_completed_run(
        store,
        "due_diligence",
        "NCT12345678",
        "agent4-output",
        output_payload={
            "output_id": "agent4-output",
            "input": {"nct_id": "NCT12345678"},
            "agent3_handoff": {"agent3_run_id": "clinical-run", "agent3_output_id": "agent3-output"},
            "confidence": 0.7,
        },
    )
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Draft the next-study protocol design", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert [step.capability_name for step in plan.steps] == ["clinical_outcome_prediction", "due_diligence", "protocol_design"]
    assert [step.action for step in plan.steps] == ["reuse", "reuse", "run"]


def test_control_tower_blocks_unavailable_future_module() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Create an enrollment feasibility plan", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    enrollment_step = plan.steps[-1]

    assert enrollment_step.capability_name == "enrollment_feasibility"
    assert enrollment_step.action == "block"
    assert "site_performance_database" in enrollment_step.blocked_by
    assert plan.blocked is True


def test_control_tower_reuses_compatible_artifact() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert plan.steps[0].action == "reuse"
    assert plan.steps[0].reuse_output_id == "agent3-output"


def test_control_tower_refreshes_incompatible_artifact() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output", validation_status="failed")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert plan.steps[0].action == "refresh"
    assert any("validation failed" in reason for artifact in snapshot.artifacts for reason in artifact.reasons)


def test_control_tower_blocks_execution_through_blocking_gate() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output", gate_decision="blocked")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert plan.steps[0].action == "block"
    assert plan.blocked is True


def test_control_tower_refreshes_downstream_when_dependency_changes() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    _save_completed_run(
        store,
        "due_diligence",
        "NCT12345678",
        "agent4-output",
        output_payload={
            "output_id": "agent4-output",
            "input": {"nct_id": "NCT12345678"},
            "agent3_handoff": {"agent3_run_id": "clinical-run", "agent3_output_id": "agent3-output"},
            "confidence": 0.7,
        },
    )
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(
        objective="Build clinical stage due diligence",
        nct_id="NCT12345678",
        force_refresh=("clinical_outcome_prediction",),
    )
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert [step.action for step in plan.steps] == ["refresh", "refresh"]
    assert "dependency outputs will change" in plan.steps[1].reason


def test_plan_validator_rejects_invalid_plans() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_plan = plan.model_copy(update={"steps": (plan.steps[0].model_copy(update={"capability_name": "unknown"}),)})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "unknown capability" in result.message for result in results)


def test_plan_validator_rejects_dependency_ordering() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Build clinical stage due diligence", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_plan = plan.model_copy(update={"steps": tuple(reversed(plan.steps))})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "dependency ordering invalid" in result.message for result in results)


def test_plan_validator_rejects_execution_of_skeleton_module() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Create an enrollment feasibility plan", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_steps = tuple(
        step.model_copy(update={"action": "run", "executable": True})
        if step.capability_name == "enrollment_feasibility"
        else step
        for step in plan.steps
    )
    bad_plan = plan.model_copy(update={"steps": bad_steps})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "non-implemented capability" in result.message for result in results)


def test_plan_validator_rejects_reuse_of_missing_artifact() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_plan = plan.model_copy(update={"steps": (plan.steps[0].model_copy(update={"action": "reuse", "reuse_run_id": "missing"}),)})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "reuse references missing" in result.message for result in results)


def test_plan_validator_rejects_unjustified_refresh_and_unnecessary_rerun() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_step = plan.steps[0].model_copy(update={"action": "refresh", "executable": True, "reason": "Refresh without reason."})
    bad_plan = plan.model_copy(update={"steps": (bad_step,)})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "refresh is not justified" in result.message for result in results)
    assert any(result.status == "failed" and "unnecessary rerun" in result.message for result in results)


def test_plan_validator_rejects_execution_through_blocking_gate() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output", gate_decision="blocked")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Build clinical stage due diligence", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    due_step = next(step for step in plan.steps if step.capability_name == "due_diligence")
    bad_plan = plan.model_copy(update={"steps": (plan.steps[0], due_step.model_copy(update={"action": "run", "executable": True}))})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "blocking human gate" in result.message for result in results)


def test_orchestrator_persists_control_tower_plan_and_safe_trace() -> None:
    store = MemoryStore(":memory:")
    record = Orchestrator(memory=store).plan(
        OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    )

    bundle = store.get_run_bundle(record.run_id)

    assert bundle.run is not None
    assert bundle.run.workflow_name == "control_tower"
    assert bundle.output_json["output_id"] == record.plan.output_id
    assert bundle.agent_traces[0].agent_name == "ControlTowerAgent"
    assert bundle.validation_results


def _save_completed_run(
    store: MemoryStore,
    workflow_name: str,
    nct_id: str,
    output_id: str,
    *,
    validation_status: str = "passed",
    gate_decision: str | None = None,
    output_payload: dict | None = None,
) -> None:
    run_id = f"{workflow_name}-run"
    run = WorkflowRun(
        run_id=run_id,
        workflow_name=workflow_name,
        status="completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        input_provenance="test",
        validation_status=validation_status,
        metadata={"nct_id": nct_id},
    )
    payload = output_payload or {
        "output_id": output_id,
        "input": {"nct_id": nct_id},
        "confidence": 0.8,
    }
    store.save_run(run, input_payload={"nct_id": nct_id}, output_payload=payload)
    store.save_sources(
        run_id,
        (
            SourceMetadata(
                source_id=f"source:{run_id}",
                title="Fixture source",
                provenance="test",
                source_type="fixture",
                retrieved_at=datetime.now(timezone.utc),
            ),
        ),
    )
    if gate_decision:
        store.save_human_gate(
            run_id,
            HumanGate(
                gate_id=f"gate-{run_id}",
                decision=gate_decision,
                gate_reason="Fixture blocking gate.",
                required_roles=("clinical_lead",),
                provenance="test",
            ),
        )
