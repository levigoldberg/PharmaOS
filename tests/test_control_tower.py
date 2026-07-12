from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pharma_os.orchestrator as orchestrator_module
import pharma_os.control_tower as control_tower_module
from pharma_os.cli import main
from pharma_os.agent_runtime import AgentRuntimeConfig, StructuredAgentResult
from pharma_os.request_understanding import _request_understanding_error_message
from pharma_os.control_tower import build_deterministic_execution_plan, run_control_tower_agent, validate_execution_plan
from pharma_os.html_report import build_run_html
from pharma_os.request_understanding import RequestUnderstandingError
from pharma_os.memory import MemoryStore
from pharma_os.orchestrator import Orchestrator
from pharma_os.registry import WorkflowRegistry
from pharma_os.schemas import (
    AgentRunTrace,
    HumanGate,
    OrchestrationRequest,
    RequestUnderstandingAssumption,
    RequestUnderstandingOutput,
    SourceMetadata,
    WorkflowRun,
)


def _json_payload_from_stdout(stdout: str, *, base_dir: object) -> dict[str, object]:
    for line in stdout.splitlines():
        if line.startswith("json: "):
            path = line.removeprefix("json: ").strip()
            from pathlib import Path

            json_path = Path(path)
            if not json_path.is_absolute():
                json_path = Path(base_dir) / json_path
            return json.loads(json_path.read_text(encoding="utf-8"))
    raise AssertionError(f"stdout did not include a json path: {stdout}")


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


def test_scientific_state_includes_pending_decision_and_requirements() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(
        objective="Phase II to Phase III decision for this trial",
        nct_id="NCT12345678",
    )

    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    assert snapshot.pending_decision is not None
    assert snapshot.pending_decision.decision_type == "phase_transition"
    assert snapshot.pending_decision.target_capability_name == "protocol_design"
    assert {requirement.requirement_id for requirement in snapshot.evidence_requirements} >= {
        "agent3-agent4-handoffs",
        "phase-transition-analog-evidence",
    }
    assert snapshot.critical_evidence_gaps


def test_phase_transition_reuses_handoffs_and_identifies_agent5_gap() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    _save_completed_run(store, "due_diligence", "NCT12345678", "agent4-output")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(
        objective="Phase II to Phase III decision for this trial",
        nct_id="NCT12345678",
    )
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    statuses = {item.requirement_id: item.status for item in snapshot.requirement_satisfaction}
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert statuses["agent3-agent4-handoffs"] == "satisfied"
    assert statuses["phase-transition-analog-evidence"] == "missing"
    assert [step.action for step in plan.steps] == ["reuse", "reuse", "run"]
    assert plan.steps[-1].capability_name == "protocol_design"


def test_live_control_tower_payload_contains_decision_state(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(
        objective="Phase II to Phase III decision for this trial",
        nct_id="NCT12345678",
    )
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    captured: dict[str, object] = {}

    def fake_run_structured_agent(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output=kwargs["offline_output"], trace=SimpleNamespace(execution_mode="deterministic_fallback"), trace_metadata={})

    monkeypatch.setattr(control_tower_module, "run_structured_agent", fake_run_structured_agent)

    run_control_tower_agent(
        run_id="run",
        request=request,
        snapshot=snapshot,
        registry=registry,
        config=AgentRuntimeConfig(disabled=True),
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["pending_decision"]["decision_type"] == "phase_transition"
    assert payload["critical_evidence_gaps"]
    assert {item["requirement_id"] for item in payload["evidence_requirements"]} >= {
        "agent3-agent4-handoffs",
        "phase-transition-analog-evidence",
    }


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


def test_plan_validator_rejects_missing_dependency() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Build clinical stage due diligence", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    due_step = next(step for step in plan.steps if step.capability_name == "due_diligence")
    bad_plan = plan.model_copy(update={"steps": (due_step,)})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "missing dependency" in result.message for result in results)


def test_plan_validator_does_not_treat_skipped_dependency_as_satisfied() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Build clinical stage due diligence", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_steps = (
        plan.steps[0].model_copy(update={"action": "skip", "executable": False}),
        plan.steps[1],
    )
    bad_plan = plan.model_copy(update={"steps": bad_steps})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "missing dependency" in result.message for result in results)


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


def test_plan_validator_accepts_referenced_older_compatible_reuse_artifact() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(
        store,
        "clinical_outcome_prediction",
        "NCT12345678",
        "older-agent3-output",
        run_id="older-agent3-run",
    )
    _save_completed_run(
        store,
        "clinical_outcome_prediction",
        "NCT12345678",
        "newer-agent3-output",
        validation_status="needs_human_review",
        gate_decision="needs_human_review",
        run_id="newer-agent3-run",
    )
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Build clinical stage due diligence", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    reuse_step = next(step for step in plan.steps if step.capability_name == "clinical_outcome_prediction")
    referenced_reuse_step = reuse_step.model_copy(
        update={
            "action": "reuse",
            "reuse_run_id": "older-agent3-run",
            "reuse_output_id": "older-agent3-output",
        }
    )
    due_step = next(step for step in plan.steps if step.capability_name == "due_diligence")
    due_step = due_step.model_copy(update={"depends_on": (referenced_reuse_step.step_id,)})
    referenced_plan = plan.model_copy(update={"steps": (referenced_reuse_step, due_step)})

    results = validate_execution_plan(run_id="run", plan=referenced_plan, snapshot=snapshot, registry=registry)

    reuse_result = next(result for result in results if result.target_id == referenced_reuse_step.step_id)
    assert reuse_result.status == "passed"


def test_plan_validator_rejects_reuse_with_wrong_output_reference_even_when_run_exists() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(
        store,
        "clinical_outcome_prediction",
        "NCT12345678",
        "agent3-output",
        run_id="agent3-run",
    )
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_step = plan.steps[0].model_copy(
        update={
            "reuse_run_id": "agent3-run",
            "reuse_output_id": "wrong-output",
        }
    )
    bad_plan = plan.model_copy(update={"steps": (bad_step,)})

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


def test_plan_validator_rejects_unrelated_requirement_execution() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    _save_completed_run(store, "due_diligence", "NCT12345678", "agent4-output")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    protocol_request = OrchestrationRequest(objective="Draft the next-study protocol design", nct_id="NCT12345678")
    protocol_snapshot = store.build_scientific_state_snapshot(protocol_request, registry=registry)
    protocol_plan = build_deterministic_execution_plan(run_id="run", request=protocol_request, snapshot=protocol_snapshot, registry=registry)
    protocol_step = next(step for step in protocol_plan.steps if step.capability_name == "protocol_design")
    bad_step = protocol_step.model_copy(
        update={
            "action": "run",
            "executable": True,
            "requirements_addressed": (),
            "reuse_run_id": None,
            "reuse_output_id": None,
        }
    )
    bad_plan = protocol_plan.model_copy(update={"request": request, "steps": (bad_step,)})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "pending decision requirements" in result.message for result in results)


def test_plan_validator_rejects_reuse_that_does_not_satisfy_requirement() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_step = plan.steps[0].model_copy(update={"reuse_output_id": "wrong-output"})
    bad_plan = plan.model_copy(update={"steps": (bad_step,)})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "reuse does not satisfy cited decision evidence requirements" in result.message for result in results)


def test_plan_validator_rejects_duplicate_capability_steps() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    duplicate_step = plan.steps[0].model_copy(
        update={
            "step_id": "duplicate-step",
            "action": "skip",
            "executable": False,
            "reason": "Conflicting duplicate step.",
        }
    )
    bad_plan = plan.model_copy(update={"steps": (*plan.steps, duplicate_step)})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "duplicate capability step" in result.message for result in results)


def test_plan_validator_rejects_capability_outside_requested_target_path() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(
        objective="Assess clinical risk",
        nct_id="NCT12345678",
        identifiers={"target_capability": "clinical_outcome_prediction"},
    )
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    due_request = OrchestrationRequest(objective="Build clinical stage due diligence", nct_id="NCT12345678")
    due_snapshot = store.build_scientific_state_snapshot(due_request, registry=registry)
    due_plan = build_deterministic_execution_plan(run_id="run", request=due_request, snapshot=due_snapshot, registry=registry)
    due_step = next(step for step in due_plan.steps if step.capability_name == "due_diligence")
    bad_plan = due_plan.model_copy(update={"request": request, "steps": (due_step,)})

    results = validate_execution_plan(run_id="run", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(result.status == "failed" and "outside requested target path" in result.message for result in results)


def test_skeleton_requirements_are_present_and_blocked() -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(objective="Create a manufacturing CMC control plan for this asset")
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)

    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)

    assert snapshot.pending_decision is not None
    assert snapshot.pending_decision.decision_type == "manufacturing_control"
    assert snapshot.pending_decision.target_capability_name == "manufacturing_biofactory"
    assert "manufacturing-control-evidence" in {requirement.requirement_id for requirement in snapshot.evidence_requirements}
    assert "manufacturing_biofactory" in snapshot.blocked_capabilities
    assert plan.steps[-1].capability_name == "manufacturing_biofactory"
    assert plan.steps[-1].action == "block"


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


def test_orchestrate_reuses_existing_artifact_without_rerun(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    )

    assert [result.status for result in record.step_results] == ["reused"]
    assert record.step_results[0].reused_output_id == "agent3-output"
    assert record.step_results[0].execution_mode == "reused_artifact"
    assert record.execution_mode_summary.reused_artifacts_used == 1
    assert calls == {}


def test_goal_only_mapped_capability_lets_control_tower_reuse_existing_artifact(capsys, tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "memory.sqlite"
    store = MemoryStore(db_path)
    _save_completed_run(store, "clinical_outcome_prediction", "NCT04903795", "agent3-output")
    calls = _install_fake_runners(monkeypatch)

    monkeypatch.setattr(
        "pharma_os.cli.understand_orchestration_goal",
        lambda **_: RequestUnderstandingOutput(
            normalized_objective="Do the clinical trial prediction for NCT04903795",
            target_capability="clinical_outcome_prediction",
            decision_type="clinical_risk_assessment",
            nct_id="NCT04903795",
            asset_name=None,
            indication=None,
            assumptions=(),
            force_refresh=(),
            skip_capabilities=(),
            requested_outputs=("clinical_outcome_prediction_output",),
            missing_required_fields=(),
            clarifying_questions=(),
            confidence=0.95,
            rationale_summary="Clinical trial prediction request with an NCT ID.",
        ),
    )

    exit_code = main(
        [
            "orchestrate",
            "--goal",
            "Do the clinical trial prediction for NCT04903795",
            "--db-path",
            str(db_path),
            "--output-json",
            str(tmp_path / "out.json"),
            "--output-html",
            str(tmp_path / "out.html"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Orchestration completed" in stdout
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert "execution_intent" not in payload["request"]["identifiers"]
    assert payload["request"]["force_refresh"] == []
    assert payload["step_results"][0]["capability_name"] == "clinical_outcome_prediction"
    assert payload["step_results"][0]["status"] == "reused"
    assert calls == {}


def test_goal_only_explicit_reuse_intent_keeps_existing_artifact(capsys, tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "memory.sqlite"
    store = MemoryStore(db_path)
    _save_completed_run(store, "clinical_outcome_prediction", "NCT04903795", "agent3-output")
    calls = _install_fake_runners(monkeypatch)

    monkeypatch.setattr(
        "pharma_os.cli.understand_orchestration_goal",
        lambda **_: RequestUnderstandingOutput(
            normalized_objective="Reuse existing clinical trial prediction for NCT04903795",
            target_capability="clinical_outcome_prediction",
            decision_type="clinical_risk_assessment",
            nct_id="NCT04903795",
            asset_name=None,
            indication=None,
            assumptions=(),
            force_refresh=(),
            skip_capabilities=(),
            requested_outputs=("clinical_outcome_prediction_output",),
            missing_required_fields=(),
            clarifying_questions=(),
            confidence=0.95,
            rationale_summary="Explicit reuse request with an NCT ID.",
        ),
    )

    exit_code = main(
        [
            "orchestrate",
            "--goal",
            "Reuse existing clinical trial prediction for NCT04903795",
            "--db-path",
            str(db_path),
            "--output-json",
            str(tmp_path / "out.json"),
            "--output-html",
            str(tmp_path / "out.html"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Orchestration completed" in stdout
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert "execution_intent" not in payload["request"]["identifiers"]
    assert payload["request"]["force_refresh"] == []
    assert payload["step_results"][0]["status"] == "reused"
    assert payload["step_results"][0]["reused_output_id"] == "agent3-output"
    assert calls == {}


def test_orchestrate_replans_after_material_state_change(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(objective="Build clinical stage due diligence", nct_id="NCT12345678")
    )

    assert [result.capability_name for result in record.step_results] == ["clinical_outcome_prediction", "due_diligence"]
    assert [result.status for result in record.step_results] == ["executed", "executed"]
    assert len(record.replans) >= 1
    assert calls == {"clinical_outcome_prediction": 1, "due_diligence": 1}


def test_orchestrate_refreshes_downstream_after_upstream_change(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "old-agent3")
    _save_completed_run(store, "due_diligence", "NCT12345678", "old-agent4")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(
            objective="Build clinical stage due diligence",
            nct_id="NCT12345678",
            force_refresh=("clinical_outcome_prediction",),
        )
    )

    assert [result.status for result in record.step_results] == ["refreshed", "refreshed"]
    assert calls == {"clinical_outcome_prediction": 1, "due_diligence": 1}
    assert len(record.replans) >= 1


def test_target_only_refresh_reuses_dependency_and_supersedes_target(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output", run_id="old-agent3-run")
    _save_completed_run(store, "due_diligence", "NCT12345678", "old-agent4", run_id="old-agent4-run")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(
            objective="Remake only commercial due diligence for NCT12345678",
            nct_id="NCT12345678",
            identifiers={"target_capability": "due_diligence", "execution_scope": "target_only", "execution_intent": "run_fresh"},
            force_refresh=("due_diligence",),
        )
    )

    assert [result.capability_name for result in record.step_results] == ["clinical_outcome_prediction", "due_diligence"]
    assert [result.status for result in record.step_results] == ["reused", "refreshed"]
    assert calls == {"due_diligence": 1}
    old_bundle = store.get_run_bundle("old-agent4-run")
    new_bundle = store.get_run_bundle("due_diligence-child-1")
    assert old_bundle.run is not None
    assert new_bundle.run is not None
    assert old_bundle.run.metadata["artifact_lineage_status"] == "superseded"
    assert old_bundle.run.metadata["superseded_by_run_id"] == "due_diligence-child-1"
    assert new_bundle.run.metadata["artifact_lineage_status"] == "current"
    assert new_bundle.run.metadata["supersedes_run_ids"] == ["old-agent4-run"]
    latest = store.get_latest_workflow_output(workflow_name="due_diligence", nct_id="NCT12345678")
    assert latest is not None
    assert latest[0].run_id == "due_diligence-child-1"


def test_target_only_refresh_blocks_missing_dependency(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "due_diligence", "NCT12345678", "old-agent4", run_id="old-agent4-run")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(
            objective="Remake only commercial due diligence for NCT12345678",
            nct_id="NCT12345678",
            identifiers={"target_capability": "due_diligence", "execution_scope": "target_only", "execution_intent": "run_fresh"},
            force_refresh=("due_diligence",),
        )
    )

    assert record.step_results[0].capability_name == "clinical_outcome_prediction"
    assert record.step_results[0].status == "blocked"
    assert "target-only scope" in record.step_results[0].rationale
    assert calls == {}


def test_plan_validator_rejects_non_target_execution_for_target_only_scope() -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output", run_id="old-agent3-run")
    _save_completed_run(store, "due_diligence", "NCT12345678", "old-agent4", run_id="old-agent4-run")
    registry = WorkflowRegistry.default()
    request = OrchestrationRequest(
        objective="Remake only commercial due diligence for NCT12345678",
        nct_id="NCT12345678",
        identifiers={"target_capability": "due_diligence", "execution_scope": "target_only"},
        force_refresh=("due_diligence",),
    )
    snapshot = store.build_scientific_state_snapshot(request, registry=registry)
    plan = build_deterministic_execution_plan(run_id="run", request=request, snapshot=snapshot, registry=registry)
    bad_steps = (
        plan.steps[0].model_copy(update={"action": "refresh", "executable": True, "reason": "Badly refresh dependency."}),
        plan.steps[1],
    )
    bad_plan = plan.model_copy(update={"steps": bad_steps})

    results = validate_execution_plan(run_id="run-validation", plan=bad_plan, snapshot=snapshot, registry=registry)

    assert any(
        result.status == "failed" and "target-only request cannot execute non-target dependency" in result.message
        for result in results
    )


def test_orchestrate_supports_skip(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(objective="Skip clinical risk for this trial", nct_id="NCT12345678")
    )

    assert [result.status for result in record.step_results] == ["skipped"]
    assert calls == {}


def test_orchestrate_blocks_skeleton_capability_with_missing_connectors(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    _save_completed_run(store, "due_diligence", "NCT12345678", "agent4-output")
    _save_completed_run(store, "protocol_design", "NCT12345678", "agent5-output")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(objective="Create an enrollment feasibility plan", nct_id="NCT12345678")
    )

    assert record.step_results[-1].capability_name == "enrollment_feasibility"
    assert record.step_results[-1].status == "blocked"
    assert "enrollment_feasibility" in record.report.unavailable_modules
    assert calls == {}


def test_orchestrate_parent_child_audit_provenance(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    )
    bundle = store.get_run_bundle(record.run_id)

    assert bundle.run is not None
    assert bundle.run.workflow_name == "control_tower_orchestration"
    assert record.child_run_ids
    assert record.step_results[0].child_run_id == record.child_run_ids[0]
    assert record.snapshots
    assert record.plans
    assert record.report is not None
    assert bundle.output_json["step_results"][0]["child_run_id"] == record.child_run_ids[0]


def test_orchestrate_records_failed_step_when_ai_plan_retry_still_invalid(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    registry = WorkflowRegistry.default()
    calls = {"count": 0}

    def invalid_control_tower_agent(*, run_id, request, snapshot, registry=None, config=None, plan_feedback=()):
        calls["count"] += 1
        base_plan = build_deterministic_execution_plan(
            run_id=f"{run_id}-{calls['count']}",
            request=request,
            snapshot=snapshot,
            registry=registry or WorkflowRegistry.default(),
        )
        bad_step = base_plan.steps[0].model_copy(
            update={
                "action": "reuse",
                "executable": False,
                "reuse_run_id": "missing-run",
                "reuse_output_id": "missing-output",
                "reason": "Invalid AI reuse plan.",
            }
        )
        bad_plan = base_plan.model_copy(
            update={
                "output_id": f"bad-plan-{calls['count']}",
                "steps": (bad_step,),
                "provenance": "test.invalid_ai_plan",
            }
        )
        return StructuredAgentResult(
            output=bad_plan,
            trace=AgentRunTrace(
                trace_id=f"trace-{calls['count']}",
                run_id=run_id,
                agent_name="ControlTowerAgent",
                output_id=bad_plan.output_id,
                output_type="ExecutionPlan",
                provenance="test",
                execution_mode="deterministic_fallback",
            ),
            trace_metadata={
                "agent_name": "ControlTowerAgent",
                "fallback": True,
                "execution_mode": "deterministic_fallback",
                "error_type": "RuntimeError",
                "error": "fixture live planner failure",
            },
        )

    monkeypatch.setattr(orchestrator_module, "run_control_tower_agent", invalid_control_tower_agent)

    record = Orchestrator(memory=store, registry=registry).orchestrate(
        OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    )

    assert calls["count"] == 2
    assert record.step_results
    assert record.step_results[0].capability_name == "control_tower"
    assert record.step_results[0].status == "failed"
    assert "plan validation failed" in record.step_results[0].rationale
    assert record.report is not None
    assert "ControlTowerAgent used deterministic fallback" in record.report.fallback_summaries[0]
    assert "fixture live planner failure" in record.report.fallback_summaries[0]


def test_orchestrate_no_unnecessary_agent_3_4_5_reruns(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    _save_completed_run(store, "due_diligence", "NCT12345678", "agent4-output")
    _save_completed_run(store, "protocol_design", "NCT12345678", "agent5-output")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(objective="Draft the next-study protocol design", nct_id="NCT12345678")
    )

    assert [result.status for result in record.step_results] == ["reused", "reused", "reused"]
    assert calls == {}


def test_orchestrate_cli_writes_json_and_html(capsys, tmp_path, monkeypatch) -> None:
    output_json = tmp_path / "control_tower.json"
    output_html = tmp_path / "control_tower.html"
    monkeypatch.setattr(
        "pharma_os.cli.understand_orchestration_goal",
        lambda **_: RequestUnderstandingOutput(
            normalized_objective="Skip clinical risk for this trial",
            target_capability="clinical_outcome_prediction",
            decision_type="clinical_risk_assessment",
            nct_id=None,
            asset_name=None,
            indication=None,
            assumptions=(),
            force_refresh=(),
            skip_capabilities=("clinical_outcome_prediction",),
            requested_outputs=(),
            missing_required_fields=(),
            clarifying_questions=(),
            confidence=0.92,
            rationale_summary="The user asked to skip the clinical-risk workflow for the explicit NCT ID.",
        ),
    )

    exit_code = main(
        [
            "orchestrate",
            "--goal",
            "Skip clinical risk for this trial",
            "--nct-id",
            "NCT12345678",
            "--db-path",
            str(tmp_path / "memory.sqlite"),
            "--output-json",
            str(output_json),
            "--output-html",
            str(output_html),
        ]
    )

    assert exit_code == 0
    assert output_json.exists()
    assert output_html.exists()
    assert "control_tower_orchestration" in output_html.read_text(encoding="utf-8")
    assert "Control Tower Orchestration" in output_html.read_text(encoding="utf-8")
    assert "step_results" in output_json.read_text(encoding="utf-8")
    stdout = capsys.readouterr().out
    assert "Orchestration completed" in stdout
    assert "json:" in stdout


def test_control_tower_html_starts_with_executive_audit(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(
            objective="Assess clinical risk",
            nct_id="NCT12345678",
            identifiers={"target_capability": "clinical_outcome_prediction", "execution_intent": "run_fresh"},
            force_refresh=("clinical_outcome_prediction",),
        )
    )

    html = build_run_html(record.run_id, memory=store)

    assert "What Happened" in html
    assert "inferred_capability" in html
    assert "clinical_outcome_prediction" in html
    assert "workflow_executed" in html
    assert "child_run_ids" in html
    assert "Human Attention Needed" in html


def test_orchestrate_goal_only_uses_ai_understanding_and_default_outputs(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    calls = _install_fake_runners(monkeypatch)

    def fake_understand(**kwargs):
        assert kwargs["goal"] == "Draft the next-study protocol design for NCT04903795"
        return RequestUnderstandingOutput(
            normalized_objective="Draft the next-study protocol design for NCT04903795",
            target_capability="protocol_design",
            decision_type="protocol_design",
            nct_id="NCT04903795",
            asset_name=None,
            indication=None,
            assumptions=(),
            force_refresh=(),
            skip_capabilities=(),
            requested_outputs=("json", "html"),
            missing_required_fields=(),
            clarifying_questions=(),
            confidence=0.86,
            rationale_summary="Protocol design request anchored to an NCT ID.",
        )

    monkeypatch.setattr("pharma_os.cli.understand_orchestration_goal", fake_understand)

    exit_code = main(
        [
            "orchestrate",
            "--goal",
            "Draft the next-study protocol design for NCT04903795",
            "--db-path",
            str(tmp_path / "memory.sqlite"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Orchestration completed" in stdout
    payload = _json_payload_from_stdout(stdout, base_dir=tmp_path)
    assert payload["request"]["nct_id"] == "NCT04903795"
    assert [result["capability_name"] for result in payload["step_results"]] == [
        "clinical_outcome_prediction",
        "due_diligence",
        "protocol_design",
    ]
    assert calls == {
        "clinical_outcome_prediction": 1,
        "due_diligence": 1,
        "protocol_design": 1,
    }
    exported = payload["exported_files"]
    parent_json = tmp_path / exported["parent_json"]
    parent_html = tmp_path / exported["parent_html"]
    cumulative_report = Path(exported["cumulative_nct_report"])
    family_dir = parent_json.parent
    assert parent_json.exists()
    assert parent_html.exists()
    assert cumulative_report.exists()
    assert cumulative_report == tmp_path / "reports" / "NCT04903795.html"
    assert "Agent 5 - Protocol Design" in cumulative_report.read_text(encoding="utf-8")
    assert parent_html.parent == family_dir
    assert len(exported["child_runs"]) == 3
    for child in exported["child_runs"]:
        child_json = tmp_path / child["json"]
        child_html = tmp_path / child["html"]
        assert child_json.exists()
        assert child_html.exists()
        assert child_json.parent == family_dir
        assert child_html.parent == family_dir
    assert list(family_dir.glob("control_tower_orchestration_*.json"))
    assert list(family_dir.glob("clinical_outcome_prediction_*.json"))
    assert list(family_dir.glob("due_diligence_*.json"))
    assert list(family_dir.glob("protocol_design_*.json"))
    assert "cumulative_nct_report:" in stdout


def test_orchestrate_goal_only_ai_unavailable_returns_clear_error(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "pharma_os.cli.understand_orchestration_goal",
        lambda **_: (_ for _ in ()).throw(RequestUnderstandingError("Natural-language goal parsing requires live AI.")),
    )

    exit_code = main(
        [
            "orchestrate",
            "--goal",
            "Draft the next-study protocol design for NCT04903795",
            "--db-path",
            str(tmp_path / "memory.sqlite"),
        ]
    )

    assert exit_code == 2
    assert "Natural-language goal parsing requires live AI" in capsys.readouterr().out


def test_request_understanding_error_reports_live_api_failure(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = AgentRuntimeConfig(model="gpt-test", model_route="request_understanding", disabled=False)

    message = _request_understanding_error_message(RuntimeError("model not found"), config)

    assert "attempted a live OpenAI call but it failed" in message
    assert "Route=request_understanding" in message
    assert "model=gpt-test" in message
    assert "model not found" in message


def test_request_understanding_error_reports_key_not_visible(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = AgentRuntimeConfig(
        model="gpt-test",
        model_route="request_understanding",
        disabled=True,
        provenance="pharma_os.request_understanding.missing_openai_api_key",
    )

    message = _request_understanding_error_message(RuntimeError("offline"), config)

    assert "OPENAI_API_KEY is not visible" in message


def test_orchestrate_goal_only_blocks_registered_unimplemented_capability(capsys, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "pharma_os.cli.understand_orchestration_goal",
        lambda **_: RequestUnderstandingOutput(
            normalized_objective="Create a manufacturing CMC control plan for this asset",
            target_capability="manufacturing_biofactory",
            decision_type="manufacturing_control",
            nct_id=None,
            asset_name=None,
            indication=None,
            assumptions=(),
            force_refresh=(),
            skip_capabilities=(),
            requested_outputs=(),
            missing_required_fields=(),
            clarifying_questions=(),
            confidence=0.82,
            rationale_summary="Manufacturing request maps to a registered skeleton capability.",
        ),
    )

    exit_code = main(
        [
            "orchestrate",
            "--goal",
            "Create a manufacturing CMC control plan for this asset",
            "--db-path",
            str(tmp_path / "memory.sqlite"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "Orchestration completed" in stdout
    payload = _json_payload_from_stdout(stdout, base_dir=tmp_path)
    assert payload["step_results"][0]["capability_name"] == "manufacturing_biofactory"
    assert payload["step_results"][0]["status"] == "blocked"


def test_ai_extracted_nct_conflict_is_rejected() -> None:
    from pharma_os.cli import _request_from_understanding

    parsed = RequestUnderstandingOutput(
        normalized_objective="Build diligence for NCT11111111",
        target_capability="due_diligence",
        decision_type="clinical_stage_due_diligence",
        nct_id="NCT11111111",
        asset_name=None,
        indication=None,
        assumptions=(),
        force_refresh=(),
        skip_capabilities=(),
        requested_outputs=(),
        missing_required_fields=(),
        clarifying_questions=(),
        confidence=0.9,
        rationale_summary="Diligence request anchored to an NCT ID.",
    )

    try:
        _request_from_understanding(
            goal="Build diligence for NCT11111111",
            parsed=parsed,
            explicit_nct="NCT22222222",
            explicit_asset_name=None,
            explicit_indication=None,
            explicit_assumptions={},
            explicit_force_refresh=(),
            registry=WorkflowRegistry.default(),
        )
    except ValueError as exc:
        assert "conflicts with AI-extracted NCT ID" in str(exc)
    else:
        raise AssertionError("expected conflicting NCT IDs to fail")


def test_ai_assumptions_are_limited_to_workflow_inputs() -> None:
    from pharma_os.cli import _request_from_understanding

    parsed = RequestUnderstandingOutput(
        normalized_objective="Build diligence for NCT11111111",
        target_capability="due_diligence",
        decision_type="clinical_stage_due_diligence",
        nct_id="NCT11111111",
        asset_name=None,
        indication=None,
        assumptions=(
            RequestUnderstandingAssumption(key="annual_patients", value="1200"),
            RequestUnderstandingAssumption(key="workflow_intent", value="The user wants diligence."),
        ),
        force_refresh=(),
        skip_capabilities=(),
        requested_outputs=(),
        missing_required_fields=(),
        clarifying_questions=(),
        confidence=0.9,
        rationale_summary="Diligence request anchored to an NCT ID.",
    )

    request = _request_from_understanding(
        goal="Build diligence for NCT11111111",
        parsed=parsed,
        explicit_nct=None,
        explicit_asset_name=None,
        explicit_indication=None,
        explicit_assumptions={},
        explicit_force_refresh=(),
        registry=WorkflowRegistry.default(),
    )

    assert request.assumptions == {"annual_patients": 1200}


def test_ai_scoped_rerun_maps_to_force_refresh_and_target_only_scope() -> None:
    from pharma_os.cli import _request_from_understanding

    parsed = RequestUnderstandingOutput(
        normalized_objective="Remake only commercial due diligence for NCT07011706",
        target_capability="due_diligence",
        decision_type="clinical_stage_due_diligence",
        nct_id="NCT07011706",
        asset_name=None,
        indication=None,
        assumptions=(),
        force_refresh=("due_diligence",),
        skip_capabilities=(),
        requested_outputs=(),
        execution_scope="target_only",
        missing_required_fields=(),
        clarifying_questions=(),
        confidence=0.94,
        rationale_summary="User requested a fresh due-diligence rerun scoped only to the target workflow.",
    )

    request = _request_from_understanding(
        goal="Remake only commercial due diligence for NCT07011706",
        parsed=parsed,
        explicit_nct=None,
        explicit_asset_name=None,
        explicit_indication=None,
        explicit_assumptions={},
        explicit_force_refresh=(),
        registry=WorkflowRegistry.default(),
    )

    assert request.nct_id == "NCT07011706"
    assert request.identifiers["target_capability"] == "due_diligence"
    assert request.identifiers["execution_scope"] == "target_only"
    assert request.identifiers["execution_intent"] == "run_fresh"
    assert request.force_refresh == ("due_diligence",)


def test_optional_commercial_assumption_gap_does_not_block_due_diligence_goal() -> None:
    from pharma_os.cli import _request_from_understanding

    parsed = RequestUnderstandingOutput(
        normalized_objective="Do the commercial due diligence for NCT05966480",
        target_capability="due_diligence",
        decision_type="clinical_stage_due_diligence",
        nct_id="NCT05966480",
        asset_name=None,
        indication=None,
        assumptions=(),
        force_refresh=(),
        skip_capabilities=(),
        requested_outputs=(),
        missing_required_fields=("reviewed_commercial_assumptions",),
        clarifying_questions=(
            "What reviewed commercial assumptions should be used for the due diligence?",
        ),
        confidence=0.84,
        rationale_summary="Diligence request anchored to an NCT ID.",
    )

    request = _request_from_understanding(
        goal="Do the commercial due diligence for NCT05966480",
        parsed=parsed,
        explicit_nct=None,
        explicit_asset_name=None,
        explicit_indication=None,
        explicit_assumptions={},
        explicit_force_refresh=(),
        registry=WorkflowRegistry.default(),
    )

    assert request.nct_id == "NCT05966480"
    assert request.identifiers["target_capability"] == "due_diligence"
    assert request.identifiers["optional_assumption_gaps"] == "reviewed_commercial_assumptions"
    assert "execution_intent" not in request.identifiers
    assert request.force_refresh == ()


def test_optional_commercial_and_identity_question_does_not_block_when_nct_present() -> None:
    from pharma_os.cli import _request_from_understanding

    parsed = RequestUnderstandingOutput(
        normalized_objective="Do the commercial due diligence for NCT05966480",
        target_capability="due_diligence",
        decision_type="clinical_stage_due_diligence",
        nct_id="NCT05966480",
        asset_name=None,
        indication=None,
        assumptions=(),
        force_refresh=(),
        skip_capabilities=(),
        requested_outputs=(),
        missing_required_fields=("reviewed_commercial_assumptions",),
        clarifying_questions=(
            "Do you want to provide any reviewed_commercial_assumptions for this diligence run "
            "(e.g., discount_rate, development_cost, launch_year, loe_year, annual_patients, "
            "peak_penetration, gross_to_net, operating_margin, wac_data_path, pos_workbook_path)? "
            "If not, should the workflow use its default commercial assumptions? Is it acceptable "
            "to proceed without an explicit asset_name/indication (NCT will be used as the primary identifier)?",
        ),
        confidence=0.84,
        rationale_summary="Diligence request anchored to an NCT ID.",
    )

    request = _request_from_understanding(
        goal="Do the commercial due diligence for NCT05966480",
        parsed=parsed,
        explicit_nct=None,
        explicit_asset_name=None,
        explicit_indication=None,
        explicit_assumptions={},
        explicit_force_refresh=(),
        registry=WorkflowRegistry.default(),
    )

    assert request.nct_id == "NCT05966480"
    assert request.identifiers["optional_assumption_gaps"] == "reviewed_commercial_assumptions"


def test_optional_default_commercial_question_without_missing_fields_does_not_block() -> None:
    from pharma_os.cli import _request_from_understanding

    parsed = RequestUnderstandingOutput(
        normalized_objective="Do the commercial due diligence for NCT05966480",
        target_capability="due_diligence",
        decision_type="clinical_stage_due_diligence",
        nct_id="NCT05966480",
        asset_name=None,
        indication=None,
        assumptions=(),
        force_refresh=(),
        skip_capabilities=(),
        requested_outputs=(),
        missing_required_fields=(),
        clarifying_questions=(
            "Do you want default commercial assumptions for the diligence "
            "(currency, WAC, peak penetration, launch year discount rate, etc.), "
            "or will you provide custom commercial assumptions?",
        ),
        confidence=0.84,
        rationale_summary="Diligence request anchored to an NCT ID.",
    )

    request = _request_from_understanding(
        goal="Do the commercial due diligence for NCT05966480",
        parsed=parsed,
        explicit_nct=None,
        explicit_asset_name=None,
        explicit_indication=None,
        explicit_assumptions={},
        explicit_force_refresh=(),
        registry=WorkflowRegistry.default(),
    )

    assert request.nct_id == "NCT05966480"
    assert request.identifiers["target_capability"] == "due_diligence"
    assert "execution_intent" not in request.identifiers


def test_output_format_scope_and_reuse_question_does_not_block_mapped_due_diligence_goal() -> None:
    from pharma_os.cli import _request_from_understanding

    parsed = RequestUnderstandingOutput(
        normalized_objective="Do the commercial due diligence for NCT05966480",
        target_capability="due_diligence",
        decision_type="clinical_stage_due_diligence",
        nct_id="NCT05966480",
        asset_name=None,
        indication=None,
        assumptions=(),
        force_refresh=(),
        skip_capabilities=(),
        requested_outputs=(),
        missing_required_fields=(),
        clarifying_questions=(
            "Do you want a full Agent-4 due diligence deliverable (asset memo, commercial model, rNPV) "
            "or a more limited commercial-only memo? Preferred output formats? (e.g., PDF memo, Excel "
            "commercial model, CSV, or slide deck) Should I reuse any existing Agent-3/Agent-4 artifacts "
            "already in the system for this NCT if present, or produce fresh analyses?",
        ),
        confidence=0.84,
        rationale_summary="Diligence request anchored to an NCT ID.",
    )

    request = _request_from_understanding(
        goal="Do the commercial due diligence for NCT05966480",
        parsed=parsed,
        explicit_nct=None,
        explicit_asset_name=None,
        explicit_indication=None,
        explicit_assumptions={},
        explicit_force_refresh=(),
        registry=WorkflowRegistry.default(),
    )

    assert request.nct_id == "NCT05966480"
    assert request.identifiers["target_capability"] == "due_diligence"
    assert "execution_intent" not in request.identifiers
    assert request.force_refresh == ()


def test_missing_nct_still_blocks_due_diligence_goal() -> None:
    from pharma_os.cli import _request_from_understanding

    parsed = RequestUnderstandingOutput(
        normalized_objective="Do the commercial due diligence",
        target_capability="due_diligence",
        decision_type="clinical_stage_due_diligence",
        nct_id=None,
        asset_name=None,
        indication=None,
        assumptions=(),
        force_refresh=(),
        skip_capabilities=(),
        requested_outputs=(),
        missing_required_fields=("reviewed_commercial_assumptions",),
        clarifying_questions=("What reviewed commercial assumptions should be used?",),
        confidence=0.84,
        rationale_summary="Diligence request missing NCT ID.",
    )

    try:
        _request_from_understanding(
            goal="Do the commercial due diligence",
            parsed=parsed,
            explicit_nct=None,
            explicit_asset_name=None,
            explicit_indication=None,
            explicit_assumptions={},
            explicit_force_refresh=(),
            registry=WorkflowRegistry.default(),
        )
    except ValueError as exc:
        assert "nct_id" in str(exc)
    else:
        raise AssertionError("expected missing NCT ID to block")


def _save_completed_run(
    store: MemoryStore,
    workflow_name: str,
    nct_id: str,
    output_id: str,
    *,
    run_id: str | None = None,
    validation_status: str = "passed",
    gate_decision: str | None = None,
    output_payload: dict | None = None,
) -> None:
    run_id = run_id or f"{workflow_name}-run"
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


def _install_fake_runners(monkeypatch) -> dict[str, int]:
    calls: dict[str, int] = {}

    def make_runner(workflow_name: str):
        def runner(input_data, *, memory):
            calls[workflow_name] = calls.get(workflow_name, 0) + 1
            run_id = f"{workflow_name}-child-{calls[workflow_name]}"
            output_id = f"{workflow_name}-output-{calls[workflow_name]}"
            memory.save_run(
                WorkflowRun(
                    run_id=run_id,
                    workflow_name=workflow_name,
                    status="completed",
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    input_provenance="fake",
                    validation_status="passed",
                    metadata={"nct_id": input_data.nct_id},
                ),
                input_payload=input_data,
                output_payload={"output_id": output_id, "input": {"nct_id": input_data.nct_id}, "confidence": 0.8},
            )
            memory.save_sources(
                run_id,
                (
                    SourceMetadata(
                        source_id=f"source:{run_id}",
                        title="Fixture source",
                        provenance="fake",
                        source_type="fixture",
                    ),
                ),
            )
            return SimpleNamespace(
                run_id=run_id,
                output_id=output_id,
                validation_status="passed",
                human_gate=None,
            )

        return runner

    monkeypatch.setitem(orchestrator_module._WORKFLOW_RUNNERS, "clinical_outcome_prediction", make_runner("clinical_outcome_prediction"))
    monkeypatch.setitem(orchestrator_module._WORKFLOW_RUNNERS, "due_diligence", make_runner("due_diligence"))
    monkeypatch.setitem(orchestrator_module._WORKFLOW_RUNNERS, "protocol_design", make_runner("protocol_design"))
    return calls
