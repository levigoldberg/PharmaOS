from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pharma_os.orchestrator as orchestrator_module
from pharma_os.cli import main
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


def test_orchestrate_reuses_existing_artifact_without_rerun(monkeypatch) -> None:
    store = MemoryStore(":memory:")
    _save_completed_run(store, "clinical_outcome_prediction", "NCT12345678", "agent3-output")
    calls = _install_fake_runners(monkeypatch)

    record = Orchestrator(memory=store).orchestrate(
        OrchestrationRequest(objective="Assess clinical risk", nct_id="NCT12345678")
    )

    assert [result.status for result in record.step_results] == ["reused"]
    assert record.step_results[0].reused_output_id == "agent3-output"
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


def test_orchestrate_cli_writes_json_and_html(capsys, tmp_path) -> None:
    output_json = tmp_path / "control_tower.json"
    output_html = tmp_path / "control_tower.html"

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
    assert "step_results" in capsys.readouterr().out


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
