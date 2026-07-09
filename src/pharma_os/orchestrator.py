"""Workflow orchestration entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pharma_os.control_tower import run_control_tower_agent, validate_execution_plan
from pharma_os.execution_modes import primary_execution_mode, summarize_execution_modes
from pharma_os.memory import MemoryStore
from pharma_os.registry import WorkflowRegistry
from pharma_os.schemas import (
    AgentOutput,
    ControlTowerReport,
    ClinicalOutcomePredictionInput,
    ClinicalTrialIntelligenceInput,
    DueDiligenceInput,
    ExecutionPlan,
    OrchestrationRequest,
    OrchestrationReplanRecord,
    OrchestrationRunRecord,
    OrchestrationStepResult,
    PlannedStep,
    ProtocolDesignInput,
    ScientificStateSnapshot,
    ValidationResult,
    WorkflowRun,
)
from pharma_os.validators import validate_workflow_name
from pharma_os.workflows.clinical_outcome_prediction import run_clinical_outcome_prediction_workflow
from pharma_os.workflows.due_diligence import run_due_diligence_workflow
from pharma_os.workflows.protocol_design import run_protocol_design_workflow
from pharma_os.workflows.trial_intelligence import run_trial_intelligence_workflow


_WORKFLOW_RUNNERS = {
    "trial_intelligence": run_trial_intelligence_workflow,
    "clinical_outcome_prediction": run_clinical_outcome_prediction_workflow,
    "due_diligence": run_due_diligence_workflow,
    "protocol_design": run_protocol_design_workflow,
}

_INPUT_TYPES = {
    "trial_intelligence": ClinicalTrialIntelligenceInput,
    "clinical_outcome_prediction": ClinicalOutcomePredictionInput,
    "due_diligence": DueDiligenceInput,
    "protocol_design": ProtocolDesignInput,
}


class Orchestrator:
    """Coordinates direct workflow runs, Control Tower planning, and bounded orchestration."""

    def __init__(self, memory: MemoryStore | None = None, registry: WorkflowRegistry | None = None) -> None:
        self.memory = memory or MemoryStore()
        self.registry = registry or WorkflowRegistry.default()

    def run(
        self,
        workflow: str,
        input_data: ClinicalTrialIntelligenceInput | DueDiligenceInput | ClinicalOutcomePredictionInput | ProtocolDesignInput | None = None,
    ) -> object:
        """Run a workflow by name."""

        workflow_name = validate_workflow_name(workflow)
        if workflow_name == "trial_intelligence":
            return self._run_registered_workflow(workflow_name, input_data)
        capability = self.registry.get(workflow_name)
        if capability is None:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        if not capability.executable:
            raise ValueError(f"Workflow capability is not executable: {workflow_name}")
        return self._run_registered_workflow(workflow_name, input_data)

    def plan(self, request: OrchestrationRequest) -> OrchestrationRunRecord:
        """Build and persist a planning-only Control Tower execution plan."""

        run_id = f"control-tower-{uuid4()}"
        started_at = datetime.now(timezone.utc)
        snapshot = self.memory.build_scientific_state_snapshot(request, registry=self.registry)
        agent_result = run_control_tower_agent(
            run_id=run_id,
            request=request,
            snapshot=snapshot,
            registry=self.registry,
        )
        validation_results = validate_execution_plan(
            run_id=run_id,
            plan=agent_result.output,
            snapshot=snapshot,
            registry=self.registry,
        )
        failed = any(result.status == "failed" for result in validation_results)
        validation_status = "failed" if failed else "needs_human_review" if agent_result.output.blocked else "passed"
        plan = agent_result.output.model_copy(update={"validation_status": validation_status})
        completed_at = datetime.now(timezone.utc)
        run = WorkflowRun(
            run_id=run_id,
            workflow_name="control_tower",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            input_provenance="control_tower.plan",
            validation_status=validation_status,
            gate_reason="; ".join(plan.block_reasons) if plan.block_reasons else None,
            metadata={"objective": request.objective[:500], "nct_id": request.nct_id},
        )
        self.memory.save_run(
            run,
            input_payload=request,
            output_payload=plan,
            trace_metadata=agent_result.trace_metadata,
        )
        self.memory.save_agent_trace(agent_result.trace)
        self.memory.save_agent_output(
            AgentOutput(
                output_id=plan.output_id,
                agent_name="ControlTowerAgent",
                run_id=run_id,
                provenance=agent_result.trace.provenance,
                confidence=plan.confidence,
                validation_status=validation_status,
                gate_reason=run.gate_reason,
                execution_mode=agent_result.trace.execution_mode,
                execution_mode_summary=summarize_execution_modes((agent_result.trace,)),
            ),
            payload=plan,
        )
        self.memory.save_validation_results(run_id, validation_results)
        execution_mode_summary = summarize_execution_modes((agent_result.trace,))
        return OrchestrationRunRecord(
            run_id=run_id,
            request=request,
            snapshot=snapshot,
            plan=plan,
            snapshots=(snapshot,),
            final_snapshot=snapshot,
            plans=(plan,),
            validation_results=validation_results,
            trace=agent_result.trace,
            execution_mode_summary=execution_mode_summary,
        )

    def orchestrate(
        self,
        request: OrchestrationRequest,
        *,
        max_steps: int = 12,
        max_replans: int = 4,
    ) -> OrchestrationRunRecord:
        """Plan and execute a bounded memory-aware orchestration loop."""

        parent_run_id = f"control-tower-orchestration-{uuid4()}"
        started_at = datetime.now(timezone.utc)
        initial_snapshot = self.memory.build_scientific_state_snapshot(request, registry=self.registry)
        snapshots: list[ScientificStateSnapshot] = [initial_snapshot]
        plans: list[ExecutionPlan] = []
        validation_results: list[ValidationResult] = []
        step_results: list[OrchestrationStepResult] = []
        replans: list[OrchestrationReplanRecord] = []
        child_run_ids: list[str] = []
        control_tower_traces = []
        completed_capabilities: set[str] = set()
        current_snapshot = initial_snapshot
        current_plan: ExecutionPlan | None = None
        step_count = 0
        replan_count = 0
        plan_validation_failed = False

        while step_count < max_steps:
            if current_plan is None:
                agent_result = run_control_tower_agent(
                    run_id=parent_run_id,
                    request=request,
                    snapshot=current_snapshot,
                    registry=self.registry,
                )
                self.memory.save_agent_trace(agent_result.trace)
                control_tower_traces.append(agent_result.trace)
                plan_results = validate_execution_plan(
                    run_id=f"{parent_run_id}-plan-{len(plans) + 1}",
                    plan=agent_result.output,
                    snapshot=current_snapshot,
                    registry=self.registry,
                )
                validation_results.extend(plan_results)
                failed = any(result.status == "failed" for result in plan_results)
                plan_status = "failed" if failed else "needs_human_review" if agent_result.output.blocked else "passed"
                current_plan = agent_result.output.model_copy(update={"validation_status": plan_status})
                plans.append(current_plan)
                self.memory.save_agent_output(
                    AgentOutput(
                        output_id=current_plan.output_id,
                        agent_name="ControlTowerAgent",
                        run_id=parent_run_id,
                        provenance=agent_result.trace.provenance,
                        confidence=current_plan.confidence,
                        validation_status=plan_status,
                        gate_reason="; ".join(current_plan.block_reasons) if current_plan.block_reasons else None,
                        execution_mode=agent_result.trace.execution_mode,
                        execution_mode_summary=summarize_execution_modes((agent_result.trace,)),
                    ),
                    payload=current_plan,
                )
                if failed:
                    plan_validation_failed = True
                    break

            step = _next_unhandled_step(current_plan, completed_capabilities)
            if step is None:
                break
            step_count += 1
            before_snapshot = current_snapshot
            result, after_snapshot = self._execute_orchestration_step(
                parent_run_id=parent_run_id,
                request=request,
                plan=current_plan,
                step=step,
                before_snapshot=before_snapshot,
            )
            step_results.append(result)
            completed_capabilities.add(step.capability_name)
            if result.child_run_id:
                child_run_ids.append(result.child_run_id)
            if result.status in {"blocked", "failed"}:
                current_snapshot = after_snapshot or before_snapshot
                if after_snapshot is not None and after_snapshot.snapshot_id != before_snapshot.snapshot_id:
                    snapshots.append(after_snapshot)
                break
            if after_snapshot is not None and after_snapshot.snapshot_id != before_snapshot.snapshot_id:
                snapshots.append(after_snapshot)
                current_snapshot = after_snapshot
            materially_changed = after_snapshot is not None and _snapshot_material_signature(before_snapshot) != _snapshot_material_signature(after_snapshot)
            if materially_changed and replan_count < max_replans:
                replan_count += 1
                previous_plan = current_plan
                current_plan = None
                replans.append(
                    OrchestrationReplanRecord(
                        replan_id=f"replan-{uuid4()}",
                        parent_run_id=parent_run_id,
                        reason=f"Material state change after {step.capability_name} {step.action}.",
                        previous_plan_output_id=previous_plan.output_id,
                        new_plan_output_id="pending",
                        before_snapshot_id=before_snapshot.snapshot_id,
                        after_snapshot_id=after_snapshot.snapshot_id,
                    )
                )
                continue
            if materially_changed and replan_count >= max_replans:
                step_results.append(
                    OrchestrationStepResult(
                        step_id=f"step-{uuid4()}",
                        capability_name="control_tower",
                        action="block",
                        status="blocked",
                        rationale="Maximum replan limit reached after material state changes.",
                        parent_run_id=parent_run_id,
                        validation_status="needs_human_review",
                        state_changed=False,
                        before_snapshot_id=current_snapshot.snapshot_id,
                        plan_output_id=current_plan.output_id,
                    )
                )
                break

        final_snapshot = current_snapshot
        replans = _fill_replan_new_plan_ids(tuple(replans), tuple(plans))
        execution_mode_summary = summarize_execution_modes(
            tuple(control_tower_traces),
            reused_artifacts=sum(1 for result in step_results if result.status == "reused"),
        )
        report = _build_control_tower_report(
            parent_run_id=parent_run_id,
            request=request,
            initial_snapshot=initial_snapshot,
            final_snapshot=final_snapshot,
            plans=tuple(plans),
            step_results=tuple(step_results),
            replans=replans,
            execution_mode_summary=execution_mode_summary,
        )
        failed = plan_validation_failed or any(result.status == "failed" for result in step_results)
        blocked = any(result.status == "blocked" for result in step_results) or any(plan.blocked for plan in plans)
        validation_status = "failed" if failed else "needs_human_review" if blocked else "passed"
        run = WorkflowRun(
            run_id=parent_run_id,
            workflow_name="control_tower_orchestration",
            status="completed",
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            input_provenance="control_tower.orchestrate",
            validation_status=validation_status,
            gate_reason="; ".join(report.unresolved_gates or report.unavailable_modules) or None,
            metadata={"objective": request.objective[:500], "nct_id": request.nct_id, "child_run_ids": child_run_ids},
        )
        record = OrchestrationRunRecord(
            run_id=parent_run_id,
            request=request,
            snapshot=initial_snapshot,
            snapshots=tuple(snapshots),
            final_snapshot=final_snapshot,
            plan=plans[-1] if plans else _empty_plan(parent_run_id, request, initial_snapshot),
            plans=tuple(plans),
            step_results=tuple(step_results),
            replans=replans,
            child_run_ids=tuple(dict.fromkeys(child_run_ids)),
            report=report,
            validation_results=tuple(validation_results),
            execution_mode_summary=execution_mode_summary,
        )
        self.memory.save_run(
            run,
            input_payload=request,
            output_payload=record,
            trace_metadata={
                "max_steps": max_steps,
                "max_replans": max_replans,
                "step_count": len(step_results),
                "replan_count": len(replans),
                "execution_mode_summary": execution_mode_summary.model_dump(mode="json"),
            },
        )
        self.memory.save_validation_results(parent_run_id, tuple(validation_results))
        return record

    def _run_registered_workflow(
        self,
        workflow_name: str,
        input_data: ClinicalTrialIntelligenceInput | DueDiligenceInput | ClinicalOutcomePredictionInput | ProtocolDesignInput | None,
    ) -> object:
        runner = _WORKFLOW_RUNNERS.get(workflow_name)
        input_type = _INPUT_TYPES.get(workflow_name)
        if runner is None or input_type is None:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        if input_data is None:
            raise ValueError(f"{workflow_name} requires {input_type.__name__}")
        if not isinstance(input_data, input_type):
            raise ValueError(f"{workflow_name} requires {input_type.__name__}")
        return runner(input_data, memory=self.memory)

    def _execute_orchestration_step(
        self,
        *,
        parent_run_id: str,
        request: OrchestrationRequest,
        plan: ExecutionPlan,
        step: PlannedStep,
        before_snapshot: ScientificStateSnapshot,
    ) -> tuple[OrchestrationStepResult, ScientificStateSnapshot | None]:
        capability = self.registry.get(step.capability_name)
        if capability is None:
            return (
                _step_result(parent_run_id, plan, step, "failed", "Unknown capability.", before_snapshot),
                None,
            )
        if step.action == "skip":
            return (
                _step_result(parent_run_id, plan, step, "skipped", step.reason, before_snapshot),
                None,
            )
        if step.action == "block":
            return (
                _step_result(parent_run_id, plan, step, "blocked", step.reason, before_snapshot, validation_status="needs_human_review"),
                None,
            )
        if step.action == "reuse":
            gates = _artifact_gates(before_snapshot, step.reuse_run_id)
            return (
                _step_result(
                    parent_run_id,
                    plan,
                    step,
                    "reused",
                    step.reason,
                    before_snapshot,
                    reused_run_id=step.reuse_run_id,
                    reused_output_id=step.reuse_output_id,
                    validation_status="needs_human_review" if gates else "passed",
                    gates=gates,
                    execution_mode="reused_artifact",
                ),
                None,
            )
        if step.action not in {"run", "refresh"}:
            return (
                _step_result(parent_run_id, plan, step, "failed", f"Unsupported action {step.action}.", before_snapshot, validation_status="failed"),
                None,
            )
        if not capability.executable:
            return (
                _step_result(parent_run_id, plan, step, "blocked", f"{capability.name} is not executable.", before_snapshot, validation_status="needs_human_review"),
                None,
            )
        workflow_input = _workflow_input_from_request(
            capability_name=capability.name,
            input_schema=getattr(capability, "input_schema", None),
            request=request,
            force_refresh=step.action == "refresh",
        )
        if workflow_input is None:
            return (
                _step_result(parent_run_id, plan, step, "blocked", f"{capability.name} requires inputs not present in the orchestration request.", before_snapshot, validation_status="needs_human_review"),
                None,
            )
        try:
            output = self._run_registered_workflow(capability.name, workflow_input)
        except Exception as exc:
            return (
                _step_result(parent_run_id, plan, step, "failed", f"{capability.name} execution failed: {exc}", before_snapshot, validation_status="failed"),
                None,
            )
        after_snapshot = self.memory.build_scientific_state_snapshot(request, registry=self.registry)
        child_run_id = getattr(output, "run_id", None)
        output_id = getattr(output, "output_id", None)
        gates = (getattr(output, "human_gate", None),)
        gates = tuple(gate for gate in gates if gate is not None)
        child_execution_summary = getattr(output, "execution_mode_summary", None)
        return (
            _step_result(
                parent_run_id,
                plan,
                step,
                "refreshed" if step.action == "refresh" else "executed",
                step.reason,
                before_snapshot,
                after_snapshot=after_snapshot,
                child_run_id=child_run_id,
                output_id=output_id,
                validation_status=getattr(output, "validation_status", "passed"),
                gates=gates,
                state_changed=_snapshot_material_signature(before_snapshot) != _snapshot_material_signature(after_snapshot),
                execution_mode=primary_execution_mode(child_execution_summary) if child_execution_summary is not None else "deterministic_fallback",
            ),
            after_snapshot,
        )


def _next_unhandled_step(plan: ExecutionPlan, completed_capabilities: set[str]) -> PlannedStep | None:
    for step in plan.steps:
        if step.capability_name not in completed_capabilities:
            return step
    return None


def _workflow_input_from_request(
    *,
    capability_name: str,
    input_schema: str | None,
    request: OrchestrationRequest,
    force_refresh: bool,
) -> ClinicalOutcomePredictionInput | DueDiligenceInput | ProtocolDesignInput | None:
    if input_schema in {None, "ClinicalOutcomePredictionInput"} and capability_name == "clinical_outcome_prediction":
        if not request.nct_id:
            return None
        return ClinicalOutcomePredictionInput(
            nct_id=request.nct_id,
            pos_workbook_path=_str_assumption(request, "pos_workbook_path"),
        )
    if input_schema == "DueDiligenceInput" or capability_name == "due_diligence":
        if not request.nct_id:
            return None
        return DueDiligenceInput(
            nct_id=request.nct_id,
            pos_workbook_path=_str_assumption(request, "pos_workbook_path"),
            wac_data_path=_str_assumption(request, "wac_data_path"),
            annual_patients=_float_assumption(request, "annual_patients"),
            peak_penetration=_float_assumption(request, "peak_penetration"),
            gross_to_net=_float_assumption(request, "gross_to_net"),
            operating_margin=_float_assumption(request, "operating_margin"),
            discount_rate=_float_assumption(request, "discount_rate"),
            development_cost=_float_assumption(request, "development_cost"),
            launch_year=_int_assumption(request, "launch_year"),
            loe_year=_int_assumption(request, "loe_year"),
            refresh_agent3=force_refresh and "clinical_outcome_prediction" in request.force_refresh,
        )
    if input_schema == "ProtocolDesignInput" or capability_name == "protocol_design":
        if not request.nct_id:
            return None
        return ProtocolDesignInput(
            nct_id=request.nct_id,
            pos_workbook_path=_str_assumption(request, "pos_workbook_path"),
            wac_data_path=_str_assumption(request, "wac_data_path"),
            annual_patients=_float_assumption(request, "annual_patients"),
            peak_penetration=_float_assumption(request, "peak_penetration"),
            gross_to_net=_float_assumption(request, "gross_to_net"),
            operating_margin=_float_assumption(request, "operating_margin"),
            discount_rate=_float_assumption(request, "discount_rate"),
            development_cost=_float_assumption(request, "development_cost"),
            launch_year=_int_assumption(request, "launch_year"),
            loe_year=_int_assumption(request, "loe_year"),
            refresh_agent3=force_refresh and "clinical_outcome_prediction" in request.force_refresh,
            refresh_agent4=force_refresh and "due_diligence" in request.force_refresh,
        )
    return None


def _str_assumption(request: OrchestrationRequest, key: str) -> str | None:
    value = request.assumptions.get(key)
    return str(value) if isinstance(value, str) and value else None


def _float_assumption(request: OrchestrationRequest, key: str) -> float | None:
    value = request.assumptions.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_assumption(request: OrchestrationRequest, key: str) -> int | None:
    value = request.assumptions.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _step_result(
    parent_run_id: str,
    plan: ExecutionPlan,
    step: PlannedStep,
    status: str,
    rationale: str,
    before_snapshot: ScientificStateSnapshot,
    *,
    after_snapshot: ScientificStateSnapshot | None = None,
    child_run_id: str | None = None,
    output_id: str | None = None,
    reused_run_id: str | None = None,
    reused_output_id: str | None = None,
    validation_status: str = "passed",
    gates: tuple[object, ...] = (),
    state_changed: bool = False,
    execution_mode: str = "deterministic_fallback",
) -> OrchestrationStepResult:
    return OrchestrationStepResult(
        step_id=step.step_id,
        capability_name=step.capability_name,
        action=step.action,
        status=status,
        rationale=rationale,
        parent_run_id=parent_run_id,
        child_run_id=child_run_id,
        output_id=output_id,
        reused_run_id=reused_run_id,
        reused_output_id=reused_output_id,
        validation_status=validation_status,
        gates=tuple(gate for gate in gates if gate is not None),
        state_changed=state_changed,
        before_snapshot_id=before_snapshot.snapshot_id,
        after_snapshot_id=after_snapshot.snapshot_id if after_snapshot else None,
        plan_output_id=plan.output_id,
        execution_mode=execution_mode,  # type: ignore[arg-type]
    )


def _artifact_gates(snapshot: ScientificStateSnapshot, run_id: str | None) -> tuple[object, ...]:
    if not run_id:
        return ()
    return tuple(gate for artifact in snapshot.artifacts if artifact.run_id == run_id for gate in artifact.open_gates)


def _snapshot_material_signature(snapshot: ScientificStateSnapshot) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                artifact.producer_workflow,
                artifact.artifact_type,
                artifact.run_id,
                artifact.output_id,
                artifact.validation_status,
                artifact.compatibility,
                artifact.freshness,
                tuple(gate.decision for gate in artifact.open_gates),
            )
            for artifact in snapshot.artifacts
        )
    )


def _fill_replan_new_plan_ids(
    replans: tuple[OrchestrationReplanRecord, ...],
    plans: tuple[ExecutionPlan, ...],
) -> tuple[OrchestrationReplanRecord, ...]:
    filled = []
    for index, replan in enumerate(replans):
        new_plan_index = index + 1
        new_plan_id = plans[new_plan_index].output_id if new_plan_index < len(plans) else replan.new_plan_output_id
        filled.append(replan.model_copy(update={"new_plan_output_id": new_plan_id}))
    return tuple(filled)


def _build_control_tower_report(
    *,
    parent_run_id: str,
    request: OrchestrationRequest,
    initial_snapshot: ScientificStateSnapshot,
    final_snapshot: ScientificStateSnapshot,
    plans: tuple[ExecutionPlan, ...],
    step_results: tuple[OrchestrationStepResult, ...],
    replans: tuple[OrchestrationReplanRecord, ...],
    execution_mode_summary: object,
) -> ControlTowerReport:
    unavailable = tuple(
        dict.fromkeys(
            result.capability_name
            for result in step_results
            if result.status == "blocked" and result.capability_name != "control_tower"
        )
    )
    unresolved_gates = tuple(
        dict.fromkeys(
            gate.gate_reason
            for snapshot in (initial_snapshot, final_snapshot)
            for gate in snapshot.open_gates
            if gate.decision in {"needs_human_review", "blocked", "rejected"}
        )
    )
    return ControlTowerReport(
        report_id=f"control-tower-report-{parent_run_id}",
        parent_run_id=parent_run_id,
        objective=request.objective,
        initial_snapshot_id=initial_snapshot.snapshot_id,
        final_snapshot_id=final_snapshot.snapshot_id,
        initial_state_summary=_state_summary(initial_snapshot),
        final_state_summary=_state_summary(final_snapshot),
        pending_decision_summary=_pending_decision_summary(final_snapshot),
        evidence_requirement_summaries=_evidence_requirement_summaries(final_snapshot),
        critical_evidence_gaps=final_snapshot.critical_evidence_gaps,
        unresolved_claims=final_snapshot.unresolved_claims,
        contradictory_claims=final_snapshot.contradictory_claims,
        plan_summaries=tuple(_plan_summary(plan) for plan in plans),
        step_summaries=tuple(_step_summary(result) for result in step_results),
        unresolved_gates=unresolved_gates,
        unavailable_modules=unavailable,
        replan_summaries=tuple(replan.reason for replan in replans),
        execution_mode_summary=execution_mode_summary,  # type: ignore[arg-type]
    )


def _state_summary(snapshot: ScientificStateSnapshot) -> str:
    compatible = sum(1 for artifact in snapshot.artifacts if artifact.compatibility == "compatible")
    incompatible = sum(1 for artifact in snapshot.artifacts if artifact.compatibility == "incompatible")
    return f"{len(snapshot.artifacts)} artifacts observed: {compatible} compatible, {incompatible} incompatible, {len(snapshot.open_gates)} open gates."


def _pending_decision_summary(snapshot: ScientificStateSnapshot) -> str | None:
    decision = snapshot.pending_decision
    if decision is None:
        return None
    return f"{decision.decision_type} via {decision.target_capability_name}: {decision.requested_decision}"


def _evidence_requirement_summaries(snapshot: ScientificStateSnapshot) -> tuple[str, ...]:
    satisfaction = {result.requirement_id: result for result in snapshot.requirement_satisfaction}
    summaries: list[str] = []
    for requirement in snapshot.evidence_requirements:
        result = satisfaction.get(requirement.requirement_id)
        status = result.status if result else "missing"
        gaps = "; ".join(result.gaps) if result and result.gaps else "no open gap"
        summaries.append(f"{requirement.requirement_id} ({requirement.criticality}): {status} - {gaps}")
    return tuple(summaries)


def _plan_summary(plan: ExecutionPlan) -> str:
    actions = ", ".join(f"{step.capability_name}:{step.action}" for step in plan.steps) or "no steps"
    return f"{plan.output_id}: {actions}"


def _step_summary(result: OrchestrationStepResult) -> str:
    target = result.child_run_id or result.reused_run_id or result.output_id or result.reused_output_id or "no output"
    return f"{result.capability_name} {result.action} -> {result.status} ({target})"


def _empty_plan(parent_run_id: str, request: OrchestrationRequest, snapshot: ScientificStateSnapshot) -> ExecutionPlan:
    return ExecutionPlan(
        output_id=f"control-tower-plan-{parent_run_id}-empty",
        run_id=parent_run_id,
        request=request,
        snapshot_id=snapshot.snapshot_id,
        objective_interpretation="No plan was produced.",
        steps=(),
        blocked=True,
        block_reasons=("No plan was produced.",),
        validation_status="failed",
        confidence=0.0,
        provenance="pharma_os.orchestrator.empty_plan",
    )
