"""Control Tower planning primitives for memory-aware orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pharma_os.agent_runtime import (
    AgentRuntimeConfig,
    StructuredAgentResult,
    agents_sdk_output_schema,
    load_agents_sdk,
    run_structured_agent,
    runtime_config_for_live_agents,
)
from pharma_os.registry import WorkflowRegistry
from pharma_os.schemas import (
    ArtifactStatus,
    ExecutionPlan,
    ModuleCapability,
    OrchestrationRequest,
    PlannedStep,
    ScientificStateSnapshot,
    ValidationResult,
)


def run_control_tower_agent(
    *,
    run_id: str,
    request: OrchestrationRequest,
    snapshot: ScientificStateSnapshot,
    registry: WorkflowRegistry | None = None,
    config: AgentRuntimeConfig | None = None,
    plan_feedback: tuple[str, ...] = (),
) -> StructuredAgentResult:
    """Return a typed execution plan without executing workflows."""

    effective_registry = registry or WorkflowRegistry.default()
    runtime_config = config or runtime_config_for_live_agents(
        disabled_provenance="pharma_os.control_tower",
        model_route="control_tower",
    )
    fallback = build_deterministic_execution_plan(
        run_id=run_id,
        request=request,
        snapshot=snapshot,
        registry=effective_registry,
    )
    agent = object()
    if not runtime_config.disabled:
        Agent, _, _, _ = load_agents_sdk()
        agent = Agent(
            name="ControlTowerAgent",
            instructions=_control_tower_instructions(),
            model=runtime_config.model,
            output_type=agents_sdk_output_schema(ExecutionPlan),
        )
    return run_structured_agent(
        agent=agent,
        payload={
            "request": request.model_dump(mode="json"),
            "scientific_state_snapshot": snapshot.model_dump(mode="json"),
            "pending_decision": snapshot.pending_decision.model_dump(mode="json") if snapshot.pending_decision else None,
            "evidence_requirements": [requirement.model_dump(mode="json") for requirement in snapshot.evidence_requirements],
            "requirement_satisfaction": [item.model_dump(mode="json") for item in snapshot.requirement_satisfaction],
            "critical_evidence_gaps": snapshot.critical_evidence_gaps,
            "unresolved_claims": snapshot.unresolved_claims,
            "contradictory_claims": snapshot.contradictory_claims,
            "human_gates": [gate.model_dump(mode="json") for gate in snapshot.open_gates],
            "workflow_registry": [capability.model_dump(mode="json") for capability in effective_registry.capabilities()],
            "prior_plan_validation_feedback": plan_feedback,
            "constraint": "Return only an ExecutionPlan. Do not execute workflows or fabricate unavailable module outputs.",
        },
        output_type=ExecutionPlan,
        agent_name="ControlTowerAgent",
        run_id=run_id,
        input_summary=f"Plan minimum justified path for objective: {request.objective[:160]}",
        config=runtime_config,
        offline_output=fallback,
        source_ids=(),
        confidence=fallback.confidence,
        rationale_summary="Control Tower produced a typed ExecutionPlan for the orchestration loop.",
    )


def build_deterministic_execution_plan(
    *,
    run_id: str,
    request: OrchestrationRequest,
    snapshot: ScientificStateSnapshot,
    registry: WorkflowRegistry,
) -> ExecutionPlan:
    """Build an offline plan from registry and memory state."""

    target_names = _infer_target_capabilities(request, registry, snapshot=snapshot)
    ordered_names = _dependency_order(target_names, registry)
    steps: list[PlannedStep] = []
    changed_capabilities: set[str] = set()
    blocked = False
    block_reasons: list[str] = []

    for name in ordered_names:
        capability = registry.require(name)
        dependencies = tuple(dep for dep in capability.dependencies if dep in ordered_names)
        artifact = _best_artifact_for_capability(capability, snapshot.artifacts)
        dependency_changed = any(dep in changed_capabilities for dep in dependencies)
        step = _planned_step(
            capability=capability,
            request=request,
            snapshot=snapshot,
            artifact=artifact,
            dependency_changed=dependency_changed,
            dependencies=dependencies,
        )
        if step.action in {"run", "refresh"}:
            changed_capabilities.add(capability.name)
        if step.action == "block":
            blocked = True
            block_reasons.extend(step.blocked_by)
        steps.append(step)
        if (
            step.action == "block"
            and _target_only_requested(request)
            and step.capability_name != _scoped_target_capability(request, snapshot)
        ):
            break

    if not steps:
        blocked = True
        block_reasons.append("No registered capability matched the orchestration objective.")
    return ExecutionPlan(
        output_id=f"control-tower-plan-{run_id}",
        run_id=run_id,
        request=request,
        snapshot_id=snapshot.snapshot_id,
        objective_interpretation=_objective_interpretation(request, target_names),
        steps=tuple(steps),
        blocked=blocked,
        block_reasons=tuple(dict.fromkeys(block_reasons)),
        validation_status="not_run",
        confidence=0.72 if steps and not blocked else 0.45,
        provenance="pharma_os.control_tower.deterministic_fallback_planner",
    )


def validate_execution_plan(
    *,
    run_id: str,
    plan: ExecutionPlan,
    snapshot: ScientificStateSnapshot,
    registry: WorkflowRegistry,
) -> tuple[ValidationResult, ...]:
    """Validate a Control Tower execution plan against deterministic rules."""

    results: list[ValidationResult] = []
    known = set(registry.names())
    ordered = [step.capability_name for step in plan.steps]
    satisfied: set[str] = _memory_satisfied_capabilities(snapshot.artifacts, registry)
    changed: set[str] = set()
    seen_capabilities: dict[str, str] = {}
    allowed_capabilities = _allowed_target_path_capabilities(plan.request, snapshot, registry)

    for index, step in enumerate(plan.steps):
        capability = registry.get(step.capability_name)
        failures: list[str] = []
        prior_action = seen_capabilities.get(step.capability_name)
        if prior_action is not None:
            failures.append(f"duplicate capability step: {step.capability_name} already planned as {prior_action}")
        seen_capabilities[step.capability_name] = step.action
        if step.capability_name not in known or capability is None:
            failures.append("unknown capability")
        else:
            if allowed_capabilities and step.capability_name not in allowed_capabilities:
                failures.append("capability outside requested target path")
            relevant_requirements = _requirements_for_capability(snapshot, capability)
            unsatisfied_requirements = _unsatisfied_requirements_for_capability(snapshot, capability)
            if step.action != "block":
                missing_deps = [dep for dep in capability.dependencies if dep not in satisfied]
                if missing_deps:
                    dep_positions = {dep: ordered.index(dep) for dep in capability.dependencies if dep in ordered}
                    current_position = index
                    late_deps = [dep for dep, position in dep_positions.items() if position > current_position]
                    if late_deps:
                        failures.append(f"dependency ordering invalid: {', '.join(late_deps)} must appear earlier")
                    else:
                        failures.append(f"missing dependency: {', '.join(missing_deps)}")
            if step.action in {"run", "refresh"} and (not capability.executable or capability.implementation_status != "implemented"):
                failures.append("non-implemented capability cannot be executed")
            if (
                _target_only_requested(plan.request)
                and step.action in {"run", "refresh"}
                and step.capability_name != _scoped_target_capability(plan.request, snapshot)
            ):
                failures.append("target-only request cannot execute non-target dependency")
            if step.action in {"run", "refresh"} and snapshot.evidence_requirements and not relevant_requirements:
                failures.append("capability does not address pending decision requirements")
            if step.action in {"run", "refresh"} and relevant_requirements and not set(step.requirements_addressed):
                failures.append("execution step does not cite decision evidence requirements")
            if step.action in {"run", "refresh"} and relevant_requirements and not unsatisfied_requirements and not _refresh_justified(step, capability, plan.request, snapshot.artifacts, changed):
                failures.append("capability does not address unmet decision requirements")
            if step.action == "reuse" and not _has_compatible_artifact(capability, snapshot.artifacts, step):
                failures.append("reuse references missing or incompatible artifact")
            if step.action == "reuse" and relevant_requirements and not _reuse_satisfies_addressed_requirements(step, snapshot):
                failures.append("reuse does not satisfy cited decision evidence requirements")
            if step.action == "reuse" and _force_refreshes(capability, plan.request):
                failures.append("reuse ignores force_refresh request")
            if step.action == "refresh" and not _refresh_justified(step, capability, plan.request, snapshot.artifacts, changed):
                failures.append("refresh is not justified by force_refresh, incompatibility, staleness, or dependency changes")
            if step.action in {"run", "refresh"} and _has_blocking_dependency_gate(capability, snapshot.artifacts):
                failures.append("execution would proceed through a blocking human gate")
            if step.action in {"run", "refresh"} and _has_compatible_artifact(capability, snapshot.artifacts, step) and not _refresh_justified(step, capability, plan.request, snapshot.artifacts, changed):
                failures.append("unnecessary rerun when a compatible artifact is available")
            if snapshot.contradictory_claims and step.action not in {"block", "refresh", "run"} and not step.human_gate_required:
                failures.append("plan does not address unresolved contradictory claims")

        if step.action in {"run", "refresh"}:
            changed.add(step.capability_name)
        if not failures and step.action in {"run", "reuse", "refresh"}:
            satisfied.add(step.capability_name)
        results.append(_plan_validation_result(run_id, index, step, failures))

    if not plan.steps:
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-plan-empty",
                target_id=plan.output_id,
                status="failed",
                validator="control_tower_plan_validator",
                message="ExecutionPlan contains no steps.",
                confidence=1.0,
                gate_reason="No executable or blocking plan was produced.",
                provenance="pharma_os.control_tower.validate_execution_plan",
            )
        )
    return tuple(results)


def _planned_step(
    *,
    capability: ModuleCapability,
    request: OrchestrationRequest,
    snapshot: ScientificStateSnapshot,
    artifact: ArtifactStatus | None,
    dependency_changed: bool,
    dependencies: tuple[str, ...],
) -> PlannedStep:
    forced = _force_refreshes(capability, request)
    blocking_reasons = _blocking_reasons(capability, artifact)
    requirements = _requirements_for_capability(snapshot, capability)
    unsatisfied = _unsatisfied_requirements_for_capability(snapshot, capability)
    requirements_addressed = tuple(requirement.requirement_id for requirement in (unsatisfied or requirements))
    scoped_target = _scoped_target_capability(request, snapshot)
    target_only_dependency = _target_only_requested(request) and scoped_target is not None and capability.name != scoped_target
    if _skip_requested(capability, request):
        return PlannedStep(
            step_id=f"step-{uuid4()}",
            capability_name=capability.name,
            action="skip",
            reason=f"Objective explicitly requested skipping {capability.name}.",
            required_artifacts=capability.required_artifacts,
            produced_artifacts=capability.produced_artifacts,
            depends_on=dependencies,
            executable=False,
            confidence=0.75,
            requirements_addressed=requirements_addressed,
            decision_rationale="Explicit user-requested skip; skipped requirements remain unsatisfied.",
            stop_reason="Skipped by objective.",
        )
    if target_only_dependency and (not artifact or artifact.compatibility != "compatible"):
        return PlannedStep(
            step_id=f"step-{uuid4()}",
            capability_name=capability.name,
            action="block",
            reason=f"{capability.name} is required by {scoped_target}, but target-only scope forbids running or refreshing dependencies.",
            required_artifacts=capability.required_artifacts,
            produced_artifacts=capability.produced_artifacts,
            depends_on=dependencies,
            blocked_by=(f"target-only scope requires an existing compatible {capability.name} artifact",),
            human_gate_required=True,
            executable=False,
            confidence=0.85,
            requirements_addressed=requirements_addressed,
            decision_rationale="The user scoped execution to only the requested target workflow; missing dependencies require clarification or a broader scope.",
            stop_reason="Run the dependency first or request the full dependency path.",
        )
    if not capability.executable or capability.implementation_status != "implemented":
        return PlannedStep(
            step_id=f"step-{uuid4()}",
            capability_name=capability.name,
            action="block",
            reason=f"{capability.name} is registered as {capability.implementation_status} and is not executable.",
            required_artifacts=capability.required_artifacts,
            produced_artifacts=capability.produced_artifacts,
            depends_on=dependencies,
            blocked_by=tuple(dict.fromkeys((*blocking_reasons, *capability.missing_connectors))),
            human_gate_required=True,
            executable=False,
            confidence=0.85,
            requirements_addressed=requirements_addressed,
            decision_rationale=f"{capability.name} is in the registry but cannot execute without required connectors/data.",
            stop_reason=f"{capability.name} remains blocked until missing connectors are available.",
        )
    if artifact and artifact.compatibility == "compatible" and not forced and not dependency_changed:
        return PlannedStep(
            step_id=f"step-{uuid4()}",
            capability_name=capability.name,
            action="reuse",
            reason=f"Reuse compatible {capability.name} artifact from run {artifact.run_id}.",
            required_artifacts=capability.required_artifacts,
            produced_artifacts=capability.produced_artifacts,
            reuse_run_id=artifact.run_id,
            reuse_output_id=artifact.output_id,
            depends_on=dependencies,
            blocked_by=(),
            human_gate_required=bool(artifact.open_gates),
            executable=False,
            confidence=artifact.confidence or 0.65,
            requirements_addressed=requirements_addressed,
            decision_rationale=f"Compatible artifacts satisfy current decision requirements for {capability.name}.",
            expected_state_change="No state change; reuse existing Scientific Memory artifact.",
            stop_reason="Human review remains required for reused artifact gates." if artifact.open_gates else None,
        )
    if blocking_reasons:
        return PlannedStep(
            step_id=f"step-{uuid4()}",
            capability_name=capability.name,
            action="block",
            reason=f"{capability.name} cannot proceed until blocking gates or incompatible upstream artifacts are resolved.",
            required_artifacts=capability.required_artifacts,
            produced_artifacts=capability.produced_artifacts,
            reuse_run_id=artifact.run_id if artifact else None,
            reuse_output_id=artifact.output_id if artifact else None,
            depends_on=dependencies,
            blocked_by=tuple(dict.fromkeys(blocking_reasons)),
            human_gate_required=True,
            executable=False,
            confidence=0.8,
            requirements_addressed=requirements_addressed,
            decision_rationale=f"{capability.name} cannot safely close decision requirements until blocking issues are resolved.",
            stop_reason="Blocked by gates, incompatible upstream artifacts, or unavailable connectors.",
        )
    if artifact:
        reason = "Refresh because requested force_refresh applies." if forced else "Refresh because dependency outputs will change." if dependency_changed else "Refresh incompatible or stale artifact."
        return PlannedStep(
            step_id=f"step-{uuid4()}",
            capability_name=capability.name,
            action="refresh",
            reason=reason,
            required_artifacts=capability.required_artifacts,
            produced_artifacts=capability.produced_artifacts,
            reuse_run_id=artifact.run_id,
            reuse_output_id=artifact.output_id,
            depends_on=dependencies,
            blocked_by=(),
            human_gate_required=False,
            executable=True,
            confidence=0.68,
            requirements_addressed=requirements_addressed,
            decision_rationale=f"Refresh {capability.name} to satisfy stale, incompatible, forced, or dependency-changed requirements.",
            expected_state_change=f"Updated {', '.join(capability.produced_artifacts)} artifacts.",
        )
    mandatory_review = "mandatory" in capability.human_gate_policy.casefold()
    return PlannedStep(
        step_id=f"step-{uuid4()}",
        capability_name=capability.name,
        action="run",
        reason=f"Run {capability.name}; no compatible existing artifact is available.",
        required_artifacts=capability.required_artifacts,
        produced_artifacts=capability.produced_artifacts,
        depends_on=dependencies,
        blocked_by=(),
        human_gate_required=mandatory_review,
        executable=True,
        confidence=0.7,
        requirements_addressed=requirements_addressed,
        decision_rationale=f"Run {capability.name} because required decision evidence is missing.",
        expected_state_change=f"New {', '.join(capability.produced_artifacts)} artifacts.",
        stop_reason=capability.human_gate_policy if mandatory_review else None,
    )


def _infer_target_capabilities(request: OrchestrationRequest, registry: WorkflowRegistry, *, snapshot: ScientificStateSnapshot | None = None) -> tuple[str, ...]:
    explicit_target = request.identifiers.get("target_capability")
    if explicit_target in registry.names():
        return (explicit_target,)
    text = f"{request.objective} {' '.join(request.identifiers.values())}".casefold()
    targets: list[str] = []
    if snapshot and snapshot.pending_decision and snapshot.pending_decision.target_capability_name in registry.names():
        targets.append(snapshot.pending_decision.target_capability_name)
    keyword_map = (
        ("protocol_design", ("protocol", "next study", "next-study", "study design", "agent 5", "phase ii", "phase iii")),
        ("due_diligence", ("diligence", "asset memo", "commercial", "rnpv", "pricing", "agent 4")),
        ("clinical_outcome_prediction", ("clinical risk", "outcome", "probability of success", "pos", "agent 3")),
        ("discovery", ("discovery", "target nomination", "target discovery")),
        ("tox_pkpd_safety", ("tox", "pkpd", "pk/pd", "safety package", "dose escalation")),
        ("enrollment_feasibility", ("enrollment", "feasibility", "site", "country startup")),
        ("trial_execution", ("trial execution", "ctms", "edc", "monitoring")),
        ("manufacturing_biofactory", ("manufacturing", "cmc", "biofactory", "batch")),
        ("launch_pv", ("launch", "pharmacovigilance", "pv", "postmarketing")),
        ("regulatory_quality_audit", ("regulatory", "quality audit", "qms", "submission")),
    )
    for name, keywords in keyword_map:
        if name in registry.names() and any(keyword in text for keyword in keywords):
            targets.append(name)
    if not targets and request.nct_id:
        targets.append("clinical_outcome_prediction")
    return tuple(dict.fromkeys(targets))


def _dependency_order(target_names: tuple[str, ...], registry: WorkflowRegistry) -> tuple[str, ...]:
    ordered: list[str] = []

    def visit(name: str) -> None:
        capability = registry.get(name)
        if capability is None:
            return
        if not capability.executable or capability.implementation_status != "implemented":
            if name not in ordered:
                ordered.append(name)
            return
        for dependency in capability.dependencies:
            visit(dependency)
        if name not in ordered:
            ordered.append(name)

    for target in target_names:
        visit(target)
    return tuple(ordered)


def _best_artifact_for_capability(capability: ModuleCapability, artifacts: tuple[ArtifactStatus, ...]) -> ArtifactStatus | None:
    candidates = [
        artifact
        for artifact in artifacts
        if artifact.producer_workflow == capability.name or artifact.producer_workflow == getattr(capability, "workflow_name", capability.name)
    ]
    if not candidates:
        return None
    compatible = [artifact for artifact in candidates if artifact.compatibility == "compatible"]
    pool = compatible or candidates
    return sorted(pool, key=lambda item: item.completed_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[0]


def _has_compatible_artifact(capability: ModuleCapability, artifacts: tuple[ArtifactStatus, ...], step: PlannedStep | None = None) -> bool:
    if step and (step.reuse_run_id or step.reuse_output_id):
        return _referenced_compatible_artifact(capability, artifacts, step) is not None
    artifact = _best_artifact_for_capability(capability, artifacts)
    if artifact is None or artifact.compatibility != "compatible":
        return False
    return True


def _referenced_compatible_artifact(
    capability: ModuleCapability,
    artifacts: tuple[ArtifactStatus, ...],
    step: PlannedStep,
) -> ArtifactStatus | None:
    for artifact in artifacts:
        if artifact.producer_workflow not in {capability.name, getattr(capability, "workflow_name", capability.name)}:
            continue
        if artifact.compatibility != "compatible":
            continue
        if step.reuse_run_id and artifact.run_id != step.reuse_run_id:
            continue
        if step.reuse_output_id and artifact.output_id != step.reuse_output_id:
            continue
        return artifact
    return None


def _memory_satisfied_capabilities(artifacts: tuple[ArtifactStatus, ...], registry: WorkflowRegistry) -> set[str]:
    satisfied: set[str] = set()
    for capability in registry.capabilities():
        if _has_compatible_artifact(capability, artifacts):
            satisfied.add(capability.name)
    return satisfied


def _allowed_target_path_capabilities(
    request: OrchestrationRequest,
    snapshot: ScientificStateSnapshot,
    registry: WorkflowRegistry,
) -> set[str]:
    target = request.identifiers.get("target_capability")
    if not target and snapshot.pending_decision:
        target = snapshot.pending_decision.target_capability_name
    if not target or target not in registry.names():
        return set()
    return set(_dependency_order((target,), registry))


def _scoped_target_capability(request: OrchestrationRequest, snapshot: ScientificStateSnapshot | None) -> str | None:
    target = request.identifiers.get("target_capability")
    if not target and snapshot and snapshot.pending_decision:
        target = snapshot.pending_decision.target_capability_name
    return target or None


def _target_only_requested(request: OrchestrationRequest) -> bool:
    return request.identifiers.get("execution_scope") == "target_only"


def _refresh_justified(
    step: PlannedStep,
    capability: ModuleCapability,
    request: OrchestrationRequest,
    artifacts: tuple[ArtifactStatus, ...],
    changed_dependencies: set[str],
) -> bool:
    if _force_refreshes(capability, request):
        return True
    if any(dep in changed_dependencies for dep in capability.dependencies):
        return True
    artifact = _best_artifact_for_capability(capability, artifacts)
    if artifact and (artifact.compatibility == "incompatible" or artifact.freshness == "stale"):
        return True
    return bool("dependency" in step.reason.casefold() or "incompatible" in step.reason.casefold() or "stale" in step.reason.casefold())


def _has_blocking_dependency_gate(capability: ModuleCapability, artifacts: tuple[ArtifactStatus, ...]) -> bool:
    required = set(capability.required_artifacts)
    for artifact in artifacts:
        if artifact.artifact_type in required and any(gate.decision in {"blocked", "rejected"} for gate in artifact.open_gates):
            return True
    return False


def _requirements_for_capability(snapshot: ScientificStateSnapshot, capability: ModuleCapability) -> tuple[object, ...]:
    return tuple(
        requirement
        for requirement in snapshot.evidence_requirements
        if capability.name in requirement.accepted_producers
    )


def _unsatisfied_requirements_for_capability(snapshot: ScientificStateSnapshot, capability: ModuleCapability) -> tuple[object, ...]:
    satisfaction = {result.requirement_id: result for result in snapshot.requirement_satisfaction}
    return tuple(
        requirement
        for requirement in _requirements_for_capability(snapshot, capability)
        if satisfaction.get(requirement.requirement_id) is None
        or satisfaction[requirement.requirement_id].status != "satisfied"
    )


def _reuse_satisfies_addressed_requirements(step: PlannedStep, snapshot: ScientificStateSnapshot) -> bool:
    if not step.requirements_addressed:
        return True
    satisfaction = {result.requirement_id: result for result in snapshot.requirement_satisfaction}
    for requirement_id in step.requirements_addressed:
        result = satisfaction.get(requirement_id)
        if result is None or result.status not in {"satisfied", "partially_satisfied"}:
            return False
        if step.reuse_output_id and step.reuse_output_id not in result.satisfying_artifact_output_ids:
            return False
        if step.reuse_run_id and step.reuse_run_id not in result.satisfying_run_ids:
            return False
    return True


def _force_refreshes(capability: ModuleCapability, request: OrchestrationRequest) -> bool:
    forced = {item.strip() for item in request.force_refresh}
    return bool({capability.name, *capability.produced_artifacts} & forced)


def _skip_requested(capability: ModuleCapability, request: OrchestrationRequest) -> bool:
    text = request.objective.casefold()
    name = capability.name.casefold()
    parsed_skips = {
        item.strip().casefold()
        for item in request.identifiers.get("skip_capabilities", "").split(",")
        if item.strip()
    }
    if name in parsed_skips:
        return True
    aliases = {
        "clinical_outcome_prediction": ("clinical risk", "agent 3", "clinical outcome prediction"),
        "due_diligence": ("diligence", "agent 4"),
        "protocol_design": ("protocol design", "agent 5", "next study"),
    }.get(capability.name, (capability.name.replace("_", " "),))
    return f"skip {name}" in text or any(f"skip {alias}" in text for alias in aliases)


def _blocking_reasons(capability: ModuleCapability, artifact: ArtifactStatus | None) -> tuple[str, ...]:
    reasons = []
    if not capability.executable:
        reasons.extend(capability.missing_connectors)
    if artifact:
        reasons.extend(reason for reason in artifact.reasons if "blocking" in reason or "gate" in reason)
        reasons.extend(gate.gate_reason for gate in artifact.open_gates if gate.decision in {"blocked", "rejected"})
    return tuple(dict.fromkeys(reasons))


def _plan_validation_result(run_id: str, index: int, step: PlannedStep, failures: list[str]) -> ValidationResult:
    status = "failed" if failures else "passed"
    return ValidationResult(
        validation_id=f"validation-{run_id}-plan-step-{index + 1}",
        target_id=step.step_id,
        status=status,
        validator="control_tower_plan_validator",
        message="; ".join(failures) if failures else f"{step.capability_name} {step.action} step passed deterministic plan validation.",
        confidence=1.0,
        gate_reason="; ".join(failures) if failures else None,
        provenance="pharma_os.control_tower.validate_execution_plan",
    )


def _objective_interpretation(request: OrchestrationRequest, target_names: tuple[str, ...]) -> str:
    if target_names:
        return f"Objective maps to registered capabilities: {', '.join(target_names)}."
    return "Objective did not map to a registered executable or skeleton capability."


def _control_tower_instructions() -> str:
    return (
        "You are ControlTowerAgent. You only produce a typed ExecutionPlan. "
        "Do not execute workflows. Use registry metadata, current scientific state, validation, gates, compatibility, freshness, "
        "dependencies, and force_refresh to choose run, reuse, refresh, skip, or block. "
        "If prior_plan_validation_feedback is supplied, correct the plan so it passes those deterministic validation checks. "
        "Only plan the requested target capability and its registry dependencies; do not add downstream workflows that the "
        "objective did not request. If request.identifiers.execution_scope is target_only, reuse compatible dependencies when "
        "needed but do not run or refresh non-target dependencies unless the request explicitly broadens scope. "
        "Do not blindly plan Agent 3 to Agent 4 to Agent 5; choose the minimum justified path. "
        "Block unavailable skeleton capabilities and state missing connectors."
    )
