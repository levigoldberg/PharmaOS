"""Deterministic scientific-state helpers for Control Tower planning."""

from __future__ import annotations

from uuid import uuid4

from pharma_os.schemas import (
    ArtifactStatus,
    DecisionType,
    EvidenceRequirement,
    HumanGate,
    ModuleCapability,
    OrchestrationRequest,
    PendingDecision,
    RequirementSatisfaction,
)


def infer_pending_decision(request: OrchestrationRequest, capabilities: tuple[ModuleCapability, ...]) -> PendingDecision:
    """Infer the downstream decision deterministically from request fields."""

    decision_type = request.decision_type or _decision_type_from_text(request)
    capability = _target_capability(decision_type, request, capabilities)
    return PendingDecision(
        decision_id=f"decision-{uuid4()}",
        decision_type=decision_type,
        lifecycle_stage=capability.lifecycle_stage if capability else "clinical_development",
        target_capability_name=capability.name if capability else "unknown",
        requested_decision=request.objective,
        rationale=_decision_rationale(decision_type, capability),
    )


def evidence_requirements_for_decision(
    pending_decision: PendingDecision,
    capabilities: tuple[ModuleCapability, ...],
) -> tuple[EvidenceRequirement, ...]:
    """Return registry requirements relevant to a pending decision."""

    requirements = [
        requirement
        for capability in capabilities
        for requirement in capability.evidence_requirements
        if requirement.decision_type == pending_decision.decision_type
        or requirement.capability_name == pending_decision.target_capability_name
    ]
    if not requirements and pending_decision.target_capability_name != "unknown":
        requirements = [
            requirement
            for capability in capabilities
            if capability.name == pending_decision.target_capability_name
            for requirement in capability.evidence_requirements
        ]
    deduped: dict[str, EvidenceRequirement] = {}
    for requirement in requirements:
        deduped.setdefault(requirement.requirement_id, requirement)
    return tuple(deduped.values())


def assess_requirement_satisfaction(
    requirements: tuple[EvidenceRequirement, ...],
    artifacts: tuple[ArtifactStatus, ...],
) -> tuple[RequirementSatisfaction, ...]:
    """Assess which existing artifacts satisfy decision evidence requirements."""

    return tuple(_assess_requirement(requirement, artifacts) for requirement in requirements)


def critical_evidence_gaps(satisfaction: tuple[RequirementSatisfaction, ...], requirements: tuple[EvidenceRequirement, ...]) -> tuple[str, ...]:
    critical = {requirement.requirement_id: requirement for requirement in requirements if requirement.criticality in {"high", "critical"}}
    gaps = []
    for result in satisfaction:
        if result.requirement_id in critical and result.status != "satisfied":
            gaps.extend(result.gaps or (critical[result.requirement_id].description,))
    return tuple(dict.fromkeys(gaps))


def unresolved_claims_from_state(
    artifacts: tuple[ArtifactStatus, ...],
    gates: tuple[HumanGate, ...],
    satisfaction: tuple[RequirementSatisfaction, ...],
) -> tuple[str, ...]:
    claims = [
        reason
        for artifact in artifacts
        for reason in artifact.reasons
        if any(token in reason.casefold() for token in ("missing", "failed", "low", "human", "gate", "stale", "incompatible"))
    ]
    claims.extend(gate.gate_reason for gate in gates if gate.decision in {"needs_human_review", "blocked", "rejected"})
    claims.extend(claim for result in satisfaction for claim in result.unresolved_claims)
    return tuple(dict.fromkeys(claims))


def contradictory_claims_from_state(artifacts: tuple[ArtifactStatus, ...], satisfaction: tuple[RequirementSatisfaction, ...]) -> tuple[str, ...]:
    claims = [
        reason
        for artifact in artifacts
        for reason in artifact.reasons
        if any(token in reason.casefold() for token in ("contradict", "conflict", "mismatch", "differs"))
    ]
    claims.extend(claim for result in satisfaction if result.status == "contradicted" for claim in result.unresolved_claims)
    return tuple(dict.fromkeys(claims))


def blocked_capabilities(capabilities: tuple[ModuleCapability, ...]) -> tuple[str, ...]:
    return tuple(capability.name for capability in capabilities if not capability.executable or capability.implementation_status != "implemented")


def _assess_requirement(requirement: EvidenceRequirement, artifacts: tuple[ArtifactStatus, ...]) -> RequirementSatisfaction:
    relevant = [
        artifact
        for artifact in artifacts
        if artifact.artifact_type in requirement.satisfying_artifact_types
        and artifact.producer_workflow in requirement.accepted_producers
    ]
    gates = tuple(gate for artifact in relevant for gate in artifact.open_gates)
    unresolved = tuple(
        reason
        for artifact in relevant
        for reason in artifact.reasons
        if any(token in reason.casefold() for token in ("missing", "failed", "low", "stale", "incompatible", "gate"))
    )
    contradictory = tuple(
        reason
        for artifact in relevant
        for reason in artifact.reasons
        if any(token in reason.casefold() for token in ("contradict", "conflict", "mismatch", "differs"))
    )
    blocking_gates = tuple(gate for gate in gates if gate.decision in {"blocked", "rejected"})
    review_gates = tuple(gate for gate in gates if gate.decision == "needs_human_review")
    compatible = [
        artifact
        for artifact in relevant
        if artifact.compatibility == "compatible"
        and artifact.validation_status in requirement.validation_statuses
        and (not requirement.freshness_required or artifact.freshness != "stale")
    ]
    stale_or_incompatible = [
        artifact
        for artifact in relevant
        if artifact.compatibility != "compatible" or (requirement.freshness_required and artifact.freshness == "stale")
    ]
    required_producers = set(requirement.accepted_producers)
    satisfied_producers = {artifact.producer_workflow for artifact in compatible}

    if blocking_gates or (requirement.human_gate_must_be_clear and review_gates):
        status = "blocked"
    elif contradictory:
        status = "contradicted"
    elif required_producers and required_producers <= satisfied_producers:
        status = "satisfied"
    elif compatible:
        status = "partially_satisfied"
    elif stale_or_incompatible:
        status = "stale"
    else:
        status = "missing"

    gaps = _requirement_gaps(requirement, status, required_producers, satisfied_producers, stale_or_incompatible)
    confidence = 0.9 if status == "satisfied" else 0.55 if status == "partially_satisfied" else 0.25
    return RequirementSatisfaction(
        requirement_id=requirement.requirement_id,
        status=status,
        satisfying_artifact_output_ids=tuple(
            dict.fromkeys(str(artifact.output_id) for artifact in compatible if artifact.output_id)
        ),
        satisfying_run_ids=tuple(dict.fromkeys(artifact.run_id for artifact in compatible)),
        gaps=gaps,
        unresolved_claims=tuple(dict.fromkeys((*unresolved, *contradictory))),
        gates=gates,
        confidence=confidence,
    )


def _requirement_gaps(
    requirement: EvidenceRequirement,
    status: str,
    required_producers: set[str],
    satisfied_producers: set[str],
    stale_or_incompatible: list[ArtifactStatus],
) -> tuple[str, ...]:
    if status == "satisfied":
        return ()
    gaps = []
    missing = sorted(required_producers - satisfied_producers)
    if missing:
        gaps.append(f"{requirement.requirement_id} missing compatible artifacts from: {', '.join(missing)}.")
    if stale_or_incompatible:
        gaps.append(f"{requirement.requirement_id} has stale or incompatible artifacts that cannot satisfy the decision.")
    if not gaps:
        gaps.append(f"{requirement.requirement_id} is {status} for the pending decision.")
    return tuple(dict.fromkeys(gaps))


def _decision_type_from_text(request: OrchestrationRequest) -> DecisionType:
    text = f"{request.objective} {' '.join(request.identifiers.values())}".casefold()
    if any(term in text for term in ("phase ii", "phase 2", "phase iii", "phase 3", "pivotal", "phase transition")):
        return "phase_transition"
    if any(term in text for term in ("protocol", "next study", "next-study", "study design", "agent 5")):
        return "protocol_design"
    if any(term in text for term in ("diligence", "asset memo", "commercial", "rnpv", "pricing", "agent 4")):
        return "clinical_stage_due_diligence"
    if any(term in text for term in ("clinical risk", "outcome", "probability of success", "pos", "agent 3")):
        return "clinical_risk_assessment"
    if any(term in text for term in ("discovery", "target nomination", "target discovery")):
        return "discovery_prioritization"
    if any(term in text for term in ("tox", "pkpd", "pk/pd", "safety package", "dose escalation")):
        return "tox_pkpd_safety"
    if any(term in text for term in ("enrollment", "feasibility", "site", "country startup")):
        return "enrollment_feasibility"
    if any(term in text for term in ("trial execution", "ctms", "edc", "monitoring")):
        return "trial_execution"
    if any(term in text for term in ("manufacturing", "cmc", "biofactory", "batch")):
        return "manufacturing_control"
    if any(term in text for term in ("launch", "pharmacovigilance", "pv", "postmarketing")):
        return "launch_pv"
    if any(term in text for term in ("regulatory", "quality audit", "qms", "submission")):
        return "regulatory_quality_audit"
    if request.nct_id:
        return "clinical_risk_assessment"
    return "unknown"


def _target_capability(decision_type: DecisionType, request: OrchestrationRequest, capabilities: tuple[ModuleCapability, ...]) -> ModuleCapability | None:
    names = {
        "clinical_risk_assessment": "clinical_outcome_prediction",
        "clinical_stage_due_diligence": "due_diligence",
        "phase_transition": "protocol_design",
        "protocol_design": "protocol_design",
        "enrollment_feasibility": "enrollment_feasibility",
        "trial_execution": "trial_execution",
        "manufacturing_control": "manufacturing_biofactory",
        "launch_pv": "launch_pv",
        "regulatory_quality_audit": "regulatory_quality_audit",
        "discovery_prioritization": "discovery",
        "tox_pkpd_safety": "tox_pkpd_safety",
    }
    requested_name = request.identifiers.get("target_capability") or names.get(decision_type)
    for capability in capabilities:
        if capability.name == requested_name:
            return capability
    return None


def _decision_rationale(decision_type: DecisionType, capability: ModuleCapability | None) -> str:
    if capability is None:
        return "The request did not map to a registered capability."
    return f"{decision_type} maps to {capability.name}; evidence requirements determine run, reuse, refresh, block, or stop."
