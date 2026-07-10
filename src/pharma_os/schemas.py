"""Shared strict Pydantic schemas for PharmaOS workflows.

These models are intentionally workflow-agnostic.  Workflow-specific schemas should
compose or extend these primitives rather than redefining provenance, evidence,
confidence, validation, or human-gate fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


ConfidenceLevel = Literal["very_low", "low", "medium", "high", "very_high"]
ValidationStatus = Literal["not_run", "passed", "failed", "warning", "needs_human_review"]
GateDecision = Literal["approved", "rejected", "needs_human_review", "blocked"]
WorkflowStatus = Literal["pending", "running", "completed", "failed", "blocked"]
MetadataValue = str | int | float | bool | None | list[str] | list[int] | list[float] | list[bool]
ExecutionMode = Literal["live_agent", "direct_llm", "deterministic_fallback", "reused_artifact"]


class StrictSchema(BaseModel):
    """Base class for strict workflow contracts."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        populate_by_name=True,
    )


class ExecutionModeSummary(StrictSchema):
    """Visible audit summary of how AI reasoning was executed."""

    requested_reasoning_steps: int = Field(default=0, ge=0)
    live_agent_calls_completed: int = Field(default=0, ge=0)
    direct_llm_calls_completed: int = Field(default=0, ge=0)
    live_ai_calls_completed: int = Field(default=0, ge=0)
    deterministic_fallbacks_used: int = Field(default=0, ge=0)
    reused_artifacts_used: int = Field(default=0, ge=0)
    summary: str = "0 reasoning steps requested, 0 live AI calls completed, 0 deterministic fallbacks used."


class SourceMetadata(StrictSchema):
    """Provenance metadata for a source used by a workflow or report."""

    source_id: str = Field(..., min_length=1, description="Stable source identifier.")
    title: str | None = Field(default=None, description="Human-readable source title.")
    url: HttpUrl | None = Field(default=None, description="Canonical source URL, when available.")
    authors: tuple[str, ...] = Field(default_factory=tuple, description="Source authors or organizations.")
    published_at: datetime | None = Field(default=None, description="Publication timestamp, when known.")
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the source was retrieved or observed.",
    )
    provenance: str = Field(..., min_length=1, description="Where and how this source entered the workflow.")
    source_type: str | None = Field(default=None, description="Source category, such as paper, registry, label, or website.")
    version: str | None = Field(default=None, description="Source version, revision, or access date label.")


class EvidenceClaim(StrictSchema):
    """A claim extracted from evidence with explicit source provenance."""

    claim_id: str = Field(..., min_length=1, description="Stable claim identifier.")
    claim_text: str = Field(..., min_length=1, description="Verbatim or normalized claim text.")
    source_ids: tuple[str, ...] = Field(..., min_length=1, description="Source IDs supporting this claim.")
    provenance: str = Field(..., min_length=1, description="Extraction method, agent, or workflow step that produced the claim.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Claim confidence score from 0.0 to 1.0.")
    confidence_level: ConfidenceLevel = Field(..., description="Bucketed confidence label.")
    qualifiers: tuple[str, ...] = Field(default_factory=tuple, description="Important conditions, caveats, or population qualifiers.")


class ConfidenceFlag(StrictSchema):
    """A confidence concern raised against a claim, source, or workflow output."""

    flag_id: str = Field(..., min_length=1, description="Stable confidence flag identifier.")
    target_id: str = Field(..., min_length=1, description="Claim, source, validation, or output ID affected by this flag.")
    reason: str = Field(..., min_length=1, description="Why confidence was reduced or review is needed.")
    severity: Literal["info", "low", "medium", "high", "critical"] = Field(..., description="Flag severity.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score after considering this flag.")
    source_ids: tuple[str, ...] = Field(default_factory=tuple, description="Sources relevant to this flag.")
    provenance: str = Field(..., min_length=1, description="Workflow step or rule that raised this flag.")


class HumanGate(StrictSchema):
    """Human review gate for regulated or low-confidence workflow decisions."""

    gate_id: str = Field(..., min_length=1, description="Stable human-gate identifier.")
    decision: GateDecision = Field(..., description="Current human-gate decision.")
    gate_reason: str = Field(..., min_length=1, description="Why the gate was required or how it was resolved.")
    required_roles: tuple[str, ...] = Field(default_factory=tuple, description="Roles required to review this gate.")
    reviewer: str | None = Field(default=None, description="Reviewer identifier, when assigned or completed.")
    reviewed_at: datetime | None = Field(default=None, description="Review completion timestamp.")
    source_ids: tuple[str, ...] = Field(default_factory=tuple, description="Sources relevant to the gate.")
    provenance: str = Field(..., min_length=1, description="Workflow step or policy that created the gate.")


class ValidationResult(StrictSchema):
    """Validation status for a claim, output, report, or workflow run."""

    validation_id: str = Field(..., min_length=1, description="Stable validation identifier.")
    target_id: str = Field(..., min_length=1, description="Validated claim, output, report, or run ID.")
    status: ValidationStatus = Field(..., description="Validation status.")
    validator: str = Field(..., min_length=1, description="Validator name, rule, or system.")
    message: str = Field(..., min_length=1, description="Validation finding or summary.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the validation result.")
    source_ids: tuple[str, ...] = Field(default_factory=tuple, description="Sources checked during validation.")
    gate_reason: str | None = Field(default=None, description="Reason human review is needed, if applicable.")
    provenance: str = Field(..., min_length=1, description="Workflow step that performed validation.")


class WorkflowRun(StrictSchema):
    """Execution envelope shared by all workflows."""

    run_id: str = Field(..., min_length=1, description="Stable workflow-run identifier.")
    workflow_name: str = Field(..., min_length=1, description="Workflow name.")
    status: WorkflowStatus = Field(..., description="Workflow execution status.")
    started_at: datetime = Field(..., description="Workflow start timestamp.")
    completed_at: datetime | None = Field(default=None, description="Workflow completion timestamp.")
    input_provenance: str = Field(..., min_length=1, description="Origin of the workflow inputs.")
    source_ids: tuple[str, ...] = Field(default_factory=tuple, description="Sources used by this run.")
    validation_status: ValidationStatus = Field(default="not_run", description="Aggregate validation status.")
    gate_reason: str | None = Field(default=None, description="Reason the run is blocked or needs review.")
    metadata: dict[str, MetadataValue] = Field(
        default_factory=dict,
        description="Strictly bounded workflow metadata.",
    )


class ClinicalTrialIntelligenceInput(StrictSchema):
    """Input contract for the first Clinical Trial Intelligence workflow."""

    disease: str = Field(..., min_length=1, description="Disease or indication to search.")
    drug: str | None = Field(default=None, description="Optional drug or intervention name.")
    target: str | None = Field(default=None, description="Optional biological target.")
    phase: str | None = Field(default=None, description="Optional trial phase filter.")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum trials to retrieve.")


class TrialIntervention(StrictSchema):
    """Normalized intervention from ClinicalTrials.gov."""

    name: str = Field(..., min_length=1)
    type: str | None = None
    description: str | None = None
    other_names: tuple[str, ...] = Field(default_factory=tuple)
    arm_group_labels: tuple[str, ...] = Field(default_factory=tuple)


class TrialArmGroup(StrictSchema):
    """Normalized ClinicalTrials.gov arm group and intervention mapping."""

    label: str = Field(..., min_length=1)
    type: str | None = None
    description: str | None = None
    intervention_names: tuple[str, ...] = Field(default_factory=tuple)


class TrialSponsor(StrictSchema):
    """Normalized trial sponsor or collaborator."""

    name: str = Field(..., min_length=1)
    sponsor_class: str | None = None


class TrialEndpoint(StrictSchema):
    """Normalized endpoint/outcome from ClinicalTrials.gov."""

    measure: str = Field(..., min_length=1)
    time_frame: str | None = None
    description: str | None = None
    endpoint_type: Literal["primary", "secondary", "other"] = "other"


class TrialLocation(StrictSchema):
    """Normalized trial site or geography from ClinicalTrials.gov."""

    facility: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    status: str | None = None


class ClinicalTrialRecord(StrictSchema):
    """A normalized ClinicalTrials.gov study record."""

    nct_id: str = Field(..., min_length=1)
    brief_title: str | None = None
    official_title: str | None = None
    overall_status: str | None = None
    phases: tuple[str, ...] = Field(default_factory=tuple)
    study_type: str | None = None
    allocation: str | None = None
    intervention_model: str | None = None
    masking: str | None = None
    observational_model: str | None = None
    number_of_arms: int | None = Field(default=None, ge=0)
    conditions: tuple[str, ...] = Field(default_factory=tuple)
    interventions: tuple[TrialIntervention, ...] = Field(default_factory=tuple)
    arm_groups: tuple[TrialArmGroup, ...] = Field(default_factory=tuple)
    lead_sponsor: TrialSponsor | None = None
    collaborators: tuple[TrialSponsor, ...] = Field(default_factory=tuple)
    enrollment_count: int | None = None
    enrollment_type: str | None = None
    start_date: str | None = None
    primary_completion_date: str | None = None
    completion_date: str | None = None
    results_available: bool = False
    primary_endpoints: tuple[TrialEndpoint, ...] = Field(default_factory=tuple)
    secondary_endpoints: tuple[TrialEndpoint, ...] = Field(default_factory=tuple)
    locations: tuple[TrialLocation, ...] = Field(default_factory=tuple)
    eligibility_criteria: str | None = None
    minimum_age: str | None = None
    maximum_age: str | None = None
    sex: str | None = None
    source_id: str = Field(..., min_length=1)


class TrialLandscapeRisk(StrictSchema):
    """A source-backed risk or uncertainty in the trial landscape."""

    risk_id: str = Field(..., min_length=1)
    trial_id: str | None = None
    risk_type: Literal[
        "terminated_or_withdrawn",
        "missing_results",
        "small_enrollment",
        "outdated_status",
        "unclear_endpoints",
        "tool_failure",
        "other",
    ]
    description: str = Field(..., min_length=1)
    severity: Literal["low", "medium", "high"] = "medium"
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class ClinicalTrialsSearchResult(StrictSchema):
    """Typed result from the ClinicalTrials.gov deterministic tool."""

    query: ClinicalTrialIntelligenceInput
    trials: tuple[ClinicalTrialRecord, ...] = Field(default_factory=tuple)
    sources: tuple[SourceMetadata, ...] = Field(default_factory=tuple)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    api_url: str = Field(..., min_length=1)
    errors: tuple[str, ...] = Field(default_factory=tuple)


class ClinicalTrialIntelligenceOutput(StrictSchema):
    """Structured output from Agent 3 trial-landscape mode."""

    output_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    input: ClinicalTrialIntelligenceInput
    trials: tuple[ClinicalTrialRecord, ...] = Field(default_factory=tuple)
    sources: tuple[SourceMetadata, ...] = Field(default_factory=tuple)
    claims: tuple[EvidenceClaim, ...] = Field(default_factory=tuple)
    risk_flags: tuple[TrialLandscapeRisk, ...] = Field(default_factory=tuple)
    landscape_summary: str = Field(..., min_length=1)
    status_summary: str = Field(..., min_length=1)
    phase_summary: str = Field(..., min_length=1)
    sponsor_summary: str = Field(..., min_length=1)
    endpoint_summary: str = Field(..., min_length=1)
    population_summary: str = Field(..., min_length=1)
    validation_results: tuple[ValidationResult, ...] = Field(default_factory=tuple)
    confidence_flags: tuple[ConfidenceFlag, ...] = Field(default_factory=tuple)
    human_gate: HumanGate | None = None
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    validation_status: ValidationStatus = "not_run"
    trace_metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    execution_mode_summary: ExecutionModeSummary = Field(default_factory=ExecutionModeSummary)


class DueDiligenceInput(StrictSchema):
    """Input contract for the PharmaOS-native due diligence workflow."""

    nct_id: str = Field(..., pattern=r"^NCT\d{8}$", description="ClinicalTrials.gov NCT identifier.")
    pos_workbook_path: str | None = Field(default=None, description="Optional PoS workbook path.")
    wac_data_path: str | None = Field(default=None, description="Optional local WAC workbook path.")
    annual_patients: float | None = Field(default=None, ge=0, description="Reviewed annual eligible patient assumption.")
    peak_penetration: float | None = Field(default=None, ge=0, le=1, description="Reviewed peak penetration assumption.")
    gross_to_net: float | None = Field(default=None, ge=0, le=1, description="Reviewed gross-to-net assumption.")
    operating_margin: float | None = Field(default=None, ge=0, le=1, description="Reviewed operating margin assumption.")
    discount_rate: float | None = Field(default=None, ge=0, le=1, description="Reviewed discount rate assumption.")
    development_cost: float | None = Field(default=None, ge=0, description="Reviewed remaining development cost assumption.")
    launch_year: int | None = Field(default=None, ge=2020, le=2100, description="Reviewed expected launch year.")
    loe_year: int | None = Field(default=None, ge=2020, le=2150, description="Reviewed expected loss-of-exclusivity year.")
    refresh_agent3: bool = Field(default=False, description="Force a fresh Agent 3 clinical outcome prediction handoff.")


class ProtocolDesignInput(StrictSchema):
    """Input contract for the Protocol Design Brief Agent 5 workflow."""

    nct_id: str = Field(..., pattern=r"^NCT\d{8}$", description="ClinicalTrials.gov NCT identifier.")
    pos_workbook_path: str | None = Field(default=None, description="Optional local PoS workbook path for upstream Agent 3/4 runs.")
    wac_data_path: str | None = Field(default=None, description="Optional local WAC workbook path for upstream Agent 4 runs.")
    annual_patients: float | None = Field(default=None, ge=0, description="Reviewed annual eligible patient assumption for upstream Agent 4 runs.")
    peak_penetration: float | None = Field(default=None, ge=0, le=1, description="Reviewed peak penetration assumption for upstream Agent 4 runs.")
    gross_to_net: float | None = Field(default=None, ge=0, le=1, description="Reviewed gross-to-net assumption for upstream Agent 4 runs.")
    operating_margin: float | None = Field(default=None, ge=0, le=1, description="Reviewed operating margin assumption for upstream Agent 4 runs.")
    discount_rate: float | None = Field(default=None, ge=0, le=1, description="Reviewed discount rate assumption for upstream Agent 4 runs.")
    development_cost: float | None = Field(default=None, ge=0, description="Reviewed remaining development cost assumption for upstream Agent 4 runs.")
    launch_year: int | None = Field(default=None, ge=2020, le=2100, description="Reviewed expected launch year for upstream Agent 4 runs.")
    loe_year: int | None = Field(default=None, ge=2020, le=2150, description="Reviewed expected loss-of-exclusivity year for upstream Agent 4 runs.")
    refresh_agent3: bool = Field(default=False, description="Force a fresh Agent 3 handoff through upstream workflows.")
    refresh_agent4: bool = Field(default=False, description="Force a fresh Agent 4 due-diligence handoff.")
    analog_top_k: int = Field(default=10, ge=1, le=25, description="Maximum selected analog trials.")


class ClinicalOutcomePredictionInput(StrictSchema):
    """Input contract for the Clinical Outcome Prediction Agent 3 workflow."""

    nct_id: str = Field(..., pattern=r"^NCT\d{8}$", description="ClinicalTrials.gov NCT identifier.")
    pos_workbook_path: str | None = Field(default=None, description="Optional local PoS workbook path.")


OrchestrationAction = Literal["run", "reuse", "refresh", "skip", "block"]
CapabilityLifecycleStage = Literal[
    "discovery",
    "preclinical",
    "clinical_development",
    "clinical_operations",
    "manufacturing",
    "launch_postmarketing",
    "quality_regulatory",
]
CapabilityImplementationStatus = Literal["implemented", "skeleton", "planned", "deprecated"]
ArtifactCompatibility = Literal["compatible", "incompatible", "unknown"]
ArtifactFreshness = Literal["fresh", "stale", "unknown"]
DecisionType = Literal[
    "clinical_risk_assessment",
    "clinical_stage_due_diligence",
    "phase_transition",
    "protocol_design",
    "enrollment_feasibility",
    "trial_execution",
    "manufacturing_control",
    "launch_pv",
    "regulatory_quality_audit",
    "discovery_prioritization",
    "tox_pkpd_safety",
    "unknown",
]
RequirementCriticality = Literal["low", "medium", "high", "critical"]
RequirementSatisfactionStatus = Literal["satisfied", "partially_satisfied", "missing", "contradicted", "stale", "blocked"]


class PendingDecision(StrictSchema):
    """Decision context the Control Tower is planning toward."""

    decision_id: str = Field(..., min_length=1)
    decision_type: DecisionType
    lifecycle_stage: CapabilityLifecycleStage
    target_capability_name: str = Field(..., min_length=1)
    requested_decision: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)


class EvidenceRequirement(StrictSchema):
    """Decision-critical evidence requirement that artifacts may satisfy."""

    requirement_id: str = Field(..., min_length=1)
    decision_type: DecisionType
    capability_name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    satisfying_artifact_types: tuple[str, ...] = Field(default_factory=tuple)
    accepted_producers: tuple[str, ...] = Field(default_factory=tuple)
    criticality: RequirementCriticality = "medium"
    freshness_required: bool = True
    validation_statuses: tuple[ValidationStatus, ...] = ("passed", "needs_human_review", "warning")
    human_gate_must_be_clear: bool = False


class RequirementSatisfaction(StrictSchema):
    """Deterministic assessment of whether existing state satisfies a requirement."""

    requirement_id: str = Field(..., min_length=1)
    status: RequirementSatisfactionStatus
    satisfying_artifact_output_ids: tuple[str, ...] = Field(default_factory=tuple)
    satisfying_run_ids: tuple[str, ...] = Field(default_factory=tuple)
    gaps: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_claims: tuple[str, ...] = Field(default_factory=tuple)
    gates: tuple[HumanGate, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class OrchestrationRequest(StrictSchema):
    """Request for the global Control Tower planning or orchestration loop."""

    objective: str = Field(..., min_length=1)
    nct_id: str | None = Field(default=None, pattern=r"^NCT\d{8}$")
    asset_name: str | None = None
    indication: str | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    assumptions: dict[str, MetadataValue] = Field(default_factory=dict)
    force_refresh: tuple[str, ...] = Field(default_factory=tuple)
    decision_type: DecisionType | None = None


class RequestUnderstandingAssumption(StrictSchema):
    """One assumption extracted from a natural-language orchestration goal."""

    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class RequestUnderstandingOutput(StrictSchema):
    """Structured parse of a natural-language orchestration goal."""

    normalized_objective: str = Field(..., min_length=1)
    target_capability: str | None
    decision_type: DecisionType
    nct_id: str | None
    asset_name: str | None
    indication: str | None
    assumptions: tuple[RequestUnderstandingAssumption, ...]
    force_refresh: tuple[str, ...]
    skip_capabilities: tuple[str, ...]
    requested_outputs: tuple[str, ...]
    missing_required_fields: tuple[str, ...]
    clarifying_questions: tuple[str, ...]
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale_summary: str = Field(..., min_length=1)


class ModuleCapability(StrictSchema):
    """Registry metadata describing a Control Tower capability."""

    name: str = Field(..., min_length=1)
    lifecycle_stage: CapabilityLifecycleStage
    implementation_status: CapabilityImplementationStatus
    accepted_inputs: tuple[str, ...] = Field(default_factory=tuple)
    required_artifacts: tuple[str, ...] = Field(default_factory=tuple)
    produced_artifacts: tuple[str, ...] = Field(default_factory=tuple)
    dependencies: tuple[str, ...] = Field(default_factory=tuple)
    executable: bool = False
    missing_connectors: tuple[str, ...] = Field(default_factory=tuple)
    human_gate_policy: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    evidence_requirements: tuple[EvidenceRequirement, ...] = Field(default_factory=tuple)


class WorkflowSpec(ModuleCapability):
    """Concrete workflow capability, including its Python implementation when present."""

    workflow_name: str = Field(..., min_length=1)
    input_schema: str | None = None
    output_schema: str | None = None
    implementation_path: str | None = None


class ArtifactStatus(StrictSchema):
    """Memory-derived status for one reusable workflow artifact."""

    artifact_type: str = Field(..., min_length=1)
    producer_workflow: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    output_id: str | None = None
    validation_status: ValidationStatus
    confidence: float | None = Field(default=None, ge=0, le=1)
    freshness: ArtifactFreshness = "unknown"
    compatibility: ArtifactCompatibility = "unknown"
    open_gates: tuple[HumanGate, ...] = Field(default_factory=tuple)
    upstream_references: tuple[str, ...] = Field(default_factory=tuple)
    input_fingerprint: str | None = None
    completed_at: datetime | None = None
    reasons: tuple[str, ...] = Field(default_factory=tuple)


class ScientificStateSnapshot(StrictSchema):
    """Current memory state relevant to a Control Tower planning request."""

    snapshot_id: str = Field(..., min_length=1)
    request: OrchestrationRequest
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifacts: tuple[ArtifactStatus, ...] = Field(default_factory=tuple)
    capabilities: tuple[ModuleCapability, ...] = Field(default_factory=tuple)
    open_gates: tuple[HumanGate, ...] = Field(default_factory=tuple)
    missing_artifacts: tuple[str, ...] = Field(default_factory=tuple)
    notes: tuple[str, ...] = Field(default_factory=tuple)
    pending_decision: PendingDecision | None = None
    evidence_requirements: tuple[EvidenceRequirement, ...] = Field(default_factory=tuple)
    requirement_satisfaction: tuple[RequirementSatisfaction, ...] = Field(default_factory=tuple)
    unresolved_claims: tuple[str, ...] = Field(default_factory=tuple)
    contradictory_claims: tuple[str, ...] = Field(default_factory=tuple)
    critical_evidence_gaps: tuple[str, ...] = Field(default_factory=tuple)
    stale_or_incompatible_artifacts: tuple[ArtifactStatus, ...] = Field(default_factory=tuple)
    blocked_capabilities: tuple[str, ...] = Field(default_factory=tuple)


class PlannedStep(StrictSchema):
    """One Control Tower action in a plan."""

    step_id: str = Field(..., min_length=1)
    capability_name: str = Field(..., min_length=1)
    action: OrchestrationAction
    reason: str = Field(..., min_length=1)
    required_artifacts: tuple[str, ...] = Field(default_factory=tuple)
    produced_artifacts: tuple[str, ...] = Field(default_factory=tuple)
    reuse_run_id: str | None = None
    reuse_output_id: str | None = None
    depends_on: tuple[str, ...] = Field(default_factory=tuple)
    blocked_by: tuple[str, ...] = Field(default_factory=tuple)
    human_gate_required: bool = False
    executable: bool = False
    confidence: float = Field(default=0.5, ge=0, le=1)
    requirements_addressed: tuple[str, ...] = Field(default_factory=tuple)
    decision_rationale: str | None = None
    expected_state_change: str | None = None
    stop_reason: str | None = None


class ExecutionPlan(StrictSchema):
    """Typed plan produced by the Control Tower agent."""

    output_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    request: OrchestrationRequest
    snapshot_id: str = Field(..., min_length=1)
    objective_interpretation: str = Field(..., min_length=1)
    steps: tuple[PlannedStep, ...] = Field(default_factory=tuple)
    blocked: bool = False
    block_reasons: tuple[str, ...] = Field(default_factory=tuple)
    validation_status: ValidationStatus = "not_run"
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)
    provenance: str = Field(..., min_length=1)


class OrchestrationStepResult(StrictSchema):
    """Persisted result for one Control Tower planned step."""

    step_id: str = Field(..., min_length=1)
    capability_name: str = Field(..., min_length=1)
    action: OrchestrationAction
    status: Literal["executed", "reused", "refreshed", "skipped", "blocked", "failed"]
    rationale: str = Field(..., min_length=1)
    parent_run_id: str = Field(..., min_length=1)
    child_run_id: str | None = None
    output_id: str | None = None
    reused_run_id: str | None = None
    reused_output_id: str | None = None
    validation_status: ValidationStatus = "not_run"
    gates: tuple[HumanGate, ...] = Field(default_factory=tuple)
    state_changed: bool = False
    before_snapshot_id: str = Field(..., min_length=1)
    after_snapshot_id: str | None = None
    plan_output_id: str = Field(..., min_length=1)
    execution_mode: ExecutionMode = "deterministic_fallback"


class OrchestrationReplanRecord(StrictSchema):
    """One replan event caused by a material state change."""

    replan_id: str = Field(..., min_length=1)
    parent_run_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    previous_plan_output_id: str = Field(..., min_length=1)
    new_plan_output_id: str = Field(..., min_length=1)
    before_snapshot_id: str = Field(..., min_length=1)
    after_snapshot_id: str = Field(..., min_length=1)


class ControlTowerReport(StrictSchema):
    """Human-facing Control Tower report payload."""

    report_id: str = Field(..., min_length=1)
    parent_run_id: str = Field(..., min_length=1)
    objective: str = Field(..., min_length=1)
    initial_snapshot_id: str = Field(..., min_length=1)
    final_snapshot_id: str = Field(..., min_length=1)
    initial_state_summary: str = Field(..., min_length=1)
    final_state_summary: str = Field(..., min_length=1)
    pending_decision_summary: str | None = None
    evidence_requirement_summaries: tuple[str, ...] = Field(default_factory=tuple)
    critical_evidence_gaps: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_claims: tuple[str, ...] = Field(default_factory=tuple)
    contradictory_claims: tuple[str, ...] = Field(default_factory=tuple)
    plan_summaries: tuple[str, ...] = Field(default_factory=tuple)
    step_summaries: tuple[str, ...] = Field(default_factory=tuple)
    unresolved_gates: tuple[str, ...] = Field(default_factory=tuple)
    unavailable_modules: tuple[str, ...] = Field(default_factory=tuple)
    replan_summaries: tuple[str, ...] = Field(default_factory=tuple)
    fallback_summaries: tuple[str, ...] = Field(default_factory=tuple)
    execution_mode_summary: ExecutionModeSummary = Field(default_factory=ExecutionModeSummary)


class HumanReadableFinding(StrictSchema):
    """One human-facing source-grounded finding from a workflow module."""

    title: str = Field(..., min_length=1)
    detail: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class HumanReadableModuleOutput(StrictSchema):
    """Structured human-readable summary generated from a typed workflow output."""

    output_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    module_name: Literal[
        "clinical_outcome_prediction",
        "due_diligence",
        "protocol_design",
        "trial_intelligence",
    ]
    module_display_name: str = Field(..., min_length=1)
    source_output_id: str = Field(..., min_length=1)
    headline: str = Field(..., min_length=1)
    plain_language_summary: str = Field(..., min_length=1)
    key_takeaways: tuple[str, ...] = Field(default_factory=tuple)
    key_findings: tuple[HumanReadableFinding, ...] = Field(default_factory=tuple)
    handoff_summary: str = Field(..., min_length=1)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    human_review_questions: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)
    provenance: str = Field(..., min_length=1)
    execution_mode: ExecutionMode = "deterministic_fallback"


class AssumptionRecord(StrictSchema):
    """A numeric or categorical assumption used by a deterministic calculator."""

    assumption_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    value: MetadataValue = None
    unit: str | None = None
    assumption_type: Literal[
        "source_derived",
        "user_reviewed",
        "config_default",
        "fallback_assumption",
        "calculated",
        "missing",
    ] = "user_reviewed"
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    provenance: str = Field(..., min_length=1)
    requires_human_review: bool = False


class MissingDataFlag(StrictSchema):
    """Missing or unavailable data needed for diligence confidence."""

    flag_id: str = Field(..., min_length=1)
    section: str = Field(..., min_length=1)
    field: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    severity: Literal["low", "medium", "high", "critical"] = "medium"


class RxNormMatch(StrictSchema):
    """Best-effort RxNorm normalized drug identity."""

    matched_name: str
    rxcui: str
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    source_id: str = Field(..., min_length=1)


class AssetIdentityOutput(StrictSchema):
    """Resolved trial asset, sponsor, modality, and indication identity."""

    nct_id: str
    asset_name: str | None = None
    raw_intervention_names: tuple[str, ...] = Field(default_factory=tuple)
    intervention_type: str | None = None
    aliases: tuple[str, ...] = Field(default_factory=tuple)
    rxnorm_match: RxNormMatch | None = None
    sponsor: str | None = None
    normalized_indication: str | None = None
    therapeutic_area: str | None = None
    modality: str | None = None
    rule_ids: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class PatentCandidate(StrictSchema):
    """Normalized patent search candidate from Lens."""

    candidate_id: str = Field(..., min_length=1)
    title: str | None = None
    jurisdiction: str | None = None
    publication_date: str | None = None
    legal_status: str | None = None
    source_id: str = Field(..., min_length=1)


class PatentExclusivityOutput(StrictSchema):
    """Patent and loss-of-exclusivity section."""

    asset_name: str | None = None
    searched_terms: tuple[str, ...] = Field(default_factory=tuple)
    candidates: tuple[PatentCandidate, ...] = Field(default_factory=tuple)
    estimated_loe_year: int | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class PoSOutput(StrictSchema):
    """Source-backed probability of success section."""

    probability_of_success: float | None = Field(default=None, ge=0, le=1)
    current_phase: str | None = None
    disease_area: str | None = None
    workbook_path: str | None = None
    lookup_key: str | None = None
    benchmark_row: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class SourceAvailabilityFlag(StrictSchema):
    """Typed availability status for desired public, local, or unavailable sources."""

    source_name: str = Field(..., min_length=1)
    status: Literal["available", "source_unavailable", "not_implemented"]
    reason: str = Field(..., min_length=1)
    source_type: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class SourceAvailabilityReport(StrictSchema):
    """Source availability summary for Agent 3."""

    flags: tuple[SourceAvailabilityFlag, ...] = Field(default_factory=tuple)


class TrialIdentity(StrictSchema):
    """Trial identity facts used by clinical outcome prediction."""

    nct_id: str = Field(..., min_length=1)
    brief_title: str | None = None
    official_title: str | None = None
    overall_status: str | None = None
    phases: tuple[str, ...] = Field(default_factory=tuple)
    conditions: tuple[str, ...] = Field(default_factory=tuple)
    sponsor: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class TrialDesignFeatures(StrictSchema):
    """Protocol design features relevant to clinical outcome risk."""

    study_type: str | None = None
    arms_count: int = Field(default=0, ge=0)
    intervention_count: int = Field(default=0, ge=0)
    enrollment_count: int | None = Field(default=None, ge=0)
    enrollment_type: str | None = None
    primary_endpoint_count: int = Field(default=0, ge=0)
    secondary_endpoint_count: int = Field(default=0, ge=0)
    primary_endpoint_measures: tuple[str, ...] = Field(default_factory=tuple)
    secondary_endpoint_measures: tuple[str, ...] = Field(default_factory=tuple)
    start_date: str | None = None
    primary_completion_date: str | None = None
    completion_date: str | None = None
    eligibility_summary: str | None = None
    countries: tuple[str, ...] = Field(default_factory=tuple)
    sites_count: int | None = Field(default=None, ge=0)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class EndpointRiskAssessment(StrictSchema):
    """Endpoint-level risk assessment with registry-backed rationale."""

    risk_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    risk_factors: tuple[str, ...] = Field(default_factory=tuple)
    rationale: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class EnrollmentDurationRisk(StrictSchema):
    """Enrollment and duration risk assessment with numeric provenance."""

    risk_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    enrollment_count: int | None = Field(default=None, ge=0)
    planned_duration_months: float | None = Field(default=None, ge=0)
    rationale: str = Field(..., min_length=1)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ComparatorBenchmarkBundle(StrictSchema):
    """Public benchmark trials matched by indication and phase."""

    matched_public_trials_count: int = Field(default=0, ge=0)
    comparator_trial_ids: tuple[str, ...] = Field(default_factory=tuple)
    benchmark_summary: str = Field(..., min_length=1)
    landscape_summary: str | None = None
    status_summary: str | None = None
    phase_summary: str | None = None
    sponsor_summary: str | None = None
    endpoint_summary: str | None = None
    population_summary: str | None = None
    risk_flags: tuple[TrialLandscapeRisk, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class HistoricalPoSEstimate(StrictSchema):
    """Historical probability-of-success estimate from the local source workbook."""

    probability_of_success: float | None = Field(default=None, ge=0, le=1)
    current_phase: str | None = None
    disease_area: str | None = None
    lookup_key: str | None = None
    benchmark_row: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    assumption_type: Literal["source_derived", "missing"] = "missing"
    source_type: str = "pos_workbook"
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ApprovalLikelihoodProxy(StrictSchema):
    """Source-backed or config-derived approval likelihood proxy, not a decision."""

    probability: float | None = Field(default=None, ge=0, le=1)
    basis: str = Field(..., min_length=1)
    assumption_type: Literal["source_derived", "heuristic", "missing"] = "missing"
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class FailureMode(StrictSchema):
    """One likely clinical failure mode for the trial."""

    category: Literal["endpoint", "enrollment", "safety", "comparator", "biology", "operational", "missing_data"]
    severity: Literal["low", "medium", "high"]
    rationale: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class FailureModeClassification(StrictSchema):
    """Structured failure-mode classification for Agent 4 consumption."""

    likely_failure_modes: tuple[FailureMode, ...] = Field(default_factory=tuple)
    overall_risk_level: Literal["low", "medium", "high", "unknown"] = "unknown"
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class SafetyContext(StrictSchema):
    """Label-derived safety context when an open public label is available."""

    label_available: bool = False
    summary: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class LabelExpansionClinicalRationale(StrictSchema):
    """Clinical rationale for label expansion; excludes commercial recommendations."""

    rationale: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ClinicalOutcomeManagerPlan(StrictSchema):
    """Manager-agent plan for coordinating Agent 3 clinical interpretation."""

    output_id: str = Field(..., min_length=1)
    nct_id: str = Field(..., min_length=1)
    ordered_agents: tuple[str, ...] = Field(default_factory=tuple)
    guardrail_summary: str = Field(..., min_length=1)
    rationale_summary: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class AssetIdentityAdjudication(StrictSchema):
    """Agent 3 adjudication record for ambiguous asset identity cases."""

    output_id: str = Field(..., min_length=1)
    nct_id: str = Field(..., min_length=1)
    is_ambiguous: bool = True
    ambiguity_reasons: tuple[str, ...] = Field(default_factory=tuple)
    recommended_asset_name: str | None = None
    recommended_modality: str | None = None
    recommended_indication: str | None = None
    review_questions: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ComparatorTrialRelevance(StrictSchema):
    """Agent 3 relevance judgment for one deterministically retrieved comparator trial."""

    nct_id: str = Field(..., min_length=1)
    relevance: Literal["relevant", "weak", "excluded"]
    rationale: str = Field(..., min_length=1)
    matched_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    mismatched_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ComparatorRelevanceOutput(StrictSchema):
    """ComparatorRelevanceAgent output over deterministic CT.gov comparator candidates."""

    output_id: str = Field(..., min_length=1)
    target_nct_id: str = Field(..., min_length=1)
    trial_relevance: tuple[ComparatorTrialRelevance, ...] = Field(default_factory=tuple)
    relevance_summary: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ClinicalOutcomePredictionOutput(StrictSchema):
    """Structured output from the Clinical Outcome Prediction Agent 3 workflow."""

    output_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    input: ClinicalOutcomePredictionInput
    trial_identity: TrialIdentity
    asset_identity: AssetIdentityOutput
    trial_design_features: TrialDesignFeatures
    endpoint_risk_assessment: EndpointRiskAssessment
    enrollment_duration_risk: EnrollmentDurationRisk
    comparator_benchmarking: ComparatorBenchmarkBundle
    historical_pos_estimate: HistoricalPoSEstimate
    approval_likelihood_proxy: ApprovalLikelihoodProxy
    failure_mode_classification: FailureModeClassification
    safety_context: SafetyContext
    label_expansion_clinical_rationale: LabelExpansionClinicalRationale
    source_availability: SourceAvailabilityReport
    sources: tuple[SourceMetadata, ...] = Field(default_factory=tuple)
    claims: tuple[EvidenceClaim, ...] = Field(default_factory=tuple)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    validation_results: tuple[ValidationResult, ...] = Field(default_factory=tuple)
    confidence_flags: tuple[ConfidenceFlag, ...] = Field(default_factory=tuple)
    human_gate: HumanGate | None = None
    human_readable_summary: HumanReadableModuleOutput | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    validation_status: ValidationStatus = "not_run"
    execution_mode_summary: ExecutionModeSummary = Field(default_factory=ExecutionModeSummary)


class PricingOutput(StrictSchema):
    """Pricing benchmark from local WAC data and openFDA label/dosing evidence."""

    annual_wac: float | None = Field(default=None, ge=0)
    wac_value: float | None = Field(default=None, ge=0)
    wac_unit_basis: str | None = None
    matched_product: str | None = None
    dosing_summary: str | None = None
    annualization_details: dict[str, MetadataValue] = Field(default_factory=dict)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class RevenueForecastYear(StrictSchema):
    """One deterministic commercial model revenue row."""

    year: int
    treated_patients: float
    net_price: float
    net_revenue: float


CaseName = Literal["downside", "base", "upside"]
MarketBasis = Literal[
    "prevalence_stock",
    "incidence_flow",
    "procedure_flow",
    "hybrid_prevalent_plus_incident",
]
AssumptionSourceType = Literal[
    "source_derived",
    "model_inferred",
    "default_assumption",
    "user_override",
    "fallback",
    "missing",
]


class ValueTriplet(StrictSchema):
    """Low/base/high numeric assumption values."""

    low: float | None = None
    base: float | None = None
    high: float | None = None


class CommercialAssumptionTriplet(StrictSchema):
    """AI-selected or defaulted market-sizing fraction with provenance."""

    low: float | None = None
    base: float | None = None
    high: float | None = None
    source_type: AssumptionSourceType
    rationale: str = Field(..., min_length=1)
    evidence_reference: str | None = None
    confidence_score: int = Field(default=0, ge=0, le=10)
    human_review_required: bool = True


class SelectedPopulationMeasure(StrictSchema):
    """Selected population denominator for commercial market sizing."""

    value: float | None = None
    unit: str | None = None
    measure_type: str | None = None
    condition: str | None = None
    geography: str | None = None
    source_type: AssumptionSourceType
    rationale: str = Field(..., min_length=1)
    evidence_reference: str | None = None
    confidence_score: int = Field(default=0, ge=0, le=10)
    human_review_required: bool = True


class MarketSizingInterpretation(StrictSchema):
    """Structured Agent 4 market-sizing interpretation for deterministic calculation."""

    calculable: bool
    selected_market_archetype: str | None = None
    market_basis: MarketBasis | None = None
    selected_population_measure: SelectedPopulationMeasure
    yearly_eligible_patient_logic: str = Field(..., min_length=1)
    diagnosed_fraction: CommercialAssumptionTriplet
    treated_fraction: CommercialAssumptionTriplet
    eligibility_fraction: CommercialAssumptionTriplet
    commercially_addressable_fraction: CommercialAssumptionTriplet
    rationale: str = Field(..., min_length=1)
    confidence_score: int = Field(default=0, ge=0, le=10)
    key_evidence_used: tuple[str, ...] = Field(default_factory=tuple)
    assumption_flags: tuple[str, ...] = Field(default_factory=tuple)
    human_review_flags: tuple[str, ...] = Field(default_factory=tuple)


class CommercialInputBundle(StrictSchema):
    """Compact evidence bundle supplied to the market-sizing interpretation agent."""

    asset_summary: dict[str, Any] = Field(default_factory=dict)
    disease_population_evidence: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    prevalence_evidence: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    incidence_evidence: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    segmentation_evidence: tuple[dict[str, Any], ...] = Field(default_factory=tuple)
    trial_eligibility_criteria: dict[str, Any] = Field(default_factory=dict)
    pricing_benchmark: dict[str, Any] = Field(default_factory=dict)
    missing_inputs: tuple[str, ...] = Field(default_factory=tuple)
    user_overrides: dict[str, Any] = Field(default_factory=dict)
    predefined_archetype_assumptions: dict[str, Any] = Field(default_factory=dict)


class PatientFunnel(StrictSchema):
    """Base-case patient funnel used by commercial model calculations."""

    starting_population: float
    diagnosed_patients: float
    treated_or_managed_patients: float
    eligible_patients: float
    commercially_addressable_patients: float
    diagnosed_fraction: float
    treated_fraction: float
    eligibility_fraction: float
    commercially_addressable_fraction: float


class CommercialPricingCase(StrictSchema):
    """Pricing assumptions for one commercial case."""

    annual_gross_wac: float
    gross_to_net: float
    net_price: float


class CommercialPenetrationCase(StrictSchema):
    """Penetration assumptions for one commercial case."""

    peak_penetration: float
    launch_ramp: tuple[float, ...] = Field(default_factory=tuple)


class CommercialCaseOutput(StrictSchema):
    """One downside/base/upside commercial forecast case."""

    patient_funnel: PatientFunnel
    pricing: CommercialPricingCase
    penetration: CommercialPenetrationCase
    revenue_forecast: tuple[RevenueForecastYear, ...] = Field(default_factory=tuple)
    peak_gross_sales: float
    peak_net_sales: float


class CommercialAssumptionLedgerRecord(StrictSchema):
    """Human-reviewable assumption ledger entry for Agent 4 commercial sizing."""

    assumption_name: str = Field(..., min_length=1)
    value: Any = None
    low: float | None = None
    base: float | None = None
    high: float | None = None
    unit: str | None = None
    source_type: AssumptionSourceType
    rationale: str = Field(..., min_length=1)
    evidence_reference: str | None = None
    confidence_score: int = Field(default=0, ge=0, le=10)
    human_review_required: bool = True


class CommercialModelOutput(StrictSchema):
    """Deterministic commercial model section."""

    calculable: bool
    annual_patients: float | None = None
    peak_penetration: float | None = None
    gross_to_net: float | None = None
    net_price: float | None = None
    peak_net_sales: float | None = None
    revenue_forecast: tuple[RevenueForecastYear, ...] = Field(default_factory=tuple)
    selected_market_archetype: str | None = None
    market_basis: MarketBasis | None = None
    selected_population_measure: SelectedPopulationMeasure | None = None
    patient_funnel: PatientFunnel | None = None
    cases: dict[CaseName, CommercialCaseOutput] = Field(default_factory=dict)
    assumption_ledger: tuple[CommercialAssumptionLedgerRecord, ...] = Field(default_factory=tuple)
    commercial_input_bundle_summary: dict[str, Any] = Field(default_factory=dict)
    confidence_flags: tuple[str, ...] = Field(default_factory=tuple)
    human_review_questions: tuple[str, ...] = Field(default_factory=tuple)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class RNPVOutput(StrictSchema):
    """Deterministic risk-adjusted NPV section."""

    calculable: bool
    rnpv: float | None = None
    probability_of_success: float | None = Field(default=None, ge=0, le=1)
    loe_year: int | None = None
    launch_year: int | None = None
    discount_rate: float | None = Field(default=None, ge=0, le=1)
    operating_margin: float | None = Field(default=None, ge=0, le=1)
    development_cost: float | None = Field(default=None, ge=0)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class Agent3HandoffReference(StrictSchema):
    """Reference to an Agent 3 output consumed by Agent 4."""

    agent3_run_id: str = Field(..., min_length=1)
    agent3_output_id: str = Field(..., min_length=1)
    nct_id: str = Field(..., min_length=1)
    generated_or_reused: Literal["generated", "reused"]
    retrieved_from_memory: bool
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ClinicalRiskSummary(StrictSchema):
    """Agent 3 clinical-risk summary consumed by Agent 4."""

    nct_id: str = Field(..., min_length=1)
    asset_name: str | None = None
    indication: str | None = None
    phase: str | None = None
    sponsor: str | None = None
    endpoint_risk_level: str | None = None
    enrollment_duration_risk_level: str | None = None
    failure_modes: tuple[FailureMode, ...] = Field(default_factory=tuple)
    historical_pos: float | None = Field(default=None, ge=0, le=1)
    approval_likelihood_proxy: float | None = Field(default=None, ge=0, le=1)
    safety_context_summary: str | None = None
    comparator_benchmark_summary: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)


class ClinicalEvidenceSummary(StrictSchema):
    """CT.gov and PubMed evidence extracted for Agent 4 diligence."""

    nct_id: str = Field(..., min_length=1)
    ctgov_summary: str = Field(..., min_length=1)
    pubmed_query: str | None = None
    pubmed_article_count: int = Field(default=0, ge=0)
    pubmed_titles: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    claims: tuple[EvidenceClaim, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class CompetitiveLandscapeSummary(StrictSchema):
    """Competitive landscape summarized from Agent 3 comparator context."""

    nct_id: str = Field(..., min_length=1)
    comparator_trial_ids: tuple[str, ...] = Field(default_factory=tuple)
    matched_public_trials_count: int = Field(default=0, ge=0)
    benchmark_summary: str = Field(..., min_length=1)
    status_summary: str | None = None
    phase_summary: str | None = None
    sponsor_summary: str | None = None
    endpoint_summary: str | None = None
    population_summary: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class SafetyLabelSummary(StrictSchema):
    """openFDA safety label summary without unsupported inference."""

    asset_name: str | None = None
    label_available: bool = False
    warnings_summary: str | None = None
    adverse_reactions_summary: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class PatentLOEReview(StrictSchema):
    """Lens-only patent and LOE review for Agent 4 diligence."""

    asset_name: str | None = None
    searched_terms: tuple[str, ...] = Field(default_factory=tuple)
    candidate_count: int = Field(default=0, ge=0)
    estimated_loe_year: int | None = None
    review_summary: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class DiligenceRedFlag(StrictSchema):
    """Rule-based due-diligence red flag."""

    flag_id: str = Field(..., min_length=1)
    category: Literal["clinical", "safety", "ip_loe", "pricing", "commercial", "rnpv", "source_coverage", "cross_agent"]
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    reason: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    provenance: str = Field(..., min_length=1)


class AssetMemo(StrictSchema):
    """Source-backed draft asset memo requiring human review."""

    memo_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    sections: tuple[str, ...] = Field(default_factory=tuple)
    source_backed_claims: tuple[str, ...] = Field(default_factory=tuple)
    assumptions_summary: tuple[str, ...] = Field(default_factory=tuple)
    missing_evidence: tuple[str, ...] = Field(default_factory=tuple)
    review_questions: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    requires_human_review: bool = True
    confidence: float = Field(default=0.0, ge=0, le=1)


class DueDiligenceManagerPlan(StrictSchema):
    """Manager-agent plan for coordinating Agent 4 diligence synthesis."""

    output_id: str = Field(..., min_length=1)
    nct_id: str = Field(..., min_length=1)
    ordered_steps: tuple[str, ...] = Field(default_factory=tuple)
    guardrail_summary: str = Field(..., min_length=1)
    rationale_summary: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class DueDiligenceSynthesisOutput(StrictSchema):
    """Typed output from an Agent 4 synthesis or critic subagent."""

    output_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    section: Literal[
        "clinical_evidence",
        "competitive_landscape",
        "safety",
        "ip_loe",
        "commercial_assumptions",
        "red_team",
    ]
    synthesis: str = Field(..., min_length=1)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    review_questions: tuple[str, ...] = Field(default_factory=tuple)
    red_flags: tuple[DiligenceRedFlag, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class DueDiligenceOutput(StrictSchema):
    """Structured PharmaOS due-diligence workflow output."""

    output_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    input: DueDiligenceInput
    target_trial: ClinicalTrialRecord
    trial: ClinicalTrialRecord
    asset_identity: AssetIdentityOutput
    agent3_handoff: Agent3HandoffReference
    clinical_risk_summary: ClinicalRiskSummary
    clinical_evidence: ClinicalEvidenceSummary
    competitive_landscape: CompetitiveLandscapeSummary
    safety_label_summary: SafetyLabelSummary
    patent_loe_review: PatentLOEReview
    patent_exclusivity: PatentExclusivityOutput
    pos: PoSOutput
    pricing: PricingOutput
    commercial_model: CommercialModelOutput
    rnpv: RNPVOutput
    red_flags: tuple[DiligenceRedFlag, ...] = Field(default_factory=tuple)
    asset_memo: AssetMemo
    sources: tuple[SourceMetadata, ...] = Field(default_factory=tuple)
    claims: tuple[EvidenceClaim, ...] = Field(default_factory=tuple)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    validation_results: tuple[ValidationResult, ...] = Field(default_factory=tuple)
    confidence_flags: tuple[ConfidenceFlag, ...] = Field(default_factory=tuple)
    human_gate: HumanGate | None = None
    human_readable_summary: HumanReadableModuleOutput | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    validation_status: ValidationStatus = "not_run"
    execution_mode_summary: ExecutionModeSummary = Field(default_factory=ExecutionModeSummary)


class Agent4HandoffReference(StrictSchema):
    """Reference to an Agent 4 output consumed by Agent 5."""

    agent4_run_id: str = Field(..., min_length=1)
    agent4_output_id: str = Field(..., min_length=1)
    nct_id: str = Field(..., min_length=1)
    generated_or_reused: Literal["generated", "reused"]
    retrieved_from_memory: bool
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.0, ge=0, le=1)


class CTGovSearchQuery(StrictSchema):
    """One deterministic ClinicalTrials.gov analog-search query."""

    query_id: str = Field(..., min_length=1)
    condition: str = Field(..., min_length=1)
    intervention: str | None = None
    phase: str | None = None
    target_or_moa: str | None = None
    endpoint_family: str | None = None
    comparator: str | None = None
    biomarker_or_line: str | None = None
    term: str | None = None
    limit: int = Field(default=25, ge=1, le=100)
    expected_analog_dimension: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)


class AnalogSearchPlanOutput(StrictSchema):
    """Structured search-strategy subagent output for CT.gov retrieval."""

    output_id: str = Field(..., min_length=1)
    target_nct_id: str = Field(..., min_length=1)
    queries: tuple[CTGovSearchQuery, ...] = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    expected_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.6, ge=0, le=1)


class NextStudyIntent(StrictSchema):
    """Agent 5 development strategy intent for the logical next study."""

    evidence_anchor_nct_id: str = Field(..., min_length=1)
    current_development_stage: str = Field(..., min_length=1)
    proposed_next_stage: str = Field(..., min_length=1)
    study_role: str = Field(..., min_length=1)
    development_objective: str = Field(..., min_length=1)
    key_clinical_question: str = Field(..., min_length=1)
    indication: str = Field(..., min_length=1)
    target_population_context: str = Field(..., min_length=1)
    regimen_context: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    alternatives_considered: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)
    requires_human_review: bool = True


class AnalogCandidateRecord(StrictSchema):
    """A normalized analog candidate with query provenance."""

    candidate_id: str = Field(..., min_length=1)
    trial: ClinicalTrialRecord
    query_ids: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    provenance: str = Field(..., min_length=1)


class SelectedAnalogTrial(StrictSchema):
    """Selected analog trial and matching rationale."""

    nct_id: str = Field(..., min_length=1)
    match_score: float = Field(..., ge=0.0, le=1.0)
    match_confidence: Literal["high", "medium", "low"]
    matched_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    mismatched_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    unknown_dimensions: tuple[str, ...] = Field(default_factory=tuple)
    reasoning: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class ExcludedAnalogTrial(StrictSchema):
    """Candidate analog excluded from benchmarking."""

    nct_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class AnalogTrialSelectionOutput(StrictSchema):
    """Analog-selection subagent output."""

    output_id: str = Field(..., min_length=1)
    target_nct_id: str = Field(..., min_length=1)
    selected_analogs: tuple[SelectedAnalogTrial, ...] = Field(default_factory=tuple)
    excluded_candidates: tuple[ExcludedAnalogTrial, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class BenchmarkNumericSummary(StrictSchema):
    """Summary statistics for an analog numeric field."""

    observed_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    iqr: float | None = None
    unit: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class BenchmarkFrequency(StrictSchema):
    """Frequency count for an analog benchmark category."""

    label: str = Field(..., min_length=1)
    count: int = Field(..., ge=0)
    frequency: float = Field(..., ge=0, le=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)


class AnalogBenchmarkBundle(StrictSchema):
    """First-class analog trial benchmark artifact for Agent 5."""

    bundle_id: str = Field(..., min_length=1)
    target_nct_id: str = Field(..., min_length=1)
    selected_analog_ids: tuple[str, ...] = Field(default_factory=tuple)
    excluded_analog_ids: tuple[str, ...] = Field(default_factory=tuple)
    search_plan: AnalogSearchPlanOutput
    selection: AnalogTrialSelectionOutput
    enrollment: BenchmarkNumericSummary
    planned_duration_months: BenchmarkNumericSummary
    randomized_frequency: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    blinding_frequency: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    arm_count_distribution: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    primary_endpoint_family_frequency: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    secondary_endpoint_family_frequency: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    comparator_categories: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    named_comparators: tuple[str, ...] = Field(default_factory=tuple)
    inclusion_themes: tuple[str, ...] = Field(default_factory=tuple)
    exclusion_themes: tuple[str, ...] = Field(default_factory=tuple)
    biomarker_testing_themes: tuple[str, ...] = Field(default_factory=tuple)
    prior_treatment_themes: tuple[str, ...] = Field(default_factory=tuple)
    safety_exclusion_themes: tuple[str, ...] = Field(default_factory=tuple)
    country_distribution: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    site_count: BenchmarkNumericSummary
    status_distribution: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    results_availability: tuple[BenchmarkFrequency, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ProtocolSectionDraft(StrictSchema):
    """One draft ProtocolDesignBrief section."""

    section_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ProtocolReviewerCritique(StrictSchema):
    """Regulatory/statistical reviewer critique without approval logic."""

    critique_id: str = Field(..., min_length=1)
    missing_elements: tuple[str, ...] = Field(default_factory=tuple)
    statistical_questions: tuple[str, ...] = Field(default_factory=tuple)
    regulatory_questions: tuple[str, ...] = Field(default_factory=tuple)
    limitations: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ProtocolDesignManagerPlan(StrictSchema):
    """Manager-agent plan for coordinating Agent 5 subagents."""

    output_id: str = Field(..., min_length=1)
    target_nct_id: str = Field(..., min_length=1)
    ordered_steps: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    guardrail_summary: str = Field(..., min_length=1)
    rationale_summary: str = Field(..., min_length=1)
    confidence: float = Field(default=0.5, ge=0, le=1)


class BenchmarkInterpretation(StrictSchema):
    """Agent interpretation of deterministic analog benchmark findings."""

    output_id: str = Field(..., min_length=1)
    target_nct_id: str = Field(..., min_length=1)
    common_design_patterns: tuple[str, ...] = Field(default_factory=tuple)
    target_alignment: tuple[str, ...] = Field(default_factory=tuple)
    target_misalignment: tuple[str, ...] = Field(default_factory=tuple)
    strategy_implications: tuple[str, ...] = Field(default_factory=tuple)
    weak_or_incomplete_findings: tuple[str, ...] = Field(default_factory=tuple)
    human_review_questions: tuple[str, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ProtocolSectionAgentOutput(StrictSchema):
    """Typed output from a section-specific protocol strategy subagent."""

    output_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    sections: tuple[ProtocolSectionDraft, ...] = Field(default_factory=tuple)
    human_review_questions: tuple[str, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ProtocolDesignBrief(StrictSchema):
    """Source-grounded draft protocol design strategy artifact."""

    brief_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    artifact_type: Literal["draft_protocol_design_brief"] = "draft_protocol_design_brief"
    requires_human_review: bool = True
    next_study_intent: NextStudyIntent
    executive_synopsis: ProtocolSectionDraft
    strategic_rationale: ProtocolSectionDraft
    analog_trial_benchmark_summary: ProtocolSectionDraft
    target_population: ProtocolSectionDraft
    study_design: ProtocolSectionDraft
    comparator_and_landscape_rationale: ProtocolSectionDraft
    endpoint_strategy: ProtocolSectionDraft
    draft_eligibility_framework: ProtocolSectionDraft
    draft_schedule_of_assessments_framework: ProtocolSectionDraft
    safety_monitoring_outline: ProtocolSectionDraft
    statistical_analysis_skeleton: ProtocolSectionDraft
    operational_feasibility_risks: ProtocolSectionDraft
    regulatory_standards_considerations: ProtocolSectionDraft
    human_review_questions: tuple[str, ...] = Field(default_factory=tuple)
    source_backed_claim_ids: tuple[str, ...] = Field(default_factory=tuple)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    reviewer_critique: ProtocolReviewerCritique
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float = Field(default=0.5, ge=0, le=1)


class ProtocolDesignOutput(StrictSchema):
    """Structured Agent 5 Protocol Design workflow output."""

    output_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    input: ProtocolDesignInput
    target_trial: ClinicalTrialRecord
    agent3_handoff: Agent3HandoffReference
    agent4_handoff: Agent4HandoffReference
    next_study_intent: NextStudyIntent
    analog_candidates: tuple[AnalogCandidateRecord, ...] = Field(default_factory=tuple)
    analog_benchmark_bundle: AnalogBenchmarkBundle
    protocol_design_brief: ProtocolDesignBrief
    sources: tuple[SourceMetadata, ...] = Field(default_factory=tuple)
    claims: tuple[EvidenceClaim, ...] = Field(default_factory=tuple)
    assumptions: tuple[AssumptionRecord, ...] = Field(default_factory=tuple)
    missing_data_flags: tuple[MissingDataFlag, ...] = Field(default_factory=tuple)
    validation_results: tuple[ValidationResult, ...] = Field(default_factory=tuple)
    confidence_flags: tuple[ConfidenceFlag, ...] = Field(default_factory=tuple)
    human_gate: HumanGate | None = None
    human_readable_summary: HumanReadableModuleOutput | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    validation_status: ValidationStatus = "not_run"
    execution_mode_summary: ExecutionModeSummary = Field(default_factory=ExecutionModeSummary)


class AgentToolCallTrace(StrictSchema):
    """Safe, user-readable trace for one agent tool call."""

    run_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    input_summary: str | None = None
    output_summary: str | None = None
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    provenance: str = Field(..., min_length=1)
    execution_mode: ExecutionMode = "live_agent"


class AgentStepTrace(StrictSchema):
    """Safe, user-readable trace for one agent step."""

    run_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    step_id: str = Field(..., min_length=1)
    input_summary: str | None = None
    output_summary: str | None = None
    tool_calls: tuple[AgentToolCallTrace, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    provenance: str = Field(..., min_length=1)
    execution_mode: ExecutionMode = "deterministic_fallback"


class AgentRunTrace(StrictSchema):
    """Safe, reportable trace for an agent run without hidden reasoning."""

    trace_id: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    agent_name: str = Field(..., min_length=1)
    input_summary: str | None = None
    output_id: str | None = None
    output_type: str | None = None
    output_summary: str | None = None
    steps: tuple[AgentStepTrace, ...] = Field(default_factory=tuple)
    tool_calls: tuple[AgentToolCallTrace, ...] = Field(default_factory=tuple)
    source_ids: tuple[str, ...] = Field(default_factory=tuple)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale_summary: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    provenance: str = Field(..., min_length=1)
    execution_mode: ExecutionMode = "deterministic_fallback"
    model: str | None = None
    model_route: str | None = None
    retry_count: int = Field(default=0, ge=0)
    retry_attempts: int = Field(default=1, ge=1)
    retry_exhausted: bool = False
    fallback_cause: str | None = None
    final_retry_reason: str | None = None


class OrchestrationRunRecord(StrictSchema):
    """Persistable planning or orchestration run envelope for the Control Tower."""

    run_id: str = Field(..., min_length=1)
    request: OrchestrationRequest
    snapshot: ScientificStateSnapshot
    plan: ExecutionPlan
    snapshots: tuple[ScientificStateSnapshot, ...] = Field(default_factory=tuple)
    final_snapshot: ScientificStateSnapshot | None = None
    plans: tuple[ExecutionPlan, ...] = Field(default_factory=tuple)
    step_results: tuple[OrchestrationStepResult, ...] = Field(default_factory=tuple)
    replans: tuple[OrchestrationReplanRecord, ...] = Field(default_factory=tuple)
    child_run_ids: tuple[str, ...] = Field(default_factory=tuple)
    report: ControlTowerReport | None = None
    validation_results: tuple[ValidationResult, ...] = Field(default_factory=tuple)
    trace: AgentRunTrace | None = None
    execution_mode_summary: ExecutionModeSummary = Field(default_factory=ExecutionModeSummary)


class AgentOutput(StrictSchema):
    """Common output contract for agents before workflow-specific schemas."""

    output_id: str = Field(..., min_length=1, description="Stable output identifier.")
    agent_name: str = Field(..., min_length=1, description="Agent that produced the output.")
    run_id: str = Field(..., min_length=1, description="Workflow run that produced this output.")
    provenance: str = Field(..., min_length=1, description="Prompt, tool, model, or step provenance.")
    claims: tuple[EvidenceClaim, ...] = Field(default_factory=tuple, description="Evidence-backed claims in this output.")
    sources: tuple[SourceMetadata, ...] = Field(default_factory=tuple, description="Sources cited by this output.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall output confidence.")
    validation_status: ValidationStatus = Field(default="not_run", description="Output validation status.")
    gate_reason: str | None = Field(default=None, description="Reason this output needs human review.")
    execution_mode: ExecutionMode = Field(default="deterministic_fallback", description="How this output's reasoning was executed.")
    execution_mode_summary: ExecutionModeSummary = Field(default_factory=ExecutionModeSummary, description="Visible AI execution mode counts for this output.")


class FinalReport(StrictSchema):
    """Final report assembled from workflow and agent outputs."""

    report_id: str = Field(..., min_length=1, description="Stable final-report identifier.")
    run_id: str = Field(..., min_length=1, description="Workflow run summarized by this report.")
    title: str = Field(..., min_length=1, description="Report title.")
    summary: str = Field(..., min_length=1, description="Executive summary.")
    claims: tuple[EvidenceClaim, ...] = Field(default_factory=tuple, description="Report claims.")
    sources: tuple[SourceMetadata, ...] = Field(default_factory=tuple, description="Report sources.")
    validation_results: tuple[ValidationResult, ...] = Field(default_factory=tuple, description="Report validation results.")
    confidence_flags: tuple[ConfidenceFlag, ...] = Field(default_factory=tuple, description="Confidence flags affecting the report.")
    human_gate: HumanGate | None = Field(default=None, description="Human gate required for report release, if any.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall report confidence.")
    validation_status: ValidationStatus = Field(default="not_run", description="Aggregate report validation status.")
    provenance: str = Field(..., min_length=1, description="Assembly and review provenance.")
    execution_mode_summary: ExecutionModeSummary = Field(default_factory=ExecutionModeSummary, description="Visible AI execution mode counts for this report.")


class NotImplementedOutput(StrictSchema):
    """Explicit placeholder for workflows that are intentionally not implemented yet."""

    output_id: str = Field(..., min_length=1, description="Stable placeholder output identifier.")
    workflow_name: str = Field(..., min_length=1, description="Workflow that is not implemented.")
    reason: str = Field(..., min_length=1, description="Why the workflow is not implemented.")
    gate_reason: str = Field(..., min_length=1, description="Reason execution must stop or route to humans.")
    validation_status: ValidationStatus = Field(default="needs_human_review", description="Placeholder validation status.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Placeholder confidence.")
    source_ids: tuple[str, ...] = Field(default_factory=tuple, description="Sources relevant to the missing implementation.")
    provenance: str = Field(..., min_length=1, description="Workflow step that emitted this placeholder.")
