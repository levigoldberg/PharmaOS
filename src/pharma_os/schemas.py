"""Shared strict Pydantic schemas for PharmaOS workflows.

These models are intentionally workflow-agnostic.  Workflow-specific schemas should
compose or extend these primitives rather than redefining provenance, evidence,
confidence, validation, or human-gate fields.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


ConfidenceLevel = Literal["very_low", "low", "medium", "high", "very_high"]
ValidationStatus = Literal["not_run", "passed", "failed", "warning", "needs_human_review"]
GateDecision = Literal["approved", "rejected", "needs_human_review", "blocked"]
WorkflowStatus = Literal["pending", "running", "completed", "failed", "blocked"]
MetadataValue = str | int | float | bool | None


class StrictSchema(BaseModel):
    """Base class for strict workflow contracts."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        populate_by_name=True,
    )


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
