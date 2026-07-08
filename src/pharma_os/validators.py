"""Validation and confidence helpers for PharmaOS workflows."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from pharma_os.schemas import (
    ConfidenceFlag,
    EvidenceClaim,
    HumanGate,
    ValidationResult,
)


HIGH_RISK_RE = re.compile(
    r"\b("
    r"go\s*/?\s*no\s*-?\s*go|target nomination|nominate(?:d)? target|"
    r"protocol approval|approve(?:d)? protocol|safety decision|"
    r"investment recommendation|recommend(?:ed)? investment|rnpv|"
    r"acquire|license|dose escalation|first[- ]in[- ]human"
    r")\b",
    re.IGNORECASE,
)
NUMERIC_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?%?|\b(?:one|two|three|four|five)\b)", re.I)
HIGH_RISK_NUMERIC_RE = re.compile(
    r"\b(enrollment|enrolled|count|rate|date|phase|results?|patients?|subjects?|nct\d{8})\b",
    re.IGNORECASE,
)


def validate_workflow_name(workflow: str) -> str:
    """Validate and normalize a workflow name."""

    normalized = workflow.strip()
    if not normalized:
        raise ValueError("workflow name must not be empty")
    return normalized


def validate_schema(
    *,
    target_id: str,
    payload: Any,
    schema_type: type[BaseModel],
    run_id: str,
) -> ValidationResult:
    """Validate a payload against a Pydantic schema."""

    validation_id = f"validation-{run_id}-schema-{_slug(target_id)}"
    try:
        if isinstance(payload, schema_type):
            schema_type.model_validate(payload.model_dump())
        else:
            schema_type.model_validate(payload)
    except ValidationError as exc:
        return ValidationResult(
            validation_id=validation_id,
            target_id=target_id,
            status="failed",
            validator="schema_validation",
            message=f"{schema_type.__name__} validation failed: {exc.errors()[0]['msg']}",
            confidence=1.0,
            gate_reason="Schema validation failed.",
            provenance="pharma_os.validators.validate_schema",
        )
    return ValidationResult(
        validation_id=validation_id,
        target_id=target_id,
        status="passed",
        validator="schema_validation",
        message=f"{schema_type.__name__} validation passed.",
        confidence=1.0,
        provenance="pharma_os.validators.validate_schema",
    )


def validate_source_coverage(
    *,
    target_id: str,
    claims: tuple[EvidenceClaim, ...],
    source_ids: set[str],
    run_id: str,
) -> ValidationResult:
    """Validate that every factual claim has known source IDs."""

    missing = [claim.claim_id for claim in claims if not claim.source_ids]
    unknown = [
        claim.claim_id
        for claim in claims
        if claim.source_ids and any(source_id not in source_ids for source_id in claim.source_ids)
    ]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"claims without source_ids: {', '.join(missing)}")
        if unknown:
            parts.append(f"claims with unknown source_ids: {', '.join(unknown)}")
        return ValidationResult(
            validation_id=f"validation-{run_id}-source-coverage-{_slug(target_id)}",
            target_id=target_id,
            status="failed",
            validator="source_coverage",
            message="; ".join(parts),
            confidence=1.0,
            gate_reason="Every factual claim must cite a persisted source.",
            provenance="pharma_os.validators.validate_source_coverage",
        )
    return ValidationResult(
        validation_id=f"validation-{run_id}-source-coverage-{_slug(target_id)}",
        target_id=target_id,
        status="passed",
        validator="source_coverage",
        message="Every claim cites at least one known source.",
        confidence=1.0,
        source_ids=tuple(sorted(source_ids)),
        provenance="pharma_os.validators.validate_source_coverage",
    )


def validate_numeric_provenance(
    *,
    target_id: str,
    claims: tuple[EvidenceClaim, ...],
    run_id: str,
) -> ValidationResult:
    """Check that numeric-looking claims have source provenance."""

    risky_missing = []
    numeric_missing = []
    for claim in claims:
        if not NUMERIC_RE.search(claim.claim_text):
            continue
        if claim.source_ids:
            continue
        if HIGH_RISK_NUMERIC_RE.search(claim.claim_text):
            risky_missing.append(claim.claim_id)
        else:
            numeric_missing.append(claim.claim_id)
    if risky_missing:
        return ValidationResult(
            validation_id=f"validation-{run_id}-numeric-provenance-{_slug(target_id)}",
            target_id=target_id,
            status="failed",
            validator="numeric_provenance",
            message=f"High-risk numeric claims lack source IDs: {', '.join(risky_missing)}",
            confidence=1.0,
            gate_reason="Numeric clinical trial claims require explicit source provenance.",
            provenance="pharma_os.validators.validate_numeric_provenance",
        )
    if numeric_missing:
        return ValidationResult(
            validation_id=f"validation-{run_id}-numeric-provenance-{_slug(target_id)}",
            target_id=target_id,
            status="warning",
            validator="numeric_provenance",
            message=f"Numeric-looking claims lack source IDs: {', '.join(numeric_missing)}",
            confidence=0.9,
            provenance="pharma_os.validators.validate_numeric_provenance",
        )
    return ValidationResult(
        validation_id=f"validation-{run_id}-numeric-provenance-{_slug(target_id)}",
        target_id=target_id,
        status="passed",
        validator="numeric_provenance",
        message="No unsupported numeric claims detected.",
        confidence=1.0,
        provenance="pharma_os.validators.validate_numeric_provenance",
    )


def assign_human_gate(
    *,
    run_id: str,
    workflow_name: str,
    validation_results: tuple[ValidationResult, ...],
    output_text: str,
) -> HumanGate | None:
    """Create a human gate if validation failed or high-risk language appears."""

    failed = [result for result in validation_results if result.status == "failed"]
    high_risk_match = HIGH_RISK_RE.search(output_text)
    if not failed and not high_risk_match:
        return None
    reasons = []
    if failed:
        reasons.append("one or more validators failed")
    if high_risk_match:
        reasons.append(f"high-risk recommendation language detected: {high_risk_match.group(0)}")
    return HumanGate(
        gate_id=f"gate-{run_id}",
        decision="needs_human_review",
        gate_reason=f"{workflow_name} requires human review because " + " and ".join(reasons) + ".",
        required_roles=("clinical_lead", "regulatory_reviewer"),
        source_ids=tuple(
            sorted({source_id for result in validation_results for source_id in result.source_ids})
        ),
        provenance="pharma_os.validators.assign_human_gate",
    )


def generate_confidence_flags(
    *,
    run_id: str,
    validation_results: tuple[ValidationResult, ...],
    risk_flags: tuple[Any, ...] = (),
) -> tuple[ConfidenceFlag, ...]:
    """Generate confidence flags from validation and workflow risk flags."""

    flags: list[ConfidenceFlag] = []
    for result in validation_results:
        if result.status not in {"failed", "warning", "needs_human_review"}:
            continue
        severity = "high" if result.status == "failed" else "medium"
        flags.append(
            ConfidenceFlag(
                flag_id=f"flag-{run_id}-{_slug(result.validation_id)}",
                target_id=result.target_id,
                reason=result.message,
                severity=severity,
                confidence=max(0.0, min(result.confidence, 1.0)),
                source_ids=result.source_ids,
                provenance="pharma_os.validators.generate_confidence_flags.validation",
            )
        )
    for index, risk in enumerate(risk_flags, start=1):
        severity = getattr(risk, "severity", "medium")
        description = getattr(risk, "description", None) or getattr(risk, "reason", str(risk))
        source_ids = tuple(getattr(risk, "source_ids", ()))
        flags.append(
            ConfidenceFlag(
                flag_id=f"flag-{run_id}-risk-{index}",
                target_id=getattr(risk, "risk_id", f"risk-{index}"),
                reason=description,
                severity="high" if severity == "high" else "medium",
                confidence=0.65 if severity == "high" else 0.8,
                source_ids=source_ids,
                provenance="pharma_os.validators.generate_confidence_flags.risk",
            )
        )
    return tuple(flags)


def aggregate_validation_status(results: tuple[ValidationResult, ...]) -> str:
    """Return an aggregate validation status."""

    statuses = {result.status for result in results}
    if "failed" in statuses:
        return "failed"
    if "needs_human_review" in statuses:
        return "needs_human_review"
    if "warning" in statuses:
        return "warning"
    if statuses:
        return "passed"
    return "not_run"


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")[:80] or "target"
