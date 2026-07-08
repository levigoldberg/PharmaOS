"""Validation and confidence helpers for PharmaOS workflows."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from pharma_os.schemas import (
    ClinicalOutcomePredictionOutput,
    ConfidenceFlag,
    DueDiligenceOutput,
    EvidenceClaim,
    HumanGate,
    ProtocolDesignOutput,
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


def validate_cross_agent_consistency(
    *,
    run_id: str,
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
) -> tuple[ValidationResult, ...]:
    """Validate typed consistency between Agent 3 and downstream workflow outputs."""

    checks: list[tuple[str, bool, str, tuple[str, ...]]] = []
    agent3_sources = {source.source_id for source in agent3_output.sources}
    agent4_sources = {
        agent4_output.trial.source_id,
        *agent4_output.asset_identity.source_ids,
        *agent4_output.pos.source_ids,
        *agent4_output.pricing.source_ids,
        *agent4_output.patent_exclusivity.source_ids,
        *agent4_output.commercial_model.source_ids,
        *agent4_output.rnpv.source_ids,
    }
    shared_sources = tuple(sorted(agent3_sources & agent4_sources))
    source_context = tuple(sorted(agent3_sources | agent4_sources))

    checks.append(
        (
            "nct_id",
            _same(agent3_output.trial_identity.nct_id, agent4_output.trial.nct_id),
            f"Agent 3 NCT ID {agent3_output.trial_identity.nct_id} matches Agent 4 NCT ID {agent4_output.trial.nct_id}.",
            tuple(dict.fromkeys((*agent3_output.trial_identity.source_ids, agent4_output.trial.source_id))),
        )
    )
    checks.append(
        (
            "asset_name",
            _same(agent3_output.asset_identity.asset_name, agent4_output.asset_identity.asset_name),
            f"Agent 3 asset {agent3_output.asset_identity.asset_name or 'unknown'} matches Agent 4 asset {agent4_output.asset_identity.asset_name or 'unknown'}.",
            tuple(dict.fromkeys((*agent3_output.asset_identity.source_ids, *agent4_output.asset_identity.source_ids))),
        )
    )
    checks.append(
        (
            "indication",
            _same(agent3_output.asset_identity.normalized_indication, agent4_output.asset_identity.normalized_indication)
            or bool(set(_norm_items(agent3_output.trial_identity.conditions)) & set(_norm_items(agent4_output.trial.conditions))),
            f"Agent 3 indication {agent3_output.asset_identity.normalized_indication or ', '.join(agent3_output.trial_identity.conditions) or 'unknown'} is consistent with Agent 4 indication {agent4_output.asset_identity.normalized_indication or ', '.join(agent4_output.trial.conditions) or 'unknown'}.",
            tuple(dict.fromkeys((*agent3_output.trial_identity.source_ids, *agent4_output.asset_identity.source_ids))),
        )
    )
    checks.append(
        (
            "phase",
            bool(
                set(_norm_phases(agent3_output.trial_identity.phases or (agent3_output.historical_pos_estimate.current_phase,)))
                & set(_norm_phases(agent4_output.trial.phases or (agent4_output.pos.current_phase,)))
            ),
            f"Agent 3 phases {', '.join(agent3_output.trial_identity.phases) or 'unknown'} are consistent with Agent 4 phases {', '.join(agent4_output.trial.phases) or agent4_output.pos.current_phase or 'unknown'}.",
            tuple(dict.fromkeys((*agent3_output.trial_identity.source_ids, agent4_output.trial.source_id))),
        )
    )
    checks.append(
        (
            "sponsor",
            _same(agent3_output.trial_identity.sponsor, agent4_output.asset_identity.sponsor),
            f"Agent 3 sponsor {agent3_output.trial_identity.sponsor or 'unknown'} matches Agent 4 sponsor {agent4_output.asset_identity.sponsor or 'unknown'}.",
            tuple(dict.fromkeys((*agent3_output.trial_identity.source_ids, *agent4_output.asset_identity.source_ids))),
        )
    )
    checks.append(
        (
            "endpoint_summary",
            agent3_output.trial_design_features.primary_endpoint_count == len(agent4_output.trial.primary_endpoints),
            f"Agent 3 primary endpoint count {agent3_output.trial_design_features.primary_endpoint_count} matches Agent 4 CT.gov endpoint count {len(agent4_output.trial.primary_endpoints)}.",
            tuple(dict.fromkeys((*agent3_output.trial_design_features.source_ids, agent4_output.trial.source_id))),
        )
    )
    checks.append(
        (
            "safety_context",
            _sources_consistent(agent3_output.safety_context.source_ids, agent4_output.pricing.source_ids, "openfda_label"),
            "Agent 3 safety context and Agent 4 pricing use consistent openFDA label source IDs, or one side has no openFDA label source.",
            tuple(dict.fromkeys((*agent3_output.safety_context.source_ids, *agent4_output.pricing.source_ids))),
        )
    )
    checks.append(
        (
            "pos_basis",
            _same(agent3_output.historical_pos_estimate.probability_of_success, agent4_output.pos.probability_of_success)
            and _same(agent3_output.historical_pos_estimate.lookup_key, agent4_output.pos.lookup_key),
            f"Agent 3 PoS basis {agent3_output.historical_pos_estimate.lookup_key or 'unknown'} matches Agent 4 PoS basis {agent4_output.pos.lookup_key or 'unknown'}.",
            tuple(dict.fromkeys((*agent3_output.historical_pos_estimate.source_ids, *agent4_output.pos.source_ids))),
        )
    )
    checks.append(
        (
            "source_ids",
            bool(shared_sources),
            "Agent 3 and Agent 4 share at least one persisted source ID.",
            shared_sources or source_context,
        )
    )

    results: list[ValidationResult] = []
    for field, passed, message, source_ids in checks:
        validation_id = f"validation-{run_id}-cross-agent-{_slug(field)}"
        if passed:
            results.append(
                ValidationResult(
                    validation_id=validation_id,
                    target_id=agent4_output.output_id,
                    status="passed",
                    validator="cross_agent_consistency",
                    message=message,
                    confidence=0.9,
                    source_ids=source_ids,
                    provenance="pharma_os.validators.validate_cross_agent_consistency",
                )
            )
        else:
            results.append(
                ValidationResult(
                    validation_id=validation_id,
                    target_id=agent4_output.output_id,
                    status="failed",
                    validator="cross_agent_consistency",
                    message=f"Cross-agent {field} mismatch. {message} Discrepancy is not silently resolved; CT.gov-derived fields are treated as primary for trial identity, but the mismatch requires review.",
                    confidence=0.95,
                    source_ids=source_ids,
                    gate_reason=f"Agent 3 and Agent 4 disagree on {field}.",
                    provenance="pharma_os.validators.validate_cross_agent_consistency",
                )
            )
    return tuple(results)


def validate_protocol_design_constraints(
    *,
    run_id: str,
    output: ProtocolDesignOutput,
) -> tuple[ValidationResult, ...]:
    """Validate Agent 5 draft-only and source-boundary constraints."""

    output_text = output.model_dump_json().casefold()
    source_types = {source.source_type for source in output.sources if source.source_type}
    allowed_source_types = {
        "clinical_trial_registry",
        "clinical_trial_registry_search",
        "literature",
        "drug_label",
        "fixture",
        "human_input",
        "configuration",
        "agent_output",
        "pos_workbook",
        "wac_workbook",
    }
    disallowed_patterns = {
        "ehr": r"\behr\b|electronic health record",
        "omop": r"\bomop\b",
        "fhir": r"\bfhir\b",
        "patient_matching": r"patient matching|match patients|patient recruitment list",
        "aact": r"\baact\b",
        "trialtrove": r"trialtrove",
        "globaldata": r"globaldata",
        "sec": r"\bsec filing\b|\b10-k\b|\b8-k\b",
        "ema_epar": r"\bema epar\b|\bepar\b",
        "orange_book": r"orange book",
        "drugbank": r"drugbank",
        "proprietary_data": r"proprietary data",
        "fake_patient_site_enrollment": r"fake patient|fake site|invented enrollment",
        "approval_logic": r"final approval|approve the protocol|protocol approved|submission-ready|irb-ready",
    }
    findings: list[str] = []
    for name, pattern in disallowed_patterns.items():
        if re.search(pattern, output_text, re.IGNORECASE):
            findings.append(name)
    invalid_source_types = sorted(source_type for source_type in source_types if source_type not in allowed_source_types)

    results: list[ValidationResult] = []
    if findings or invalid_source_types:
        message_parts = []
        if findings:
            message_parts.append(f"disallowed source/scope terms detected: {', '.join(findings)}")
        if invalid_source_types:
            message_parts.append(f"unsupported source types detected: {', '.join(invalid_source_types)}")
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-protocol-design-source-boundary",
                target_id=output.output_id,
                status="failed",
                validator="protocol_design_source_boundary",
                message="; ".join(message_parts),
                confidence=1.0,
                source_ids=tuple(source.source_id for source in output.sources),
                gate_reason="Agent 5 must stay inside the approved source and draft-strategy scope.",
                provenance="pharma_os.validators.validate_protocol_design_constraints",
            )
        )
    else:
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-protocol-design-source-boundary",
                target_id=output.output_id,
                status="passed",
                validator="protocol_design_source_boundary",
                message="Agent 5 output stayed within approved source and draft-strategy boundaries.",
                confidence=1.0,
                source_ids=tuple(source.source_id for source in output.sources),
                provenance="pharma_os.validators.validate_protocol_design_constraints",
            )
        )

    brief = output.protocol_design_brief
    if not brief.requires_human_review or brief.artifact_type != "draft_protocol_design_brief" or output.human_gate is None:
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-protocol-design-draft-gate",
                target_id=output.output_id,
                status="failed",
                validator="protocol_design_draft_gate",
                message="ProtocolDesignBrief must be marked as a draft strategy artifact and carry an open human-review gate.",
                confidence=1.0,
                gate_reason="Protocol design artifacts require human clinical, statistical, and regulatory review.",
                provenance="pharma_os.validators.validate_protocol_design_constraints",
            )
        )
    else:
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-protocol-design-draft-gate",
                target_id=output.output_id,
                status="passed",
                validator="protocol_design_draft_gate",
                message="ProtocolDesignBrief is marked as a draft strategy artifact requiring human review.",
                confidence=1.0,
                source_ids=brief.source_ids,
                provenance="pharma_os.validators.validate_protocol_design_constraints",
            )
        )

    if not output.analog_benchmark_bundle.selected_analog_ids:
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-protocol-design-analog-benchmark",
                target_id=output.output_id,
                status="warning",
                validator="protocol_design_analog_benchmark",
                message="No selected analog trials were available for benchmark calculations.",
                confidence=0.9,
                source_ids=output.analog_benchmark_bundle.source_ids,
                gate_reason="Analog trial benchmarking is the backbone of Agent 5 and requires review when empty.",
                provenance="pharma_os.validators.validate_protocol_design_constraints",
            )
        )
    else:
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-protocol-design-analog-benchmark",
                target_id=output.output_id,
                status="passed",
                validator="protocol_design_analog_benchmark",
                message="Analog benchmark bundle includes selected analog trials and query provenance.",
                confidence=0.95,
                source_ids=output.analog_benchmark_bundle.source_ids,
                provenance="pharma_os.validators.validate_protocol_design_constraints",
            )
        )
    return tuple(results)


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
        flag_severity = "critical" if severity == "critical" else "high" if severity == "high" else "medium"
        flags.append(
            ConfidenceFlag(
                flag_id=f"flag-{run_id}-risk-{index}",
                target_id=getattr(risk, "risk_id", None) or getattr(risk, "flag_id", f"risk-{index}"),
                reason=description,
                severity=flag_severity,
                confidence=0.4 if severity == "critical" else 0.65 if severity == "high" else 0.8,
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


def _same(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, float) or isinstance(right, float):
        try:
            return abs(float(left) - float(right)) < 0.000001
        except (TypeError, ValueError):
            return False
    return str(left).strip().casefold() == str(right).strip().casefold()


def _norm_items(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item.strip().casefold().replace(" ", "") for item in values if item)


def _norm_phases(values: tuple[str | None, ...]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        text = str(value or "").strip().casefold().replace(" ", "").replace("_", "")
        text = text.replace("phaseiii", "phase3").replace("phaseii", "phase2").replace("phasei", "phase1")
        if text:
            normalized.append(text)
    return tuple(normalized)


def _sources_consistent(left: tuple[str, ...], right: tuple[str, ...], prefix: str) -> bool:
    left_matching = {source_id for source_id in left if source_id.startswith(prefix)}
    right_matching = {source_id for source_id in right if source_id.startswith(prefix)}
    if not left_matching or not right_matching:
        return True
    return bool(left_matching & right_matching)
