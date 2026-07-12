"""Validation and confidence helpers for PharmaOS workflows."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ValidationError

from pharma_os.review_flags import canonical_review_flags
from pharma_os.schemas import (
    ClinicalOutcomePredictionOutput,
    ConfidenceFlag,
    DueDiligenceOutput,
    EvidenceClaim,
    HumanGate,
    ProtocolDesignOutput,
    ValidationResult,
)
from pharma_os.tools.clinical_semantics import comparable_modality, endpoint_family, same_indication


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


def validate_clinical_outcome_constraints(
    *,
    run_id: str,
    output: ClinicalOutcomePredictionOutput,
) -> tuple[ValidationResult, ...]:
    """Validate Agent 3 clinical-reasoning guardrails."""

    results: list[ValidationResult] = []
    output_text = output.model_dump_json().casefold()
    disallowed_patterns = {
        "final_decision": r"\bgo\s*/?\s*no-go\b|\bgo decision\b|\bno-go decision\b|\bapproval decision\b|\binvestment recommendation\b|\blicensing recommendation\b",
        "invented_efficacy": r"\b(?:efficacy|response|orr|pfs|overall survival|os)\s+(?:rate|result)?\s*(?:was|is|=)?\s*(?:assumed|estimated|set\s+at\s+)?\d",
        "invented_safety_rate": r"\b(?:adverse event|ae|toxicity|serious adverse event|sae|safety)\s+(?:rate)?\s*(?:was|is|=)?\s*(?:assumed|estimated|set\s+at\s+)?\d",
        "invented_pos_or_approval": r"\b(?:pos|probability of success|approval probability|approval likelihood)\s+(?:is|=)\s+(?:assumed|estimated|set)\s+\d",
    }
    findings = [name for name, pattern in disallowed_patterns.items() if re.search(pattern, output_text, re.IGNORECASE)]
    if output.approval_likelihood_proxy.probability is not None and (
        output.approval_likelihood_proxy.assumption_type != "source_derived"
        or not output.approval_likelihood_proxy.source_ids
    ):
        findings.append("approval_proxy_not_source_backed")
    if output.historical_pos_estimate.probability_of_success is not None and (
        output.historical_pos_estimate.assumption_type != "source_derived"
        or not output.historical_pos_estimate.source_ids
    ):
        findings.append("pos_not_source_backed")
    if findings:
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-clinical-outcome-guardrails",
                target_id=output.output_id,
                status="failed",
                validator="clinical_outcome_guardrails",
                message=f"disallowed clinical outcome language or unsupported probability detected: {', '.join(tuple(dict.fromkeys(findings)))}",
                confidence=1.0,
                source_ids=tuple(source.source_id for source in output.sources),
                gate_reason="Agent 3 must remain a source-backed clinical risk artifact, not an outcome oracle or decision engine.",
                provenance="pharma_os.validators.validate_clinical_outcome_constraints",
            )
        )
    else:
        results.append(
            ValidationResult(
                validation_id=f"validation-{run_id}-clinical-outcome-guardrails",
                target_id=output.output_id,
                status="passed",
                validator="clinical_outcome_guardrails",
                message="Agent 3 output stayed within clinical risk reasoning guardrails.",
                confidence=1.0,
                source_ids=tuple(source.source_id for source in output.sources),
                provenance="pharma_os.validators.validate_clinical_outcome_constraints",
            )
        )
    return tuple(results)


def validate_due_diligence_constraints(
    *,
    run_id: str,
    output: DueDiligenceOutput,
) -> tuple[ValidationResult, ...]:
    """Validate Agent 4 draft-only diligence guardrails."""

    output_text = output.model_dump_json().casefold()
    disallowed_patterns = {
        "investment_recommendation": r"\brecommend(?:ed)?\s+(?:investment|investing|license|licensing|acquisition|acquire)\b",
        "go_no_go": r"\bgo\s*/?\s*no-go\b|\bgo decision\b|\bno-go decision\b",
        "approval_decision": r"\bapproval decision\b|\bapprove(?:d)?\s+(?:the\s+)?(?:asset|trial|protocol|investment)\b|\brecommend(?:ed)?\s+approval\b",
        "legal_conclusion": r"\blegal conclusion\b|\bfreedom to operate\b|\bfto opinion\b",
        "invented_diligence_values": r"\b(?:loe|pricing|market size|eligible patients|penetration|rnpv)\s+(?:is|=)\s+(?:assumed|estimated|set)\s+\d",
    }
    findings = [name for name, pattern in disallowed_patterns.items() if re.search(pattern, output_text, re.IGNORECASE)]
    if findings:
        return (
            ValidationResult(
                validation_id=f"validation-{run_id}-due-diligence-guardrails",
                target_id=output.output_id,
                status="failed",
                validator="due_diligence_guardrails",
                message=f"disallowed due-diligence decision or invented-value language detected: {', '.join(findings)}",
                confidence=1.0,
                source_ids=tuple(source.source_id for source in output.sources),
                gate_reason="Agent 4 must remain a draft diligence artifact without final decisions or invented values.",
                provenance="pharma_os.validators.validate_due_diligence_constraints",
            ),
        )
    return (
        ValidationResult(
            validation_id=f"validation-{run_id}-due-diligence-guardrails",
            target_id=output.output_id,
            status="passed",
            validator="due_diligence_guardrails",
            message="Agent 4 output stayed within draft diligence guardrails.",
            confidence=1.0,
            source_ids=tuple(source.source_id for source in output.sources),
            provenance="pharma_os.validators.validate_due_diligence_constraints",
        ),
    )


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
        "approval_logic": r"final approval|approve the protocol|protocol approved|submission-ready|irb-ready|enrollment-ready|go\s*/?\s*no-go|go decision|no-go decision",
        "final_protocol_language": r"\b(?:is|as|approved as)\s+(?:a\s+)?final protocol\b|\b(?:is|as|approved as)\s+(?:the\s+)?final design\b|\brecommended design\b|\brecommended protocol\b",
        "invented_statistical_design": r"\b(sample size|power|effect size|alpha allocation)\s+(?:is|=|of)\s+\d",
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

    results.extend(_protocol_design_semantic_results(run_id=run_id, output=output))
    return tuple(results)


def _protocol_design_semantic_results(*, run_id: str, output: ProtocolDesignOutput) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    results.append(_validate_candidate_disposition(run_id=run_id, output=output))
    results.append(_validate_similarity_consistency(run_id=run_id, output=output))
    results.append(_validate_endpoint_family_consistency(run_id=run_id, output=output))
    results.append(_validate_zero_follow_on_semantics(run_id=run_id, output=output))
    results.append(_validate_missing_flag_contradictions(run_id=run_id, output=output))
    results.append(_validate_fractional_enrollment(run_id=run_id, output=output))
    results.append(_validate_duration_label_semantics(run_id=run_id, output=output))
    results.append(_validate_support_source_consistency(run_id=run_id, output=output))
    return results


def _validate_candidate_disposition(*, run_id: str, output: ProtocolDesignOutput) -> ValidationResult:
    candidate_ids = [candidate.trial.nct_id for candidate in output.analog_candidates]
    selection = output.analog_benchmark_bundle.selection
    dispositions = [
        *(item.nct_id for item in selection.selected_analogs),
        *(item.nct_id for item in selection.excluded_candidates),
        *(item.nct_id for item in selection.unevaluable_candidates),
    ]
    missing = sorted(set(candidate_ids) - set(dispositions))
    duplicate = sorted(nct_id for nct_id in set(dispositions) if dispositions.count(nct_id) > 1)
    unknown = sorted(set(dispositions) - set(candidate_ids))
    status = "failed" if missing or duplicate or unknown else "passed"
    return ValidationResult(
        validation_id=f"validation-{run_id}-protocol-design-candidate-disposition",
        target_id=output.output_id,
        status=status,
        validator="protocol_design_candidate_disposition",
        message=(
            f"Analog candidate disposition gaps: missing={missing}; duplicate={duplicate}; unknown={unknown}."
            if status == "failed"
            else "Every retrieved analog candidate has exactly one selected, excluded, or unevaluable disposition."
        ),
        confidence=1.0,
        gate_reason="Every retrieved analog candidate must have exactly one disposition." if status == "failed" else None,
        provenance="pharma_os.validators.validate_protocol_design_constraints",
    )


def _validate_similarity_consistency(*, run_id: str, output: ProtocolDesignOutput) -> ValidationResult:
    indication_mismatches = []
    modality_mismatches = []
    for candidate in output.analog_candidates:
        features = candidate.similarity_features or {}
        if same_indication(output.target_trial.conditions, candidate.trial.conditions) and features.get("same_indication") is False:
            indication_mismatches.append(candidate.trial.nct_id)
        if comparable_modality(output.target_trial, candidate.trial) is True and features.get("same_modality") is False:
            modality_mismatches.append(candidate.trial.nct_id)
    findings = []
    if indication_mismatches:
        findings.append(f"same-indication false despite normalized match: {', '.join(indication_mismatches[:8])}")
    if modality_mismatches:
        findings.append(f"same-modality false despite normalized active route/modality match: {', '.join(modality_mismatches[:8])}")
    status = "warning" if findings else "passed"
    return ValidationResult(
        validation_id=f"validation-{run_id}-protocol-design-similarity-semantics",
        target_id=output.output_id,
        status=status,
        validator="protocol_design_similarity_semantics",
        message="; ".join(findings) if findings else "Indication and modality similarity features are semantically consistent.",
        confidence=0.95,
        provenance="pharma_os.validators.validate_protocol_design_constraints",
    )


def _validate_endpoint_family_consistency(*, run_id: str, output: ProtocolDesignOutput) -> ValidationResult:
    bad = []
    trials = (output.target_trial, *(candidate.trial for candidate in output.analog_candidates), *output.follow_on_trials)
    for trial in trials:
        for endpoint in (*trial.primary_endpoints, *trial.secondary_endpoints):
            text = " ".join(item for item in (endpoint.measure, endpoint.description, endpoint.time_frame) if item).casefold()
            family = endpoint_family(endpoint.measure, endpoint.description, endpoint.time_frame)
            if any(term in text for term in ("pasi", "pga", "iga", "dlqi")) and family in {"safety", "other"}:
                bad.append(f"{trial.nct_id}:{endpoint.measure}")
    status = "warning" if bad else "passed"
    return ValidationResult(
        validation_id=f"validation-{run_id}-protocol-design-endpoint-family-semantics",
        target_id=output.output_id,
        status=status,
        validator="protocol_design_endpoint_family_semantics",
        message=f"Endpoint-family misclassifications detected: {'; '.join(bad[:8])}" if bad else "Endpoint families are semantically classified for PASI/PGA/IGA/DLQI endpoints.",
        confidence=0.95,
        provenance="pharma_os.validators.validate_protocol_design_constraints",
    )


def _validate_zero_follow_on_semantics(*, run_id: str, output: ProtocolDesignOutput) -> ValidationResult:
    text = " ".join(_string_values(output.protocol_design_brief.model_dump(mode="json"))).casefold()
    decisions = output.analog_derived_design_decisions
    contradiction = not output.follow_on_trials and (
        "observed follow-on" in text
        or "selected follow-on trial" in text
        or any(decision.support_source_type == "follow_on_supported" for decision in decisions)
    )
    return ValidationResult(
        validation_id=f"validation-{run_id}-protocol-design-zero-follow-on-semantics",
        target_id=output.output_id,
        status="failed" if contradiction else "passed",
        validator="protocol_design_zero_follow_on_semantics",
        message=(
            "Output claims follow-on-supported precedent despite zero selected follow-on trials."
            if contradiction
            else "Zero-follow-on outputs do not claim observed follow-on precedent."
        ),
        confidence=1.0,
        gate_reason="Follow-on-supported language requires selected follow-on trials." if contradiction else None,
        provenance="pharma_os.validators.validate_protocol_design_constraints",
    )


def _validate_missing_flag_contradictions(*, run_id: str, output: ProtocolDesignOutput) -> ValidationResult:
    eligibility = output.target_trial.eligibility_criteria or ""
    eligibility_lower = eligibility.casefold()
    full_text = len(eligibility) >= 1000 and "inclusion" in eligibility_lower and "exclusion" in eligibility_lower
    contradicted = [
        flag.flag_id
        for flag in output.missing_data_flags
        if full_text
        and "eligibility" in f"{flag.flag_id} {flag.field} {flag.reason}".casefold()
        and "truncat" in f"{flag.flag_id} {flag.reason}".casefold()
    ]
    return ValidationResult(
        validation_id=f"validation-{run_id}-protocol-design-missing-flag-semantics",
        target_id=output.output_id,
        status="warning" if contradicted else "passed",
        validator="protocol_design_missing_flag_semantics",
        message=f"Contradicted eligibility missing-data flags remain: {', '.join(contradicted)}" if contradicted else "Missing-data flags do not contradict full target eligibility text.",
        confidence=0.95,
        provenance="pharma_os.validators.validate_protocol_design_constraints",
    )


def _validate_fractional_enrollment(*, run_id: str, output: ProtocolDesignOutput) -> ValidationResult:
    text = " ".join(_string_values(output.protocol_design_brief.model_dump(mode="json"))).casefold()
    bad = re.findall(r"\b\d+\.\d+\s+(?:participants|patients|subjects)\b", text)
    return ValidationResult(
        validation_id=f"validation-{run_id}-protocol-design-fractional-enrollment",
        target_id=output.output_id,
        status="failed" if bad else "passed",
        validator="protocol_design_fractional_enrollment",
        message=f"Fractional enrollment recommendation detected: {', '.join(sorted(set(bad)))}" if bad else "Enrollment recommendations use whole participants or nonnumeric review language.",
        confidence=1.0,
        gate_reason="Participant counts must not be fractional." if bad else None,
        provenance="pharma_os.validators.validate_protocol_design_constraints",
    )


def _validate_duration_label_semantics(*, run_id: str, output: ProtocolDesignOutput) -> ValidationResult:
    text = " ".join(_string_values(output.protocol_design_brief.model_dump(mode="json"))).casefold()
    bad = bool(re.search(r"treatment duration[^.]{0,120}start-to-primary-completion|start-to-primary-completion[^.]{0,120}treatment duration", text))
    return ValidationResult(
        validation_id=f"validation-{run_id}-protocol-design-duration-labels",
        target_id=output.output_id,
        status="warning" if bad else "passed",
        validator="protocol_design_duration_label_semantics",
        message=(
            "Brief may conflate treatment duration with start-to-primary-completion interval."
            if bad
            else "Duration language separates treatment duration from execution/primary-completion intervals."
        ),
        confidence=0.9,
        provenance="pharma_os.validators.validate_protocol_design_constraints",
    )


def _validate_support_source_consistency(*, run_id: str, output: ProtocolDesignOutput) -> ValidationResult:
    bad = []
    for decision in output.analog_derived_design_decisions:
        if decision.support_source_type == "follow_on_supported" and not decision.supporting_follow_on_nct_ids:
            bad.append(decision.decision_id)
        if decision.support_source_type == "analog_majority_supported" and not decision.supporting_analog_nct_ids:
            bad.append(decision.decision_id)
    status = "failed" if bad else "passed"
    return ValidationResult(
        validation_id=f"validation-{run_id}-protocol-design-support-source-consistency",
        target_id=output.output_id,
        status=status,
        validator="protocol_design_support_source_consistency",
        message=f"Support-source labels lack matching supporting IDs: {', '.join(bad)}" if bad else "Support-source labels are consistent with supporting trial IDs.",
        confidence=1.0,
        gate_reason="Support-source labels must match available evidence." if bad else None,
        provenance="pharma_os.validators.validate_protocol_design_constraints",
    )


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_string_values(item))
        return values
    return []


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
    human_gate: HumanGate | None = None,
) -> tuple[ConfidenceFlag, ...]:
    """Generate canonical human-review flags from validation, risk, and gate signals."""

    return canonical_review_flags(
        run_id=run_id,
        validation_results=validation_results,
        risk_flags=risk_flags,
        human_gate=human_gate,
    )


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
