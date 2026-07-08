"""Deterministic Clinical Outcome Prediction Agent 3."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

import httpx

from pharma_os.schemas import (
    ApprovalLikelihoodProxy,
    AssetIdentityOutput,
    AssumptionRecord,
    ClinicalOutcomePredictionInput,
    ClinicalOutcomePredictionOutput,
    ClinicalTrialRecord,
    ComparatorBenchmarkBundle,
    EndpointRiskAssessment,
    EnrollmentDurationRisk,
    EvidenceClaim,
    FailureMode,
    FailureModeClassification,
    HistoricalPoSEstimate,
    LabelExpansionClinicalRationale,
    MissingDataFlag,
    SafetyContext,
    SourceAvailabilityFlag,
    SourceAvailabilityReport,
    SourceMetadata,
    TrialDesignFeatures,
    TrialIdentity,
)
from pharma_os.components.trial_landscape import search_trial_landscape
from pharma_os.tools.asset_identity import resolve_asset_identity
from pharma_os.tools.clinicaltrials import ClinicalTrialsGovClient, ClinicalTrialsGovError
from pharma_os.tools.pos import lookup_pos


OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"


def run_clinical_outcome_prediction_agent(
    input_data: ClinicalOutcomePredictionInput,
    *,
    run_id: str,
    ctgov_client: ClinicalTrialsGovClient | None = None,
    label_client: httpx.Client | None = None,
) -> ClinicalOutcomePredictionOutput:
    """Run deterministic Agent 3 components for one NCT ID."""

    client = ctgov_client or ClinicalTrialsGovClient()
    trial = client.fetch_trial(input_data.nct_id)
    asset, asset_sources = resolve_asset_identity(trial)
    pos, pos_source = lookup_pos(trial, asset, workbook_path=input_data.pos_workbook_path)
    historical_pos = _historical_pos(pos)
    comparator, comparator_sources = _comparator_benchmarks(client, trial, run_id)
    safety_context, label_rationale, label_sources = _label_context(asset, trial, client=label_client)

    trial_identity = _trial_identity(trial)
    design = _trial_design_features(trial)
    endpoint_risk = _endpoint_risk(trial)
    enrollment_risk = _enrollment_duration_risk(trial)
    approval_proxy = _approval_likelihood_proxy(historical_pos, run_id)
    failure_modes = _failure_modes(
        endpoint_risk=endpoint_risk,
        enrollment_risk=enrollment_risk,
        comparator=comparator,
        safety_context=safety_context,
        asset=asset,
    )

    sources = _dedupe_sources(
        (
            *asset_sources,
            pos_source,
            *comparator_sources,
            *label_sources,
        )
    )
    missing_flags = _dedupe_missing_flags(
        (
            *asset.missing_data_flags,
            *historical_pos.missing_data_flags,
            *endpoint_risk.missing_data_flags,
            *enrollment_risk.missing_data_flags,
            *comparator.missing_data_flags,
            *safety_context.missing_data_flags,
            *label_rationale.missing_data_flags,
        )
    )
    assumptions = (*enrollment_risk.assumptions, *approval_proxy.assumptions)
    source_availability = _source_availability(
        trial=trial,
        asset=asset,
        historical_pos=historical_pos,
        safety=safety_context,
    )
    claims = _claims(
        run_id=run_id,
        trial=trial,
        asset=asset,
        design=design,
        endpoint_risk=endpoint_risk,
        enrollment_risk=enrollment_risk,
        comparator=comparator,
        historical_pos=historical_pos,
        approval_proxy=approval_proxy,
        failure_modes=failure_modes,
        safety=safety_context,
        label_rationale=label_rationale,
    )

    return ClinicalOutcomePredictionOutput(
        output_id=f"clinical-outcome-prediction-output-{run_id}",
        run_id=run_id,
        input=input_data,
        trial_identity=trial_identity,
        asset_identity=asset,
        trial_design_features=design,
        endpoint_risk_assessment=endpoint_risk,
        enrollment_duration_risk=enrollment_risk,
        comparator_benchmarking=comparator,
        historical_pos_estimate=historical_pos,
        approval_likelihood_proxy=approval_proxy,
        failure_mode_classification=failure_modes,
        safety_context=safety_context,
        label_expansion_clinical_rationale=label_rationale,
        source_availability=source_availability,
        sources=sources,
        claims=claims,
        assumptions=assumptions,
        missing_data_flags=missing_flags,
        confidence=_overall_confidence(missing_flags, historical_pos, comparator, safety_context),
    )


def _trial_identity(trial: ClinicalTrialRecord) -> TrialIdentity:
    return TrialIdentity(
        nct_id=trial.nct_id,
        brief_title=trial.brief_title,
        official_title=trial.official_title,
        overall_status=trial.overall_status,
        phases=trial.phases,
        conditions=trial.conditions,
        sponsor=trial.lead_sponsor.name if trial.lead_sponsor else None,
        source_ids=(trial.source_id,),
    )


def _trial_design_features(trial: ClinicalTrialRecord) -> TrialDesignFeatures:
    countries = tuple(dict.fromkeys(location.country for location in trial.locations if location.country))
    criteria = trial.eligibility_criteria
    eligibility_summary = None
    if criteria:
        compact = " ".join(criteria.split())
        eligibility_summary = compact[:500]
    return TrialDesignFeatures(
        study_type=trial.study_type,
        arms_count=max(1, len(trial.interventions)) if trial.interventions else 0,
        intervention_count=len(trial.interventions),
        enrollment_count=trial.enrollment_count,
        enrollment_type=trial.enrollment_type,
        primary_endpoint_count=len(trial.primary_endpoints),
        secondary_endpoint_count=len(trial.secondary_endpoints),
        primary_endpoint_measures=tuple(endpoint.measure for endpoint in trial.primary_endpoints),
        secondary_endpoint_measures=tuple(endpoint.measure for endpoint in trial.secondary_endpoints),
        start_date=trial.start_date,
        primary_completion_date=trial.primary_completion_date,
        completion_date=trial.completion_date,
        eligibility_summary=eligibility_summary,
        countries=countries,
        sites_count=len(trial.locations) if trial.locations else None,
        source_ids=(trial.source_id,),
    )


def _endpoint_risk(trial: ClinicalTrialRecord) -> EndpointRiskAssessment:
    flags: list[MissingDataFlag] = []
    factors: list[str] = []
    if not trial.primary_endpoints:
        flags.append(_missing("cop-endpoint-primary-missing", "endpoint_risk", "primary_endpoints", "No primary endpoints were available from ClinicalTrials.gov.", "high"))
        return EndpointRiskAssessment(
            risk_level="high",
            risk_factors=("missing primary endpoints",),
            rationale="Endpoint risk is high because the registry record has no normalized primary endpoint.",
            source_ids=(trial.source_id,),
            missing_data_flags=tuple(flags),
            confidence=0.8,
        )

    measures = " | ".join(endpoint.measure for endpoint in trial.primary_endpoints).casefold()
    if len(trial.primary_endpoints) > 2:
        factors.append("multiple primary endpoints")
    if any(term in measures for term in ("biomarker", "surrogate", "response rate", "orr")):
        factors.append("surrogate or biomarker-oriented primary endpoint language")
    if any(term in measures for term in ("overall survival", "mortality", "death")):
        factors.append("clinically definitive survival or mortality endpoint")

    risk = "low"
    if "multiple primary endpoints" in factors:
        risk = "medium"
    if "surrogate or biomarker-oriented primary endpoint language" in factors and _late_phase(trial):
        risk = "medium"
    rationale = "Endpoint risk is based on primary endpoint count and endpoint wording in the registry record."
    return EndpointRiskAssessment(
        risk_level=risk,
        risk_factors=tuple(factors) or ("single registry primary endpoint",),
        rationale=rationale,
        source_ids=(trial.source_id,),
        confidence=0.7,
    )


def _enrollment_duration_risk(trial: ClinicalTrialRecord) -> EnrollmentDurationRisk:
    flags: list[MissingDataFlag] = []
    factors: list[str] = []
    if trial.enrollment_count is None:
        flags.append(_missing("cop-enrollment-count-missing", "enrollment_duration_risk", "enrollment_count", "ClinicalTrials.gov did not provide enrollment count.", "high"))
    elif trial.enrollment_count < 30:
        factors.append("very small enrollment")
    elif _late_phase(trial) and trial.enrollment_count < 100:
        factors.append("small late-phase enrollment")

    duration_months = _planned_duration_months(trial.start_date, trial.primary_completion_date or trial.completion_date)
    assumptions: list[AssumptionRecord] = []
    if duration_months is None:
        flags.append(_missing("cop-duration-missing", "enrollment_duration_risk", "planned_duration_months", "Start and completion dates could not be converted into a planned duration.", "medium"))
    else:
        assumptions.append(
            AssumptionRecord(
                assumption_id=f"cop-duration-{trial.nct_id}",
                name="planned_duration_months",
                value=round(duration_months, 1),
                unit="months",
                assumption_type="calculated",
                source_ids=(trial.source_id,),
                provenance="ClinicalTrials.gov start_date and primary_completion_date month difference",
            )
        )
        if duration_months > 48:
            factors.append("long planned duration")

    if any("very small" in factor or "small late" in factor for factor in factors):
        risk = "high"
    elif factors or flags:
        risk = "medium"
    else:
        risk = "low"
    rationale = "Enrollment and duration risk is based on registry enrollment count and planned trial timeline."
    return EnrollmentDurationRisk(
        risk_level=risk,
        enrollment_count=trial.enrollment_count,
        planned_duration_months=round(duration_months, 1) if duration_months is not None else None,
        rationale=rationale,
        assumptions=tuple(assumptions),
        source_ids=(trial.source_id,),
        missing_data_flags=tuple(flags),
        confidence=0.75 if not flags else 0.55,
    )


def _comparator_benchmarks(
    client: ClinicalTrialsGovClient,
    trial: ClinicalTrialRecord,
    run_id: str,
) -> tuple[ComparatorBenchmarkBundle, tuple[SourceMetadata, ...]]:
    condition = trial.conditions[0] if trial.conditions else None
    phase = trial.phases[0] if trial.phases else None
    search_source = SourceMetadata(
        source_id=f"ctgov_search:clinical_outcome_prediction:{run_id}",
        title=f"ClinicalTrials.gov comparator search for {trial.nct_id}",
        provenance="ClinicalTrials.gov API v2 studies search by condition and phase",
        source_type="clinical_trial_registry_search",
        version="v2",
    )
    if not condition:
        flag = _missing("cop-comparator-condition-missing", "comparator_benchmarking", "conditions", "No condition was available for comparator benchmark search.", "medium")
        return (
            ComparatorBenchmarkBundle(
                benchmark_summary="Comparator benchmarking was not run because no condition was available.",
                source_ids=(trial.source_id,),
                missing_data_flags=(flag,),
                confidence=0.2,
            ),
            (search_source,),
        )
    try:
        landscape = search_trial_landscape(
            disease=condition,
            phase=phase,
            limit=10,
            run_id=f"{run_id}-agent3-landscape",
            client=client,
        )
    except (ClinicalTrialsGovError, ValueError) as exc:
        flag = _missing("cop-comparator-search-failed", "comparator_benchmarking", "matched_public_trials_count", f"ClinicalTrials.gov comparator search failed: {exc.__class__.__name__}.", "medium")
        return (
            ComparatorBenchmarkBundle(
                benchmark_summary="Comparator benchmarking could not retrieve public trial matches.",
                source_ids=(trial.source_id,),
                missing_data_flags=(flag,),
                confidence=0.2,
            ),
            (search_source,),
        )
    comparators = tuple(record for record in landscape.trials if record.nct_id != trial.nct_id)
    comparator_ids = tuple(record.nct_id for record in comparators[:5])
    summary = f"ClinicalTrials.gov search found {len(comparators)} public comparator trials for {condition}"
    if phase:
        summary += f" and {phase}"
    summary += "."
    sources = _dedupe_sources((search_source, *landscape.sources))
    return (
        ComparatorBenchmarkBundle(
            matched_public_trials_count=len(comparators),
            comparator_trial_ids=comparator_ids,
            benchmark_summary=summary,
            landscape_summary=landscape.landscape_summary,
            status_summary=landscape.status_summary,
            phase_summary=landscape.phase_summary,
            sponsor_summary=landscape.sponsor_summary,
            endpoint_summary=landscape.endpoint_summary,
            population_summary=landscape.population_summary,
            risk_flags=landscape.risk_flags,
            source_ids=tuple(source.source_id for source in sources),
            confidence=0.65 if comparators else 0.4,
        ),
        sources,
    )


def _historical_pos(pos: Any) -> HistoricalPoSEstimate:
    return HistoricalPoSEstimate(
        probability_of_success=pos.probability_of_success,
        current_phase=pos.current_phase,
        disease_area=pos.disease_area,
        lookup_key=pos.lookup_key,
        benchmark_row=pos.benchmark_row,
        assumption_type="source_derived" if pos.probability_of_success is not None else "missing",
        source_ids=pos.source_ids,
        missing_data_flags=pos.missing_data_flags,
        confidence=pos.confidence,
    )


def _approval_likelihood_proxy(pos: HistoricalPoSEstimate, run_id: str) -> ApprovalLikelihoodProxy:
    if pos.probability_of_success is None:
        return ApprovalLikelihoodProxy(
            probability=None,
            basis="No source-backed historical PoS was available, so no approval likelihood proxy was calculated.",
            assumption_type="missing",
            confidence=0.0,
        )
    assumption = AssumptionRecord(
        assumption_id=f"cop-approval-proxy-{run_id}",
        name="approval_likelihood_proxy",
        value=pos.probability_of_success,
        unit="probability",
        assumption_type="source_derived",
        source_ids=pos.source_ids,
        provenance="Clinical outcome prediction maps source-workbook historical PoS directly to approval_likelihood_proxy.",
    )
    return ApprovalLikelihoodProxy(
        probability=pos.probability_of_success,
        basis="Proxy equals the source-workbook historical phase/therapeutic-area PoS; it is not a development decision.",
        assumption_type="source_derived",
        source_ids=pos.source_ids,
        assumptions=(assumption,),
        confidence=pos.confidence,
    )


def _failure_modes(
    *,
    endpoint_risk: EndpointRiskAssessment,
    enrollment_risk: EnrollmentDurationRisk,
    comparator: ComparatorBenchmarkBundle,
    safety_context: SafetyContext,
    asset: AssetIdentityOutput,
) -> FailureModeClassification:
    modes: list[FailureMode] = []
    if endpoint_risk.risk_level in {"medium", "high"}:
        modes.append(
            FailureMode(
                category="endpoint",
                severity=endpoint_risk.risk_level,  # type: ignore[arg-type]
                rationale=endpoint_risk.rationale,
                source_ids=endpoint_risk.source_ids,
            )
        )
    if enrollment_risk.risk_level in {"medium", "high"}:
        modes.append(
            FailureMode(
                category="enrollment",
                severity=enrollment_risk.risk_level,  # type: ignore[arg-type]
                rationale=enrollment_risk.rationale,
                source_ids=enrollment_risk.source_ids,
            )
        )
    if comparator.matched_public_trials_count == 0:
        modes.append(
            FailureMode(
                category="comparator",
                severity="medium",
                rationale="No public comparator trials were matched in the ClinicalTrials.gov benchmark search.",
                source_ids=comparator.source_ids,
            )
        )
    if not safety_context.label_available:
        modes.append(
            FailureMode(
                category="safety",
                severity="medium",
                rationale="No public label safety context was available for the asset or close label match.",
                source_ids=asset.source_ids,
            )
        )
    if asset.confidence < 0.5:
        modes.append(
            FailureMode(
                category="missing_data",
                severity="high",
                rationale="Asset identity confidence is low because key registry or normalization fields are missing.",
                source_ids=asset.source_ids,
            )
        )
    severities = {mode.severity for mode in modes}
    overall = "high" if "high" in severities else "medium" if "medium" in severities else "low"
    return FailureModeClassification(
        likely_failure_modes=tuple(modes),
        overall_risk_level=overall,
        source_ids=tuple(dict.fromkeys(source_id for mode in modes for source_id in mode.source_ids)),
        confidence=0.7 if modes else 0.6,
    )


def _label_context(
    asset: AssetIdentityOutput,
    trial: ClinicalTrialRecord,
    *,
    client: httpx.Client | None,
) -> tuple[SafetyContext, LabelExpansionClinicalRationale, tuple[SourceMetadata, ...]]:
    terms = tuple(dict.fromkeys(term for term in (asset.asset_name, *asset.aliases) if term))
    if not terms:
        flag = _missing("cop-label-terms-missing", "safety_context", "summary", "No asset terms were available for openFDA label lookup.", "medium")
        rationale = LabelExpansionClinicalRationale(
            rationale="Clinical label-expansion rationale could not be assessed because no label search terms were available.",
            source_ids=asset.source_ids,
            missing_data_flags=(flag,),
            confidence=0.1,
        )
        return SafetyContext(missing_data_flags=(flag,), confidence=0.1), rationale, ()

    http_client = client or httpx.Client(timeout=20.0)
    for term in terms[:5]:
        try:
            payload = _openfda_label_search(http_client, term)
        except Exception:
            continue
        if not payload:
            continue
        result = payload
        source = SourceMetadata(
            source_id=f"openfda_label:{_slug(term)}",
            title=f"openFDA label search for {term}",
            url=OPENFDA_LABEL_URL,
            provenance="openFDA drug label API",
            source_type="drug_label",
            version="openFDA",
        )
        warnings = _first_text(result.get("warnings"))
        adverse = _first_text(result.get("adverse_reactions"))
        indication = _first_text(result.get("indications_and_usage"))
        summary_parts = [part for part in (warnings, adverse) if part]
        summary = " ".join(summary_parts)[:1000] if summary_parts else "A public label match was found, but warnings/adverse reaction text was not available."
        condition_text = ", ".join(trial.conditions) or "the trial condition"
        rationale_text = f"Registry condition context ({condition_text}) can be compared with public label indications for clinical expansion rationale."
        if indication:
            rationale_text += f" Label indication context: {indication[:600]}"
        return (
            SafetyContext(
                label_available=True,
                summary=summary,
                source_ids=(source.source_id,),
                confidence=0.65,
            ),
            LabelExpansionClinicalRationale(
                rationale=rationale_text,
                source_ids=(trial.source_id, source.source_id),
                confidence=0.55,
            ),
            (source,),
        )

    flag = _missing("cop-label-not-found", "safety_context", "summary", "openFDA returned no usable label match for the asset terms.", "medium")
    rationale = LabelExpansionClinicalRationale(
        rationale="Clinical label-expansion rationale is limited to registry condition and asset identity because no public label match was found.",
        source_ids=tuple(dict.fromkeys((trial.source_id, *asset.source_ids))),
        missing_data_flags=(flag,),
        confidence=0.25,
    )
    return SafetyContext(missing_data_flags=(flag,), confidence=0.15), rationale, ()


def _openfda_label_search(client: httpx.Client, term: str) -> dict[str, Any] | None:
    for field in ("openfda.brand_name", "openfda.generic_name"):
        response = client.get(
            OPENFDA_LABEL_URL,
            params={"search": f'{field}:"{term}"', "limit": "1"},
            timeout=20.0,
        )
        if response.status_code == 404:
            continue
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return results[0]
    return None


def _source_availability(
    *,
    trial: ClinicalTrialRecord,
    asset: AssetIdentityOutput,
    historical_pos: HistoricalPoSEstimate,
    safety: SafetyContext,
) -> SourceAvailabilityReport:
    flags = [
        SourceAvailabilityFlag(
            source_name="ClinicalTrials.gov API",
            status="available",
            reason="Trial protocol data were retrieved from the public ClinicalTrials.gov API.",
            source_type="clinical_trial_registry",
            source_ids=(trial.source_id,),
        ),
        SourceAvailabilityFlag(
            source_name="RxNorm/shared normalization config",
            status="available" if asset.rxnorm_match or asset.rule_ids else "source_unavailable",
            reason="Asset identity used RxNorm when matched and shared deterministic rules/config.",
            source_type="drug_normalization",
            source_ids=asset.source_ids,
        ),
        SourceAvailabilityFlag(
            source_name="Source-based PoS workbook",
            status="available" if historical_pos.probability_of_success is not None else "source_unavailable",
            reason="Historical PoS lookup used the local workbook when a phase and disease-area row matched.",
            source_type="pos_workbook",
            source_ids=historical_pos.source_ids,
        ),
        SourceAvailabilityFlag(
            source_name="openFDA drug labels",
            status="available" if safety.label_available else "source_unavailable",
            reason="Public label safety context is included only when openFDA returns a usable label match.",
            source_type="drug_label",
            source_ids=safety.source_ids,
        ),
        SourceAvailabilityFlag(
            source_name="PubMed literature extraction",
            status="not_implemented",
            reason="Bounded PubMed evidence extraction is outside this pass; no literature claims were generated.",
            source_type="literature",
        ),
        SourceAvailabilityFlag(
            source_name="AACT",
            status="not_implemented",
            reason="AACT was optional and not added in this pass.",
            source_type="clinical_trial_registry_database",
        ),
        SourceAvailabilityFlag(
            source_name="CTOD",
            status="not_implemented",
            reason="CTOD integration is explicitly out of scope for this pass.",
            source_type="external_model",
        ),
        SourceAvailabilityFlag(
            source_name="TrialBench/HINT/TOP-style models",
            status="not_implemented",
            reason="External trial-outcome models are explicitly out of scope for this pass.",
            source_type="external_model",
        ),
        SourceAvailabilityFlag(
            source_name="Paid or proprietary clinical intelligence databases",
            status="not_implemented",
            reason="Commercial databases and proprietary RWE sources are explicitly out of scope.",
            source_type="commercial_database",
        ),
    ]
    return SourceAvailabilityReport(flags=tuple(flags))


def _claims(
    *,
    run_id: str,
    trial: ClinicalTrialRecord,
    asset: AssetIdentityOutput,
    design: TrialDesignFeatures,
    endpoint_risk: EndpointRiskAssessment,
    enrollment_risk: EnrollmentDurationRisk,
    comparator: ComparatorBenchmarkBundle,
    historical_pos: HistoricalPoSEstimate,
    approval_proxy: ApprovalLikelihoodProxy,
    failure_modes: FailureModeClassification,
    safety: SafetyContext,
    label_rationale: LabelExpansionClinicalRationale,
) -> tuple[EvidenceClaim, ...]:
    claims = [
        EvidenceClaim(
            claim_id=f"claim-{run_id}-trial-identity",
            claim_text=f"{trial.nct_id} has ClinicalTrials.gov status {trial.overall_status or 'unknown'} and phases {', '.join(trial.phases) or 'unknown'}.",
            source_ids=(trial.source_id,),
            provenance="clinical_outcome_prediction.ctgov_trial_identity",
            confidence=0.95,
            confidence_level="very_high",
        ),
        EvidenceClaim(
            claim_id=f"claim-{run_id}-trial-design",
            claim_text=f"{trial.nct_id} reports enrollment {design.enrollment_count if design.enrollment_count is not None else 'unknown'} and {design.primary_endpoint_count} primary endpoints.",
            source_ids=(trial.source_id,),
            provenance="clinical_outcome_prediction.ctgov_design_features",
            confidence=0.9,
            confidence_level="high",
        ),
        EvidenceClaim(
            claim_id=f"claim-{run_id}-endpoint-risk",
            claim_text=f"Endpoint risk level is {endpoint_risk.risk_level} based on registry primary endpoint structure.",
            source_ids=endpoint_risk.source_ids,
            provenance="clinical_outcome_prediction.endpoint_risk_rules",
            confidence=endpoint_risk.confidence,
            confidence_level=_confidence_level(endpoint_risk.confidence),
        ),
        EvidenceClaim(
            claim_id=f"claim-{run_id}-enrollment-risk",
            claim_text=f"Enrollment and duration risk level is {enrollment_risk.risk_level}.",
            source_ids=enrollment_risk.source_ids,
            provenance="clinical_outcome_prediction.enrollment_duration_rules",
            confidence=enrollment_risk.confidence,
            confidence_level=_confidence_level(enrollment_risk.confidence),
        ),
        EvidenceClaim(
            claim_id=f"claim-{run_id}-comparator-benchmarking",
            claim_text=f"ClinicalTrials.gov comparator benchmarking matched {comparator.matched_public_trials_count} public trials.",
            source_ids=comparator.source_ids,
            provenance="clinical_outcome_prediction.ctgov_comparator_search",
            confidence=comparator.confidence,
            confidence_level=_confidence_level(comparator.confidence),
        ),
        EvidenceClaim(
            claim_id=f"claim-{run_id}-failure-modes",
            claim_text=f"Failure-mode classification overall risk level is {failure_modes.overall_risk_level}.",
            source_ids=failure_modes.source_ids or (trial.source_id,),
            provenance="clinical_outcome_prediction.failure_mode_rules",
            confidence=failure_modes.confidence,
            confidence_level=_confidence_level(failure_modes.confidence),
        ),
    ]
    if asset.asset_name:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-asset-identity",
                claim_text=f"The inferred clinical asset for {trial.nct_id} is {asset.asset_name}.",
                source_ids=asset.source_ids or (trial.source_id,),
                provenance="clinical_outcome_prediction.asset_identity",
                confidence=asset.confidence,
                confidence_level=_confidence_level(asset.confidence),
            )
        )
    if historical_pos.probability_of_success is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-historical-pos",
                claim_text=f"Historical PoS estimate is {historical_pos.probability_of_success:.3f} for {historical_pos.disease_area} {historical_pos.current_phase}.",
                source_ids=historical_pos.source_ids,
                provenance="clinical_outcome_prediction.pos_workbook",
                confidence=historical_pos.confidence,
                confidence_level=_confidence_level(historical_pos.confidence),
            )
        )
    if approval_proxy.probability is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-approval-proxy",
                claim_text=f"Approval likelihood proxy is {approval_proxy.probability:.3f} using the source-backed historical PoS basis.",
                source_ids=approval_proxy.source_ids,
                provenance="clinical_outcome_prediction.approval_likelihood_proxy",
                confidence=approval_proxy.confidence,
                confidence_level=_confidence_level(approval_proxy.confidence),
            )
        )
    if safety.label_available:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-safety-context",
                claim_text="Public label safety context was found for the asset or close label term.",
                source_ids=safety.source_ids,
                provenance="clinical_outcome_prediction.openfda_label",
                confidence=safety.confidence,
                confidence_level=_confidence_level(safety.confidence),
            )
        )
    if label_rationale.source_ids:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-label-expansion-rationale",
                claim_text="Clinical label-expansion rationale was structured from registry condition context and available public label context.",
                source_ids=label_rationale.source_ids,
                provenance="clinical_outcome_prediction.label_expansion_rationale",
                confidence=label_rationale.confidence,
                confidence_level=_confidence_level(label_rationale.confidence),
            )
        )
    return tuple(claims)


def _planned_duration_months(start_value: str | None, end_value: str | None) -> float | None:
    start = _parse_date(start_value)
    end = _parse_date(end_value)
    if not start or not end or end < start:
        return None
    return (end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.44


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.date()
        except ValueError:
            continue
    return None


def _late_phase(trial: ClinicalTrialRecord) -> bool:
    text = " ".join(trial.phases).upper()
    return "PHASE2" in text or "PHASE 2" in text or "PHASE3" in text or "PHASE 3" in text


def _missing(flag_id: str, section: str, field: str, reason: str, severity: str) -> MissingDataFlag:
    return MissingDataFlag(flag_id=flag_id, section=section, field=field, reason=reason, severity=severity)  # type: ignore[arg-type]


def _dedupe_sources(sources: tuple[SourceMetadata, ...]) -> tuple[SourceMetadata, ...]:
    deduped: dict[str, SourceMetadata] = {}
    for source in sources:
        deduped[source.source_id] = source
    return tuple(deduped.values())


def _dedupe_missing_flags(flags: tuple[MissingDataFlag, ...]) -> tuple[MissingDataFlag, ...]:
    deduped: dict[str, MissingDataFlag] = {}
    for flag in flags:
        deduped[flag.flag_id] = flag
    return tuple(deduped.values())


def _first_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return _first_text(value[0]) if value else None
    if isinstance(value, dict):
        for item in value.values():
            text = _first_text(item)
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "unknown"


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"


def _overall_confidence(
    flags: tuple[MissingDataFlag, ...],
    historical_pos: HistoricalPoSEstimate,
    comparator: ComparatorBenchmarkBundle,
    safety: SafetyContext,
) -> float:
    confidence = 0.8
    confidence -= sum(0.12 for flag in flags if flag.severity == "high")
    confidence -= sum(0.05 for flag in flags if flag.severity == "medium")
    if historical_pos.probability_of_success is None:
        confidence -= 0.15
    if comparator.matched_public_trials_count == 0:
        confidence -= 0.05
    if not safety.label_available:
        confidence -= 0.05
    return max(0.1, min(0.9, confidence))
