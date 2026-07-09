"""Deterministic Clinical Outcome Prediction Agent 3."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import httpx
from pydantic import BaseModel

from pharma_os.agent_runtime import (
    AgentRuntimeConfig,
    StructuredAgentResult,
    load_agents_sdk,
    run_structured_agent,
    run_structured_llm_call,
    runtime_config_for_live_agents,
)
from pharma_os.schemas import (
    AgentRunTrace,
    ApprovalLikelihoodProxy,
    AssetIdentityAdjudication,
    AssetIdentityOutput,
    AssumptionRecord,
    ClinicalOutcomeManagerPlan,
    ClinicalOutcomePredictionInput,
    ClinicalOutcomePredictionOutput,
    ClinicalTrialRecord,
    ComparatorBenchmarkBundle,
    ComparatorRelevanceOutput,
    ComparatorTrialRelevance,
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

_DIRECT_LLM_AGENT_NAMES = frozenset(
    {
        "AssetIdentityAdjudicatorAgent",
        "EndpointRiskAgent",
        "ComparatorRelevanceAgent",
        "EnrollmentFeasibilityAgent",
        "SafetyContextAgent",
        "FailureModeSynthesisAgent",
    }
)


@dataclass(frozen=True)
class ClinicalOutcomePredictionAgentResult:
    """Agent 3 output plus persisted reasoning artifacts."""

    output: ClinicalOutcomePredictionOutput
    traces: tuple[AgentRunTrace, ...]
    subagent_payloads: tuple[BaseModel, ...]


@dataclass(frozen=True)
class ClinicalOutcomeManagerResult:
    """Manager/subagent results used to preserve the final Agent 3 output shape."""

    manager_plan: ClinicalOutcomeManagerPlan
    asset_identity: AssetIdentityOutput
    endpoint_risk_assessment: EndpointRiskAssessment
    enrollment_duration_risk: EnrollmentDurationRisk
    comparator_benchmarking: ComparatorBenchmarkBundle
    comparator_relevance: ComparatorRelevanceOutput
    failure_mode_classification: FailureModeClassification
    safety_context: SafetyContext
    label_expansion_clinical_rationale: LabelExpansionClinicalRationale
    traces: tuple[AgentRunTrace, ...]
    subagent_payloads: tuple[BaseModel, ...]


def run_clinical_outcome_prediction_agent(
    input_data: ClinicalOutcomePredictionInput,
    *,
    run_id: str,
    ctgov_client: ClinicalTrialsGovClient | None = None,
    label_client: httpx.Client | None = None,
    config: AgentRuntimeConfig | None = None,
) -> ClinicalOutcomePredictionOutput:
    """Run SDK-backed Agent 3 clinical reasoning and return the final output."""

    return run_clinical_outcome_prediction_agent_result(
        input_data,
        run_id=run_id,
        ctgov_client=ctgov_client,
        label_client=label_client,
        config=config,
    ).output


def run_clinical_outcome_prediction_agent_result(
    input_data: ClinicalOutcomePredictionInput,
    *,
    run_id: str,
    ctgov_client: ClinicalTrialsGovClient | None = None,
    label_client: httpx.Client | None = None,
    config: AgentRuntimeConfig | None = None,
) -> ClinicalOutcomePredictionAgentResult:
    """Run deterministic Agent 3 retrieval/math, then bounded clinical-reasoning agents."""

    client = ctgov_client or ClinicalTrialsGovClient()
    trial = client.fetch_trial(input_data.nct_id)
    asset, asset_sources = resolve_asset_identity(trial)
    pos, pos_source = lookup_pos(trial, asset, workbook_path=input_data.pos_workbook_path)
    historical_pos = _historical_pos(pos)
    comparator, comparator_sources, comparator_records = _comparator_benchmarks(client, trial, run_id)
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
    manager_result = run_clinical_outcome_manager_agent(
        run_id=run_id,
        input_data=input_data,
        trial=trial,
        comparator_records=comparator_records,
        trial_identity=trial_identity,
        design=design,
        asset=asset,
        endpoint_risk=endpoint_risk,
        enrollment_risk=enrollment_risk,
        comparator=comparator,
        historical_pos=historical_pos,
        approval_proxy=approval_proxy,
        failure_modes=failure_modes,
        safety_context=safety_context,
        label_rationale=label_rationale,
        source_ids=tuple(
            source.source_id
            for source in _dedupe_sources(
                (
                    *asset_sources,
                    pos_source,
                    *comparator_sources,
                    *label_sources,
                )
            )
        ),
        config=config,
    )
    asset = manager_result.asset_identity
    endpoint_risk = manager_result.endpoint_risk_assessment
    enrollment_risk = manager_result.enrollment_duration_risk
    comparator = manager_result.comparator_benchmarking
    failure_modes = manager_result.failure_mode_classification
    safety_context = manager_result.safety_context
    label_rationale = manager_result.label_expansion_clinical_rationale

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

    output = ClinicalOutcomePredictionOutput(
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
    return ClinicalOutcomePredictionAgentResult(
        output=output,
        traces=manager_result.traces,
        subagent_payloads=manager_result.subagent_payloads,
    )


def run_clinical_outcome_manager_agent(
    *,
    run_id: str,
    input_data: ClinicalOutcomePredictionInput,
    trial: ClinicalTrialRecord,
    comparator_records: tuple[ClinicalTrialRecord, ...],
    trial_identity: TrialIdentity,
    design: TrialDesignFeatures,
    asset: AssetIdentityOutput,
    endpoint_risk: EndpointRiskAssessment,
    enrollment_risk: EnrollmentDurationRisk,
    comparator: ComparatorBenchmarkBundle,
    historical_pos: HistoricalPoSEstimate,
    approval_proxy: ApprovalLikelihoodProxy,
    failure_modes: FailureModeClassification,
    safety_context: SafetyContext,
    label_rationale: LabelExpansionClinicalRationale,
    source_ids: tuple[str, ...],
    config: AgentRuntimeConfig | None = None,
) -> ClinicalOutcomeManagerResult:
    """Run Agent 3 manager/subagents after deterministic retrieval and math."""

    runtime_config = _agent3_runtime_config(config)
    traces: list[AgentRunTrace] = []
    payloads: list[BaseModel] = []
    common_payload = _agent3_base_payload(
        input_data=input_data,
        trial=trial,
        comparator_records=comparator_records,
        trial_identity=trial_identity,
        design=design,
        asset=asset,
        endpoint_risk=endpoint_risk,
        enrollment_risk=enrollment_risk,
        comparator=comparator,
        historical_pos=historical_pos,
        approval_proxy=approval_proxy,
        failure_modes=failure_modes,
        safety_context=safety_context,
        label_rationale=label_rationale,
        source_ids=source_ids,
    )

    manager_plan = _run_typed_agent(
        agent_name="ClinicalOutcomeManagerAgent",
        instructions=_manager_instructions(),
        output_type=ClinicalOutcomeManagerPlan,
        run_id=run_id,
        input_summary=f"Coordinate Agent 3 clinical risk reasoning for {trial.nct_id}.",
        payload=common_payload,
        fallback_output=_manager_plan_fallback(run_id, trial, asset, source_ids),
        source_ids=source_ids,
        confidence=0.7,
        config=runtime_config,
        rationale_summary="Coordinate bounded clinical interpretation after deterministic retrieval/math.",
    )
    traces.append(manager_plan.trace)
    payloads.append(manager_plan.output)

    updated_asset = asset
    if _asset_identity_ambiguous(trial, asset):
        adjudication = _run_typed_agent(
            agent_name="AssetIdentityAdjudicatorAgent",
            instructions=_asset_identity_instructions(),
            output_type=AssetIdentityAdjudication,
            run_id=run_id,
            input_summary=f"Adjudicate ambiguous asset identity for {trial.nct_id}.",
            payload=common_payload,
            fallback_output=_asset_adjudication_fallback(run_id, trial, asset),
            source_ids=asset.source_ids or (trial.source_id,),
            confidence=min(0.75, asset.confidence),
            config=runtime_config,
            rationale_summary="Flag asset identity ambiguity without forcing certainty.",
        )
        traces.append(adjudication.trace)
        payloads.append(adjudication.output)
        updated_asset = _apply_asset_adjudication(asset, adjudication.output)

    endpoint_result = _run_typed_agent(
        agent_name="EndpointRiskAgent",
        instructions=_endpoint_instructions(),
        output_type=EndpointRiskAssessment,
        run_id=run_id,
        input_summary=f"Interpret endpoint risks for {trial.nct_id}.",
        payload=common_payload,
        fallback_output=endpoint_risk,
        source_ids=endpoint_risk.source_ids,
        confidence=endpoint_risk.confidence,
        config=runtime_config,
        rationale_summary="Classify endpoint family, hierarchy, phase fit, and credibility risks without inferring success.",
    )
    traces.append(endpoint_result.trace)
    payloads.append(endpoint_result.output)
    updated_endpoint = _ensure_component_sources(endpoint_result.output, endpoint_risk.source_ids)

    relevance_result = _run_typed_agent(
        agent_name="ComparatorRelevanceAgent",
        instructions=_comparator_instructions(),
        output_type=ComparatorRelevanceOutput,
        run_id=run_id,
        input_summary=f"Judge comparator relevance for {trial.nct_id}.",
        payload=common_payload,
        fallback_output=_comparator_relevance_fallback(run_id, trial, comparator_records, comparator),
        source_ids=comparator.source_ids,
        confidence=comparator.confidence,
        config=runtime_config,
        rationale_summary="Classify deterministic comparator candidates as relevant, weak, or excluded without inventing outcomes.",
    )
    traces.append(relevance_result.trace)
    payloads.append(relevance_result.output)
    comparator_relevance = relevance_result.output
    updated_comparator = _apply_comparator_relevance(comparator, comparator_relevance)

    enrollment_result = _run_typed_agent(
        agent_name="EnrollmentFeasibilityAgent",
        instructions=_enrollment_instructions(),
        output_type=EnrollmentDurationRisk,
        run_id=run_id,
        input_summary=f"Interpret enrollment feasibility for {trial.nct_id}.",
        payload=common_payload,
        fallback_output=enrollment_risk,
        source_ids=enrollment_risk.source_ids,
        confidence=enrollment_risk.confidence,
        config=runtime_config,
        rationale_summary="Explain feasibility risk from registry enrollment, dates, sites, eligibility, and restrictions.",
    )
    traces.append(enrollment_result.trace)
    payloads.append(enrollment_result.output)
    updated_enrollment = _ensure_component_sources(enrollment_result.output, enrollment_risk.source_ids)

    safety_result = _run_typed_agent(
        agent_name="SafetyContextAgent",
        instructions=_safety_instructions(),
        output_type=SafetyContext,
        run_id=run_id,
        input_summary=f"Interpret label and registry safety context for {trial.nct_id}.",
        payload=common_payload,
        fallback_output=safety_context,
        source_ids=safety_context.source_ids or updated_asset.source_ids,
        confidence=safety_context.confidence,
        config=runtime_config,
        rationale_summary="Separate known label-derived risks from missing safety context without inventing adverse-event rates.",
    )
    traces.append(safety_result.trace)
    payloads.append(safety_result.output)
    updated_safety = _ensure_component_sources(safety_result.output, safety_context.source_ids)

    updated_label_rationale = label_rationale
    if updated_safety is not safety_context:
        updated_label_rationale = label_rationale.model_copy(
            update={
                "source_ids": tuple(dict.fromkeys((*label_rationale.source_ids, *updated_safety.source_ids))),
                "confidence": min(label_rationale.confidence or 0.0, updated_safety.confidence or 0.0)
                if label_rationale.confidence
                else updated_safety.confidence,
            }
        )

    failure_fallback = _failure_modes(
        endpoint_risk=updated_endpoint,
        enrollment_risk=updated_enrollment,
        comparator=updated_comparator,
        safety_context=updated_safety,
        asset=updated_asset,
    )
    failure_payload = {
        **common_payload,
        "updated_asset_identity": updated_asset.model_dump(mode="json"),
        "updated_endpoint_risk_assessment": updated_endpoint.model_dump(mode="json"),
        "updated_enrollment_duration_risk": updated_enrollment.model_dump(mode="json"),
        "updated_comparator_benchmarking": updated_comparator.model_dump(mode="json"),
        "updated_safety_context": updated_safety.model_dump(mode="json"),
        "comparator_relevance": comparator_relevance.model_dump(mode="json"),
    }
    failure_result = _run_typed_agent(
        agent_name="FailureModeSynthesisAgent",
        instructions=_failure_mode_instructions(),
        output_type=FailureModeClassification,
        run_id=run_id,
        input_summary=f"Synthesize likely failure-mode categories for {trial.nct_id}.",
        payload=failure_payload,
        fallback_output=failure_fallback,
        source_ids=failure_fallback.source_ids or source_ids,
        confidence=failure_fallback.confidence,
        config=runtime_config,
        rationale_summary="Integrate risk patterns into categorical failure modes, not an outcome prediction.",
    )
    traces.append(failure_result.trace)
    payloads.append(failure_result.output)
    updated_failure = _ensure_component_sources(failure_result.output, failure_fallback.source_ids or source_ids)

    return ClinicalOutcomeManagerResult(
        manager_plan=manager_plan.output,
        asset_identity=updated_asset,
        endpoint_risk_assessment=updated_endpoint,
        enrollment_duration_risk=updated_enrollment,
        comparator_benchmarking=updated_comparator,
        comparator_relevance=comparator_relevance,
        failure_mode_classification=updated_failure,
        safety_context=updated_safety,
        label_expansion_clinical_rationale=updated_label_rationale,
        traces=tuple(traces),
        subagent_payloads=tuple(payloads),
    )


def _agent3_runtime_config(config: AgentRuntimeConfig | None) -> AgentRuntimeConfig:
    if config is not None:
        return config
    return runtime_config_for_live_agents(disabled_provenance="pharma_os.agents.clinical_outcome_prediction")


def _run_typed_agent(
    *,
    agent_name: str,
    instructions: str,
    output_type: type[Any],
    run_id: str,
    input_summary: str,
    payload: dict[str, Any],
    fallback_output: BaseModel,
    source_ids: tuple[str, ...],
    confidence: float,
    config: AgentRuntimeConfig,
    rationale_summary: str,
) -> StructuredAgentResult:
    if agent_name in _DIRECT_LLM_AGENT_NAMES:
        return run_structured_llm_call(
            agent_name=agent_name,
            instructions=instructions,
            payload=payload,
            output_type=output_type,
            run_id=run_id,
            input_summary=input_summary,
            config=config,
            offline_output=fallback_output,
            source_ids=source_ids,
            confidence=confidence,
            rationale_summary=rationale_summary,
        )

    agent = object()
    if not config.disabled:
        Agent, _, _, _ = load_agents_sdk()
        agent = Agent(
            name=agent_name,
            instructions=instructions,
            model=config.model,
            output_type=output_type,
        )
    return run_structured_agent(
        agent=agent,
        payload=payload,
        output_type=output_type,
        agent_name=agent_name,
        run_id=run_id,
        input_summary=input_summary,
        config=config,
        offline_output=fallback_output,
        source_ids=source_ids,
        confidence=confidence,
        rationale_summary=rationale_summary,
    )


def _agent3_base_payload(
    *,
    input_data: ClinicalOutcomePredictionInput,
    trial: ClinicalTrialRecord,
    comparator_records: tuple[ClinicalTrialRecord, ...],
    trial_identity: TrialIdentity,
    design: TrialDesignFeatures,
    asset: AssetIdentityOutput,
    endpoint_risk: EndpointRiskAssessment,
    enrollment_risk: EnrollmentDurationRisk,
    comparator: ComparatorBenchmarkBundle,
    historical_pos: HistoricalPoSEstimate,
    approval_proxy: ApprovalLikelihoodProxy,
    failure_modes: FailureModeClassification,
    safety_context: SafetyContext,
    label_rationale: LabelExpansionClinicalRationale,
    source_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "input": input_data.model_dump(mode="json"),
        "target_trial": trial.model_dump(mode="json"),
        "comparator_candidates": [record.model_dump(mode="json") for record in comparator_records],
        "trial_identity": trial_identity.model_dump(mode="json"),
        "trial_design_features": design.model_dump(mode="json"),
        "asset_identity": asset.model_dump(mode="json"),
        "endpoint_risk_assessment": endpoint_risk.model_dump(mode="json"),
        "enrollment_duration_risk": enrollment_risk.model_dump(mode="json"),
        "comparator_benchmarking": comparator.model_dump(mode="json"),
        "historical_pos_estimate": historical_pos.model_dump(mode="json"),
        "approval_likelihood_proxy": approval_proxy.model_dump(mode="json"),
        "failure_mode_classification": failure_modes.model_dump(mode="json"),
        "safety_context": safety_context.model_dump(mode="json"),
        "label_expansion_clinical_rationale": label_rationale.model_dump(mode="json"),
        "source_ids": source_ids,
        "guardrails": (
            "Agent 3 is not an outcome oracle. Use only source-backed evidence. Do not invent efficacy results, "
            "safety rates, approval probability, PoS, enrollment rates, site performance, competitor outcomes, or final "
            "clinical, approval, investment, licensing, or go/no-go decisions."
        ),
    }


def _manager_plan_fallback(
    run_id: str,
    trial: ClinicalTrialRecord,
    asset: AssetIdentityOutput,
    source_ids: tuple[str, ...],
) -> ClinicalOutcomeManagerPlan:
    ordered_agents = [
        "EndpointRiskAgent",
        "ComparatorRelevanceAgent",
        "EnrollmentFeasibilityAgent",
        "SafetyContextAgent",
        "FailureModeSynthesisAgent",
    ]
    if _asset_identity_ambiguous(trial, asset):
        ordered_agents.insert(0, "AssetIdentityAdjudicatorAgent")
    return ClinicalOutcomeManagerPlan(
        output_id=f"clinical-outcome-manager-plan-{run_id}",
        nct_id=trial.nct_id,
        ordered_agents=tuple(ordered_agents),
        guardrail_summary="Agent 3 interprets clinical risk patterns only; missing or ambiguous evidence is routed to flags and human review.",
        rationale_summary="Coordinate clinical reasoning after deterministic CT.gov, RxNorm, PoS, comparator, duration, and openFDA steps.",
        source_ids=source_ids,
        missing_data_flags=asset.missing_data_flags,
        confidence=0.7 if not asset.missing_data_flags else 0.55,
    )


def _asset_identity_ambiguous(trial: ClinicalTrialRecord, asset: AssetIdentityOutput) -> bool:
    non_placebo = tuple(item for item in trial.interventions if "placebo" not in item.name.casefold())
    return any(
        (
            len(non_placebo) > 1,
            asset.rxnorm_match is None,
            not asset.asset_name,
            not asset.sponsor,
            not asset.normalized_indication,
            (asset.modality or "unknown") == "unknown",
            any(flag.field in {"asset_name", "rxnorm_match", "modality", "normalized_indication", "sponsor"} for flag in asset.missing_data_flags),
        )
    )


def _asset_adjudication_fallback(
    run_id: str,
    trial: ClinicalTrialRecord,
    asset: AssetIdentityOutput,
) -> AssetIdentityAdjudication:
    reasons = []
    if len(tuple(item for item in trial.interventions if "placebo" not in item.name.casefold())) > 1:
        reasons.append("multiple non-placebo interventions or combination therapy")
    if asset.rxnorm_match is None:
        reasons.append("RxNorm returned no deterministic match")
    if not asset.sponsor:
        reasons.append("lead sponsor is missing or ambiguous")
    if not asset.normalized_indication:
        reasons.append("normalized indication is missing")
    if (asset.modality or "unknown") == "unknown":
        reasons.append("modality is unknown")
    flags = (
        *asset.missing_data_flags,
        _missing(
            f"cop-asset-adjudication-{trial.nct_id}",
            "asset_identity",
            "asset_name",
            "Asset identity requires clinical review because deterministic resolution has ambiguity signals.",
            "medium",
        ),
    )
    return AssetIdentityAdjudication(
        output_id=f"asset-identity-adjudication-{run_id}",
        nct_id=trial.nct_id,
        is_ambiguous=True,
        ambiguity_reasons=tuple(dict.fromkeys(reasons or ("asset identity ambiguity detected",))),
        recommended_asset_name=asset.asset_name,
        recommended_modality=asset.modality if asset.modality != "unknown" else None,
        recommended_indication=asset.normalized_indication,
        review_questions=("Which intervention is the target asset for downstream diligence?",),
        source_ids=asset.source_ids or (trial.source_id,),
        missing_data_flags=_dedupe_missing_flags(flags),
        confidence=min(asset.confidence, 0.55),
    )


def _apply_asset_adjudication(
    asset: AssetIdentityOutput,
    adjudication: AssetIdentityAdjudication,
) -> AssetIdentityOutput:
    updates: dict[str, Any] = {
        "missing_data_flags": _dedupe_missing_flags((*asset.missing_data_flags, *adjudication.missing_data_flags)),
        "confidence": min(asset.confidence, adjudication.confidence),
    }
    if not asset.asset_name and adjudication.recommended_asset_name:
        updates["asset_name"] = adjudication.recommended_asset_name
    if (not asset.modality or asset.modality == "unknown") and adjudication.recommended_modality:
        updates["modality"] = adjudication.recommended_modality
    if not asset.normalized_indication and adjudication.recommended_indication:
        updates["normalized_indication"] = adjudication.recommended_indication
    return asset.model_copy(update=updates)


def _comparator_relevance_fallback(
    run_id: str,
    trial: ClinicalTrialRecord,
    comparator_records: tuple[ClinicalTrialRecord, ...],
    comparator: ComparatorBenchmarkBundle,
) -> ComparatorRelevanceOutput:
    judgments = tuple(_judge_comparator_relevance(trial, candidate) for candidate in comparator_records if candidate.nct_id != trial.nct_id)
    counts = {key: sum(1 for item in judgments if item.relevance == key) for key in ("relevant", "weak", "excluded")}
    if not judgments and comparator.missing_data_flags:
        summary = "Comparator relevance could not be reviewed because no comparator candidates were available."
    else:
        summary = f"Comparator relevance review classified {counts['relevant']} relevant, {counts['weak']} weak, and {counts['excluded']} excluded CT.gov candidates."
    return ComparatorRelevanceOutput(
        output_id=f"comparator-relevance-{run_id}",
        target_nct_id=trial.nct_id,
        trial_relevance=judgments,
        relevance_summary=summary,
        source_ids=comparator.source_ids,
        missing_data_flags=comparator.missing_data_flags,
        confidence=comparator.confidence,
    )


def _judge_comparator_relevance(
    target: ClinicalTrialRecord,
    candidate: ClinicalTrialRecord,
) -> ComparatorTrialRelevance:
    matched: list[str] = []
    mismatched: list[str] = []
    target_conditions = {item.casefold() for item in target.conditions}
    candidate_conditions = {item.casefold() for item in candidate.conditions}
    if target_conditions & candidate_conditions:
        matched.append("indication")
    else:
        mismatched.append("indication")
    if set(target.phases) & set(candidate.phases):
        matched.append("phase")
    else:
        mismatched.append("phase")
    target_endpoint = _endpoint_family_text(target)
    candidate_endpoint = _endpoint_family_text(candidate)
    if target_endpoint and candidate_endpoint and target_endpoint == candidate_endpoint:
        matched.append("endpoint_family")
    elif candidate_endpoint:
        mismatched.append("endpoint_family")
    target_types = {(item.type or "").casefold() for item in target.interventions if item.type}
    candidate_types = {(item.type or "").casefold() for item in candidate.interventions if item.type}
    if target_types and candidate_types and target_types & candidate_types:
        matched.append("modality_or_intervention_type")
    elif candidate_types:
        mismatched.append("modality_or_intervention_type")

    if "indication" in matched and "phase" in matched and "endpoint_family" in matched:
        relevance = "relevant"
    elif "indication" in matched or ("phase" in matched and "endpoint_family" in matched):
        relevance = "weak"
    else:
        relevance = "excluded"
    rationale = f"{candidate.nct_id} comparator relevance is {relevance} based on matched dimensions: {', '.join(matched) or 'none'}."
    return ComparatorTrialRelevance(
        nct_id=candidate.nct_id,
        relevance=relevance,
        rationale=rationale,
        matched_dimensions=tuple(matched),
        mismatched_dimensions=tuple(mismatched),
        source_ids=(candidate.source_id,),
        confidence=0.7 if relevance == "relevant" else 0.55 if relevance == "weak" else 0.5,
    )


def _endpoint_family_text(trial: ClinicalTrialRecord) -> str | None:
    text = " ".join(endpoint.measure for endpoint in trial.primary_endpoints).casefold()
    if not text:
        return None
    if any(term in text for term in ("overall survival", "mortality", "death", "os")):
        return "survival"
    if any(term in text for term in ("progression-free survival", "pfs", "time to progression")):
        return "time_to_event"
    if any(term in text for term in ("response rate", "orr", "complete response", "partial response")):
        return "response"
    if any(term in text for term in ("biomarker", "pharmacodynamic", "immune response")):
        return "biomarker"
    return "other"


def _apply_comparator_relevance(
    comparator: ComparatorBenchmarkBundle,
    relevance: ComparatorRelevanceOutput,
) -> ComparatorBenchmarkBundle:
    relevant_or_weak = tuple(item.nct_id for item in relevance.trial_relevance if item.relevance in {"relevant", "weak"})
    return comparator.model_copy(
        update={
            "comparator_trial_ids": relevant_or_weak[:5] or comparator.comparator_trial_ids,
            "benchmark_summary": f"{comparator.benchmark_summary} {relevance.relevance_summary}",
            "source_ids": tuple(dict.fromkeys((*comparator.source_ids, *relevance.source_ids))),
            "missing_data_flags": _dedupe_missing_flags((*comparator.missing_data_flags, *relevance.missing_data_flags)),
            "confidence": min(0.85, max(comparator.confidence, relevance.confidence)),
        }
    )


def _ensure_component_sources(component: BaseModel, fallback_source_ids: tuple[str, ...]) -> Any:
    source_ids = getattr(component, "source_ids", ())
    if source_ids or not fallback_source_ids:
        return component
    return component.model_copy(update={"source_ids": fallback_source_ids})


def _shared_instructions() -> str:
    return (
        "Use only supplied structured evidence and source_ids. Store concise rationale summaries only. "
        "Missing evidence must become missing_data_flags, confidence reductions, or review questions. "
        "Do not invent efficacy results, safety rates, approval probability, PoS, enrollment rates, site performance, "
        "competitor outcomes, or final clinical/investment/licensing/go-no-go decisions."
    )


def _manager_instructions() -> str:
    return (
        "You are ClinicalOutcomeManagerAgent. Coordinate Agent 3 subagents after deterministic retrieval is complete. "
        "Return only a ClinicalOutcomeManagerPlan. " + _shared_instructions()
    )


def _asset_identity_instructions() -> str:
    return (
        "You are AssetIdentityAdjudicatorAgent. Review CT.gov interventions, aliases, RxNorm output, sponsor, modality, "
        "and indication only when identity is ambiguous. Flag ambiguity instead of forcing certainty. "
        "Return AssetIdentityAdjudication. " + _shared_instructions()
    )


def _endpoint_instructions() -> str:
    return (
        "You are EndpointRiskAgent. Interpret endpoint family, surrogate versus clinical nature, hierarchy ambiguity, "
        "phase appropriateness, and credibility risks. Return EndpointRiskAssessment. Do not infer efficacy or success. "
        + _shared_instructions()
    )


def _comparator_instructions() -> str:
    return (
        "You are ComparatorRelevanceAgent. Classify deterministically retrieved CT.gov comparator candidates as relevant, "
        "weak, or excluded by indication, phase, modality/MOA, endpoint family, comparator/control, population, and design. "
        "Do not invent competitor outcomes. Return ComparatorRelevanceOutput. " + _shared_instructions()
    )


def _enrollment_instructions() -> str:
    return (
        "You are EnrollmentFeasibilityAgent. Interpret enrollment count, deterministic planned duration, sites/countries, "
        "eligibility, biomarker or prior-treatment restrictions, and operational burden. Return EnrollmentDurationRisk. "
        "Do not invent enrollment rates or site performance. " + _shared_instructions()
    )


def _safety_instructions() -> str:
    return (
        "You are SafetyContextAgent. Interpret openFDA label context and registry safety exclusions. Separate known "
        "label-derived risks from missing context. Return SafetyContext. Do not invent adverse-event rates. "
        + _shared_instructions()
    )


def _failure_mode_instructions() -> str:
    return (
        "You are FailureModeSynthesisAgent. Integrate endpoint, enrollment, comparator, safety, biology, operational, "
        "asset-identity ambiguity, missing-data, and PoS availability into FailureModeClassification categories. "
        "This is risk pattern synthesis, not an outcome prediction. " + _shared_instructions()
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
) -> tuple[ComparatorBenchmarkBundle, tuple[SourceMetadata, ...], tuple[ClinicalTrialRecord, ...]]:
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
            (),
        )
    try:
        landscape = search_trial_landscape(
            disease=condition,
            phase=phase,
            limit=_env_int("PHARMA_OS_CTGV_MAX_RESULTS", 10, minimum=1, maximum=100),
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
            (),
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
        comparators,
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


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


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
