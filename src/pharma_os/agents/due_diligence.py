"""SDK-backed Agent 4 due-diligence synthesis with deterministic fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from pharma_os.agent_runtime import AgentRuntimeConfig, StructuredAgentResult, load_agents_sdk, run_structured_agent, runtime_config_for_live_agents
from pharma_os.components.due_diligence_sections import build_asset_memo
from pharma_os.schemas import (
    AgentRunTrace,
    AssetIdentityOutput,
    AssetMemo,
    ClinicalEvidenceSummary,
    ClinicalOutcomePredictionOutput,
    ClinicalRiskSummary,
    CommercialModelOutput,
    CompetitiveLandscapeSummary,
    DiligenceRedFlag,
    DueDiligenceManagerPlan,
    DueDiligenceSynthesisOutput,
    EvidenceClaim,
    MissingDataFlag,
    PatentLOEReview,
    PricingOutput,
    RNPVOutput,
    SafetyLabelSummary,
)


@dataclass(frozen=True)
class DueDiligenceManagerResult:
    """Agent 4 manager/subagent outputs used to preserve DueDiligenceOutput shape."""

    manager_plan: DueDiligenceManagerPlan
    clinical_evidence: ClinicalEvidenceSummary
    competitive_landscape: CompetitiveLandscapeSummary
    safety_label_summary: SafetyLabelSummary
    patent_loe_review: PatentLOEReview
    red_flags: tuple[DiligenceRedFlag, ...]
    asset_memo: AssetMemo
    synthesis_outputs: tuple[DueDiligenceSynthesisOutput, ...]
    traces: tuple[AgentRunTrace, ...]
    subagent_payloads: tuple[BaseModel, ...]


def run_due_diligence_manager_agent(
    *,
    run_id: str,
    agent3_output: ClinicalOutcomePredictionOutput,
    asset: AssetIdentityOutput,
    clinical_risk: ClinicalRiskSummary,
    clinical_evidence: ClinicalEvidenceSummary,
    landscape: CompetitiveLandscapeSummary,
    safety: SafetyLabelSummary,
    patent: PatentLOEReview,
    pricing: PricingOutput,
    commercial: CommercialModelOutput,
    rnpv: RNPVOutput,
    red_flags: tuple[DiligenceRedFlag, ...],
    claims: tuple[EvidenceClaim, ...],
    assumptions: tuple[object, ...],
    missing_data_flags: tuple[MissingDataFlag, ...],
    source_ids: tuple[str, ...],
    config: AgentRuntimeConfig | None = None,
) -> DueDiligenceManagerResult:
    """Run Agent 4 manager/subagents after deterministic retrieval and math."""

    runtime_config = _agent4_runtime_config(config)
    traces: list[AgentRunTrace] = []
    payloads: list[BaseModel] = []

    manager_plan = _run_typed_agent(
        agent_name="DueDiligenceManagerAgent",
        instructions=_manager_instructions(),
        output_type=DueDiligenceManagerPlan,
        run_id=run_id,
        input_summary=f"Coordinate Agent 4 diligence synthesis for {asset.asset_name or asset.nct_id}.",
        payload=_base_payload(
            agent3_output=agent3_output,
            asset=asset,
            clinical_risk=clinical_risk,
            clinical_evidence=clinical_evidence,
            landscape=landscape,
            safety=safety,
            patent=patent,
            pricing=pricing,
            commercial=commercial,
            rnpv=rnpv,
            red_flags=red_flags,
            claims=claims,
            assumptions=assumptions,
            missing_data_flags=missing_data_flags,
            source_ids=source_ids,
        ),
        fallback_output=_fallback_manager_plan(run_id, asset, source_ids, missing_data_flags),
        source_ids=source_ids,
        confidence=0.7,
        config=runtime_config,
        rationale_summary="Coordinate Agent 4 synthesis and critique after deterministic retrieval/math.",
    )
    traces.append(manager_plan.trace)
    payloads.append(manager_plan.output)

    subagent_results = [
        _run_synthesis_agent(
            agent_name="ClinicalEvidenceSynthesisAgent",
            section="clinical_evidence",
            instructions=_clinical_evidence_instructions(),
            fallback_output=_clinical_evidence_fallback(run_id, clinical_evidence),
            run_id=run_id,
            source_ids=clinical_evidence.source_ids,
            confidence=clinical_evidence.confidence,
            payload=_base_payload(
                agent3_output=agent3_output,
                asset=asset,
                clinical_risk=clinical_risk,
                clinical_evidence=clinical_evidence,
                landscape=landscape,
                safety=safety,
                patent=patent,
                pricing=pricing,
                commercial=commercial,
                rnpv=rnpv,
                red_flags=red_flags,
                claims=claims,
                assumptions=assumptions,
                missing_data_flags=missing_data_flags,
                source_ids=source_ids,
            ),
            config=runtime_config,
        ),
        _run_synthesis_agent(
            agent_name="CompetitiveLandscapeAgent",
            section="competitive_landscape",
            instructions=_landscape_instructions(),
            fallback_output=_landscape_fallback(run_id, landscape),
            run_id=run_id,
            source_ids=landscape.source_ids,
            confidence=landscape.confidence,
            payload=_base_payload(
                agent3_output=agent3_output,
                asset=asset,
                clinical_risk=clinical_risk,
                clinical_evidence=clinical_evidence,
                landscape=landscape,
                safety=safety,
                patent=patent,
                pricing=pricing,
                commercial=commercial,
                rnpv=rnpv,
                red_flags=red_flags,
                claims=claims,
                assumptions=assumptions,
                missing_data_flags=missing_data_flags,
                source_ids=source_ids,
            ),
            config=runtime_config,
        ),
        _run_synthesis_agent(
            agent_name="SafetyDiligenceAgent",
            section="safety",
            instructions=_safety_instructions(),
            fallback_output=_safety_fallback(run_id, safety, agent3_output),
            run_id=run_id,
            source_ids=safety.source_ids or agent3_output.safety_context.source_ids,
            confidence=max(safety.confidence, agent3_output.safety_context.confidence),
            payload=_base_payload(
                agent3_output=agent3_output,
                asset=asset,
                clinical_risk=clinical_risk,
                clinical_evidence=clinical_evidence,
                landscape=landscape,
                safety=safety,
                patent=patent,
                pricing=pricing,
                commercial=commercial,
                rnpv=rnpv,
                red_flags=red_flags,
                claims=claims,
                assumptions=assumptions,
                missing_data_flags=missing_data_flags,
                source_ids=source_ids,
            ),
            config=runtime_config,
        ),
        _run_synthesis_agent(
            agent_name="IPLOECriticAgent",
            section="ip_loe",
            instructions=_ip_loe_instructions(),
            fallback_output=_ip_loe_fallback(run_id, patent),
            run_id=run_id,
            source_ids=patent.source_ids,
            confidence=patent.confidence,
            payload=_base_payload(
                agent3_output=agent3_output,
                asset=asset,
                clinical_risk=clinical_risk,
                clinical_evidence=clinical_evidence,
                landscape=landscape,
                safety=safety,
                patent=patent,
                pricing=pricing,
                commercial=commercial,
                rnpv=rnpv,
                red_flags=red_flags,
                claims=claims,
                assumptions=assumptions,
                missing_data_flags=missing_data_flags,
                source_ids=source_ids,
            ),
            config=runtime_config,
        ),
        _run_synthesis_agent(
            agent_name="CommercialAssumptionsCriticAgent",
            section="commercial_assumptions",
            instructions=_commercial_instructions(),
            fallback_output=_commercial_fallback(run_id, pricing, commercial, rnpv, assumptions),
            run_id=run_id,
            source_ids=tuple(dict.fromkeys((*pricing.source_ids, *commercial.source_ids, *rnpv.source_ids))),
            confidence=min(0.8, max(pricing.confidence, commercial.confidence, rnpv.confidence)),
            payload=_base_payload(
                agent3_output=agent3_output,
                asset=asset,
                clinical_risk=clinical_risk,
                clinical_evidence=clinical_evidence,
                landscape=landscape,
                safety=safety,
                patent=patent,
                pricing=pricing,
                commercial=commercial,
                rnpv=rnpv,
                red_flags=red_flags,
                claims=claims,
                assumptions=assumptions,
                missing_data_flags=missing_data_flags,
                source_ids=source_ids,
            ),
            config=runtime_config,
        ),
        _run_synthesis_agent(
            agent_name="DiligenceRedTeamAgent",
            section="red_team",
            instructions=_red_team_instructions(),
            fallback_output=_red_team_fallback(run_id, red_flags, missing_data_flags, source_ids),
            run_id=run_id,
            source_ids=source_ids,
            confidence=0.65,
            payload=_base_payload(
                agent3_output=agent3_output,
                asset=asset,
                clinical_risk=clinical_risk,
                clinical_evidence=clinical_evidence,
                landscape=landscape,
                safety=safety,
                patent=patent,
                pricing=pricing,
                commercial=commercial,
                rnpv=rnpv,
                red_flags=red_flags,
                claims=claims,
                assumptions=assumptions,
                missing_data_flags=missing_data_flags,
                source_ids=source_ids,
            ),
            config=runtime_config,
        ),
    ]
    synthesis_outputs = tuple(result.output for result in subagent_results)
    for result in subagent_results:
        traces.append(result.trace)
        payloads.append(result.output)

    updated_clinical = clinical_evidence.model_copy(
        update={"ctgov_summary": _by_section(synthesis_outputs, "clinical_evidence").synthesis}
    )
    updated_landscape = landscape.model_copy(
        update={"benchmark_summary": _by_section(synthesis_outputs, "competitive_landscape").synthesis}
    )
    updated_patent = patent.model_copy(
        update={"review_summary": _by_section(synthesis_outputs, "ip_loe").synthesis}
    )
    updated_red_flags = _dedupe_red_flags((*red_flags, *_by_section(synthesis_outputs, "red_team").red_flags))

    asset_memo_fallback = _asset_memo_fallback(
        run_id=run_id,
        asset=asset,
        clinical_risk=clinical_risk,
        clinical_evidence=updated_clinical,
        landscape=updated_landscape,
        safety=safety,
        patent=updated_patent,
        pricing=pricing,
        commercial=commercial,
        rnpv=rnpv,
        red_flags=updated_red_flags,
        claims=claims,
        assumptions=assumptions,
        missing_data_flags=missing_data_flags,
        synthesis_outputs=synthesis_outputs,
    )
    memo_result = _run_typed_agent(
        agent_name="AssetMemoAgent",
        instructions=_asset_memo_instructions(),
        output_type=AssetMemo,
        run_id=run_id,
        input_summary=f"Write draft asset memo for {asset.asset_name or asset.nct_id}.",
        payload={
            **_base_payload(
                agent3_output=agent3_output,
                asset=asset,
                clinical_risk=clinical_risk,
                clinical_evidence=updated_clinical,
                landscape=updated_landscape,
                safety=safety,
                patent=updated_patent,
                pricing=pricing,
                commercial=commercial,
                rnpv=rnpv,
                red_flags=updated_red_flags,
                claims=claims,
                assumptions=assumptions,
                missing_data_flags=missing_data_flags,
                source_ids=source_ids,
            ),
            "synthesis_outputs": [output.model_dump(mode="json") for output in synthesis_outputs],
        },
        fallback_output=asset_memo_fallback,
        source_ids=asset_memo_fallback.source_ids,
        confidence=asset_memo_fallback.confidence,
        config=runtime_config,
        rationale_summary="Write draft-only asset memo from deterministic evidence and Agent 4 subagent critiques.",
    )
    traces.append(memo_result.trace)
    payloads.append(memo_result.output)

    return DueDiligenceManagerResult(
        manager_plan=manager_plan.output,
        clinical_evidence=updated_clinical,
        competitive_landscape=updated_landscape,
        safety_label_summary=safety,
        patent_loe_review=updated_patent,
        red_flags=updated_red_flags,
        asset_memo=memo_result.output,
        synthesis_outputs=synthesis_outputs,
        traces=tuple(traces),
        subagent_payloads=tuple(payloads),
    )


def _agent4_runtime_config(config: AgentRuntimeConfig | None) -> AgentRuntimeConfig:
    if config is not None:
        return config
    return runtime_config_for_live_agents(disabled_provenance="pharma_os.agents.due_diligence")


def _run_synthesis_agent(
    *,
    agent_name: str,
    section: str,
    instructions: str,
    fallback_output: DueDiligenceSynthesisOutput,
    run_id: str,
    source_ids: tuple[str, ...],
    confidence: float,
    payload: dict[str, Any],
    config: AgentRuntimeConfig,
) -> StructuredAgentResult:
    return _run_typed_agent(
        agent_name=agent_name,
        instructions=instructions,
        output_type=DueDiligenceSynthesisOutput,
        run_id=run_id,
        input_summary=f"Interpret Agent 4 {section} evidence.",
        payload=payload,
        fallback_output=fallback_output,
        source_ids=source_ids,
        confidence=confidence,
        config=config,
        rationale_summary=f"{agent_name} synthesized {section} evidence without inventing facts.",
    )


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


def _base_payload(
    *,
    agent3_output: ClinicalOutcomePredictionOutput,
    asset: AssetIdentityOutput,
    clinical_risk: ClinicalRiskSummary,
    clinical_evidence: ClinicalEvidenceSummary,
    landscape: CompetitiveLandscapeSummary,
    safety: SafetyLabelSummary,
    patent: PatentLOEReview,
    pricing: PricingOutput,
    commercial: CommercialModelOutput,
    rnpv: RNPVOutput,
    red_flags: tuple[DiligenceRedFlag, ...],
    claims: tuple[EvidenceClaim, ...],
    assumptions: tuple[object, ...],
    missing_data_flags: tuple[MissingDataFlag, ...],
    source_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "agent3_context": {
            "trial_identity": agent3_output.trial_identity.model_dump(mode="json"),
            "trial_design_features": agent3_output.trial_design_features.model_dump(mode="json"),
            "endpoint_risk_assessment": agent3_output.endpoint_risk_assessment.model_dump(mode="json"),
            "enrollment_duration_risk": agent3_output.enrollment_duration_risk.model_dump(mode="json"),
            "comparator_benchmarking": agent3_output.comparator_benchmarking.model_dump(mode="json"),
            "failure_mode_classification": agent3_output.failure_mode_classification.model_dump(mode="json"),
            "safety_context": agent3_output.safety_context.model_dump(mode="json"),
        },
        "asset_identity": asset.model_dump(mode="json"),
        "clinical_risk_summary": clinical_risk.model_dump(mode="json"),
        "clinical_evidence": clinical_evidence.model_dump(mode="json"),
        "competitive_landscape": landscape.model_dump(mode="json"),
        "safety_label_summary": safety.model_dump(mode="json"),
        "patent_loe_review": patent.model_dump(mode="json"),
        "pricing": pricing.model_dump(mode="json"),
        "commercial_model": commercial.model_dump(mode="json"),
        "rnpv": rnpv.model_dump(mode="json"),
        "red_flags": [flag.model_dump(mode="json") for flag in red_flags],
        "claims": [claim.model_dump(mode="json") for claim in claims],
        "assumptions": [assumption.model_dump(mode="json") if hasattr(assumption, "model_dump") else str(assumption) for assumption in assumptions],
        "missing_data_flags": [flag.model_dump(mode="json") for flag in missing_data_flags],
        "source_ids": source_ids,
        "guardrails": (
            "Draft diligence only. Do not invent LOE, PoS, pricing, market size, eligible patients, penetration, "
            "efficacy/safety numbers, or rNPV inputs. Do not make investment, licensing, acquisition, legal, regulatory, "
            "clinical, approval, or go/no-go recommendations."
        ),
    }


def _fallback_manager_plan(
    run_id: str,
    asset: AssetIdentityOutput,
    source_ids: tuple[str, ...],
    missing_data_flags: tuple[MissingDataFlag, ...],
) -> DueDiligenceManagerPlan:
    return DueDiligenceManagerPlan(
        output_id=f"due-diligence-manager-plan-{run_id}",
        nct_id=asset.nct_id,
        ordered_steps=(
            "clinical_evidence_synthesis",
            "competitive_landscape_synthesis",
            "safety_diligence",
            "ip_loe_critique",
            "commercial_assumptions_critique",
            "diligence_red_team",
            "asset_memo_draft",
        ),
        guardrail_summary="Agent 4 interprets retrieved evidence only and escalates missing evidence to flags, limitations, and human-review questions.",
        rationale_summary="Coordinate synthesis after deterministic retrieval/math while preserving DueDiligenceOutput shape.",
        source_ids=source_ids,
        missing_data_flags=missing_data_flags,
        confidence=0.7 if not missing_data_flags else 0.55,
    )


def _clinical_evidence_fallback(run_id: str, evidence: ClinicalEvidenceSummary) -> DueDiligenceSynthesisOutput:
    limitations = tuple(flag.reason for flag in evidence.missing_data_flags)
    return DueDiligenceSynthesisOutput(
        output_id=f"clinical-evidence-synthesis-{run_id}",
        agent_name="ClinicalEvidenceSynthesisAgent",
        section="clinical_evidence",
        synthesis=f"{evidence.ctgov_summary} PubMed metadata count: {evidence.pubmed_article_count}; titles: {', '.join(evidence.pubmed_titles) or 'none retrieved'}.",
        limitations=limitations,
        review_questions=("Which PubMed snippets, if any, support efficacy or safety interpretation without full-text overreach?",),
        source_ids=evidence.source_ids,
        missing_data_flags=evidence.missing_data_flags,
        confidence=evidence.confidence,
    )


def _landscape_fallback(run_id: str, landscape: CompetitiveLandscapeSummary) -> DueDiligenceSynthesisOutput:
    return DueDiligenceSynthesisOutput(
        output_id=f"competitive-landscape-synthesis-{run_id}",
        agent_name="CompetitiveLandscapeAgent",
        section="competitive_landscape",
        synthesis=landscape.benchmark_summary,
        limitations=tuple(flag.reason for flag in landscape.missing_data_flags),
        review_questions=("Which comparator trials are sufficiently relevant to diligence conclusions?",),
        source_ids=landscape.source_ids,
        missing_data_flags=landscape.missing_data_flags,
        confidence=landscape.confidence,
    )


def _safety_fallback(
    run_id: str,
    safety: SafetyLabelSummary,
    agent3_output: ClinicalOutcomePredictionOutput,
) -> DueDiligenceSynthesisOutput:
    known = "openFDA label safety context is available." if safety.label_available else "openFDA label safety context is missing or unavailable."
    return DueDiligenceSynthesisOutput(
        output_id=f"safety-diligence-synthesis-{run_id}",
        agent_name="SafetyDiligenceAgent",
        section="safety",
        synthesis=f"{known} Agent 3 safety context: {agent3_output.safety_context.summary or 'not available'}.",
        limitations=tuple(flag.reason for flag in (*safety.missing_data_flags, *agent3_output.safety_context.missing_data_flags)),
        review_questions=("Which label-derived safety issues require clinical review?",),
        source_ids=tuple(dict.fromkeys((*safety.source_ids, *agent3_output.safety_context.source_ids))),
        missing_data_flags=tuple((*safety.missing_data_flags, *agent3_output.safety_context.missing_data_flags)),
        confidence=max(safety.confidence, agent3_output.safety_context.confidence),
    )


def _ip_loe_fallback(run_id: str, patent: PatentLOEReview) -> DueDiligenceSynthesisOutput:
    return DueDiligenceSynthesisOutput(
        output_id=f"ip-loe-critique-{run_id}",
        agent_name="IPLOECriticAgent",
        section="ip_loe",
        synthesis=patent.review_summary,
        limitations=tuple(flag.reason for flag in patent.missing_data_flags),
        review_questions=("Does LOE support require human IP counsel review before diligence use?",),
        source_ids=patent.source_ids,
        missing_data_flags=patent.missing_data_flags,
        confidence=patent.confidence,
    )


def _commercial_fallback(
    run_id: str,
    pricing: PricingOutput,
    commercial: CommercialModelOutput,
    rnpv: RNPVOutput,
    assumptions: tuple[object, ...],
) -> DueDiligenceSynthesisOutput:
    assumption_lines = tuple(
        f"{getattr(assumption, 'name', 'assumption')}: {getattr(assumption, 'assumption_type', 'unknown')}"
        for assumption in assumptions
    )
    flags = tuple((*pricing.missing_data_flags, *commercial.missing_data_flags, *rnpv.missing_data_flags))
    return DueDiligenceSynthesisOutput(
        output_id=f"commercial-assumptions-critique-{run_id}",
        agent_name="CommercialAssumptionsCriticAgent",
        section="commercial_assumptions",
        synthesis=(
            f"Pricing annual WAC: {pricing.annual_wac if pricing.annual_wac is not None else 'not available'}; "
            f"commercial calculable: {commercial.calculable}; rNPV calculable: {rnpv.calculable}. "
            f"Assumption basis: {'; '.join(assumption_lines) or 'none recorded'}."
        ),
        limitations=tuple(flag.reason for flag in flags),
        review_questions=("Which commercial and rNPV assumptions drive diligence sensitivity?",),
        source_ids=tuple(dict.fromkeys((*pricing.source_ids, *commercial.source_ids, *rnpv.source_ids))),
        missing_data_flags=flags,
        confidence=min(0.8, max(pricing.confidence, commercial.confidence, rnpv.confidence)),
    )


def _red_team_fallback(
    run_id: str,
    red_flags: tuple[DiligenceRedFlag, ...],
    missing_data_flags: tuple[MissingDataFlag, ...],
    source_ids: tuple[str, ...],
) -> DueDiligenceSynthesisOutput:
    return DueDiligenceSynthesisOutput(
        output_id=f"diligence-red-team-{run_id}",
        agent_name="DiligenceRedTeamAgent",
        section="red_team",
        synthesis=f"Red-team review found {len(red_flags)} existing red flags and {len(missing_data_flags)} missing-data flags requiring human review.",
        limitations=tuple(flag.reason for flag in missing_data_flags),
        review_questions=("Where does the memo risk overconfidence or decision-like language?",),
        red_flags=red_flags,
        source_ids=source_ids,
        missing_data_flags=missing_data_flags,
        confidence=0.65,
    )


def _asset_memo_fallback(
    *,
    run_id: str,
    asset: AssetIdentityOutput,
    clinical_risk: ClinicalRiskSummary,
    clinical_evidence: ClinicalEvidenceSummary,
    landscape: CompetitiveLandscapeSummary,
    safety: SafetyLabelSummary,
    patent: PatentLOEReview,
    pricing: PricingOutput,
    commercial: CommercialModelOutput,
    rnpv: RNPVOutput,
    red_flags: tuple[DiligenceRedFlag, ...],
    claims: tuple[EvidenceClaim, ...],
    assumptions: tuple[object, ...],
    missing_data_flags: tuple[MissingDataFlag, ...],
    synthesis_outputs: tuple[DueDiligenceSynthesisOutput, ...],
) -> AssetMemo:
    memo = build_asset_memo(
        run_id=run_id,
        asset=asset,
        clinical_risk=clinical_risk,
        evidence=clinical_evidence,
        landscape=landscape,
        safety=safety,
        patent=patent,
        pricing=pricing,
        commercial=commercial,
        rnpv=rnpv,
        red_flags=red_flags,
        claims=claims,
        assumptions=assumptions,  # type: ignore[arg-type]
        missing_data_flags=missing_data_flags,
    )
    synthesis_sections = tuple(f"{output.agent_name}: {output.synthesis}" for output in synthesis_outputs)
    questions = tuple(dict.fromkeys((*memo.review_questions, *(question for output in synthesis_outputs for question in output.review_questions))))
    return memo.model_copy(update={"sections": (*memo.sections, *synthesis_sections), "review_questions": questions})


def _by_section(outputs: tuple[DueDiligenceSynthesisOutput, ...], section: str) -> DueDiligenceSynthesisOutput:
    for output in outputs:
        if output.section == section:
            return output
    raise ValueError(f"missing due diligence synthesis section: {section}")


def _dedupe_red_flags(flags: tuple[DiligenceRedFlag, ...]) -> tuple[DiligenceRedFlag, ...]:
    deduped: dict[str, DiligenceRedFlag] = {}
    for flag in flags:
        deduped[flag.flag_id] = flag
    return tuple(deduped.values())


def _shared_guardrails() -> str:
    return (
        "You are part of PharmaOS Agent 4 due diligence. Return only the requested strict structured output. "
        "Use supplied sources and values only. Do not invent LOE, PoS, pricing, market size, eligible patients, penetration, "
        "efficacy/safety numbers, or rNPV inputs. Do not make final business, legal, regulatory, clinical, approval, "
        "licensing, acquisition, investment, or go/no-go recommendations. Missing evidence becomes flags, limitations, "
        "confidence reductions, red flags, or human-review questions."
    )


def _manager_instructions() -> str:
    return _shared_guardrails() + " Coordinate subagents after deterministic retrieval and math. Preserve DueDiligenceOutput shape."


def _clinical_evidence_instructions() -> str:
    return _shared_guardrails() + " Synthesize CT.gov and PubMed metadata/abstract snippets only; do not claim full-text findings unless snippets provide them."


def _landscape_instructions() -> str:
    return _shared_guardrails() + " Interpret Agent 3 comparator context and public CT.gov landscape without proprietary sources or invented outcomes."


def _safety_instructions() -> str:
    return _shared_guardrails() + " Separate label-derived safety issues, missing safety context, and clinical review questions. Do not invent adverse event rates."


def _ip_loe_instructions() -> str:
    return _shared_guardrails() + " Review Lens candidates and LOE support. No legal conclusions, FTO opinions, or invented LOE."


def _commercial_instructions() -> str:
    return _shared_guardrails() + " Critique PoS, pricing, commercial model, rNPV, and assumptions. Do not calculate new rNPV or invent values."


def _red_team_instructions() -> str:
    return _shared_guardrails() + " Identify unsupported claims, overconfidence, missing sources, gaps, and decision-like language."


def _asset_memo_instructions() -> str:
    return _shared_guardrails() + " Write a draft AssetMemo only, with human-review questions and no final recommendations."
