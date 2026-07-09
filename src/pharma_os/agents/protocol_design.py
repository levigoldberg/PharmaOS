"""Bounded helper subagents for the Protocol Design Brief workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from pharma_os.agent_runtime import AgentRuntimeConfig, StructuredAgentResult, load_agents_sdk, run_structured_agent, runtime_config_for_live_agents
from pharma_os.schemas import (
    AnalogCandidateRecord,
    AnalogBenchmarkBundle,
    AnalogSearchPlanOutput,
    AnalogTrialSelectionOutput,
    AgentRunTrace,
    BenchmarkInterpretation,
    CTGovSearchQuery,
    ClinicalOutcomePredictionOutput,
    ClinicalTrialRecord,
    DueDiligenceOutput,
    ExcludedAnalogTrial,
    MissingDataFlag,
    ProtocolDesignBrief,
    ProtocolDesignManagerPlan,
    ProtocolReviewerCritique,
    ProtocolSectionAgentOutput,
    ProtocolSectionDraft,
    SelectedAnalogTrial,
)
from pharma_os.tools._due_diligence_common import norm
from pharma_os.tools.protocol_design import build_benchmark_summary, build_protocol_design_brief


@dataclass(frozen=True)
class ProtocolDesignManagerResult:
    """All Agent 5 manager/subagent pieces needed by the workflow."""

    manager_plan: ProtocolDesignManagerPlan
    search_plan: AnalogSearchPlanOutput
    analog_candidates: tuple[AnalogCandidateRecord, ...]
    analog_sources: tuple[object, ...]
    retrieval_flags: tuple[MissingDataFlag, ...]
    selection: AnalogTrialSelectionOutput
    benchmark_bundle: AnalogBenchmarkBundle
    benchmark_interpretation: BenchmarkInterpretation
    section_outputs: tuple[ProtocolSectionAgentOutput, ...]
    reviewer_critique: ProtocolReviewerCritique
    protocol_design_brief: ProtocolDesignBrief
    traces: tuple[AgentRunTrace, ...]
    subagent_payloads: tuple[BaseModel, ...]


SearchExecutor = Callable[
    [AnalogSearchPlanOutput, str],
    tuple[tuple[AnalogCandidateRecord, ...], tuple[object, ...], tuple[MissingDataFlag, ...]],
]

BenchmarkCalculator = Callable[
    [ClinicalTrialRecord, tuple[AnalogCandidateRecord, ...], AnalogTrialSelectionOutput, AnalogSearchPlanOutput],
    AnalogBenchmarkBundle,
]


def run_protocol_design_manager_agent(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
    source_ids: tuple[str, ...],
    assumptions: tuple[object, ...],
    missing_data_flags: tuple[MissingDataFlag, ...],
    claims: tuple[object, ...],
    top_k: int,
    execute_search_plan: SearchExecutor,
    calculate_benchmark: BenchmarkCalculator,
    config: AgentRuntimeConfig | None = None,
) -> ProtocolDesignManagerResult:
    """Run the Agent 5 manager and scoped subagents with deterministic fallbacks."""

    runtime_config = _agent5_runtime_config(config)
    traces: list[AgentRunTrace] = []
    payloads: list[BaseModel] = []

    manager_plan = _run_typed_agent(
        agent_name="ProtocolDesignManagerAgent",
        instructions=_manager_instructions(),
        output_type=ProtocolDesignManagerPlan,
        run_id=run_id,
        input_summary=f"Coordinate Agent 5 for {target_trial.nct_id}.",
        payload=_base_payload(target_trial, agent3_output, agent4_output, source_ids),
        fallback_output=_fallback_manager_plan(run_id, target_trial, source_ids, missing_data_flags),
        source_ids=source_ids,
        confidence=0.7,
        config=runtime_config,
        rationale_summary="Coordinate typed Agent 5 subagents and deterministic benchmark steps.",
    )
    traces.append(manager_plan.trace)
    payloads.append(manager_plan.output)

    search_plan = _run_typed_agent(
        agent_name="AnalogSearchPlannerAgent",
        instructions=_search_planner_instructions(),
        output_type=AnalogSearchPlanOutput,
        run_id=run_id,
        input_summary=f"Plan CT.gov analog searches for {target_trial.nct_id}.",
        payload=_base_payload(target_trial, agent3_output, agent4_output, source_ids),
        fallback_output=build_search_strategy(
            run_id=run_id,
            target_trial=target_trial,
            agent3_output=agent3_output,
            agent4_output=agent4_output,
        ),
        source_ids=source_ids,
        confidence=0.7,
        config=runtime_config,
        rationale_summary="Plan bounded CT.gov searches only; deterministic code executes them.",
    )
    traces.append(search_plan.trace)
    payloads.append(search_plan.output)

    analog_candidates, analog_sources, retrieval_flags = execute_search_plan(search_plan.output, target_trial.nct_id)
    selection = _run_typed_agent(
        agent_name="AnalogSelectionAgent",
        instructions=_analog_selection_instructions(),
        output_type=AnalogTrialSelectionOutput,
        run_id=run_id,
        input_summary=f"Select relevant analogs for {target_trial.nct_id} from {len(analog_candidates)} candidates.",
        payload={
            **_base_payload(target_trial, agent3_output, agent4_output, source_ids),
            "search_plan": search_plan.output.model_dump(mode="json"),
            "analog_candidates": [candidate.model_dump(mode="json") for candidate in analog_candidates],
            "top_k": top_k,
        },
        fallback_output=select_analog_trials(
            run_id=run_id,
            target_trial=target_trial,
            candidates=analog_candidates,
            agent3_output=agent3_output,
            agent4_output=agent4_output,
            search_plan=search_plan.output,
            top_k=top_k,
        ),
        source_ids=tuple(dict.fromkeys(source_id for candidate in analog_candidates for source_id in candidate.source_ids)),
        confidence=0.7 if analog_candidates else 0.25,
        config=runtime_config,
        rationale_summary="Select analogs by match dimensions, not by retrieval alone.",
    )
    traces.append(selection.trace)
    payloads.append(selection.output)

    benchmark_bundle = calculate_benchmark(target_trial, analog_candidates, selection.output, search_plan.output)
    source_ids = tuple(dict.fromkeys((*source_ids, *benchmark_bundle.source_ids)))
    benchmark_interpretation = _run_typed_agent(
        agent_name="AnalogBenchmarkInterpreterAgent",
        instructions=_benchmark_interpreter_instructions(),
        output_type=BenchmarkInterpretation,
        run_id=run_id,
        input_summary=f"Interpret deterministic benchmark bundle {benchmark_bundle.bundle_id}.",
        payload={
            **_base_payload(target_trial, agent3_output, agent4_output, source_ids),
            "benchmark_bundle": benchmark_bundle.model_dump(mode="json"),
        },
        fallback_output=_fallback_benchmark_interpretation(run_id, target_trial, benchmark_bundle),
        source_ids=benchmark_bundle.source_ids,
        confidence=benchmark_bundle.confidence,
        config=runtime_config,
        rationale_summary="Interpret benchmark patterns without inventing new numbers.",
    )
    traces.append(benchmark_interpretation.trace)
    payloads.append(benchmark_interpretation.output)

    section_outputs = tuple(
        _run_section_agent(
            run_id=run_id,
            agent_name=agent_name,
            instructions=instructions,
            fallback_output=fallback_output,
            target_trial=target_trial,
            agent3_output=agent3_output,
            agent4_output=agent4_output,
            benchmark_bundle=benchmark_bundle,
            benchmark_interpretation=benchmark_interpretation.output,
            source_ids=source_ids,
            config=runtime_config,
        )
        for agent_name, instructions, fallback_output in _section_agent_specs(
            run_id=run_id,
            target_trial=target_trial,
            agent3_output=agent3_output,
            agent4_output=agent4_output,
            benchmark_bundle=benchmark_bundle,
            benchmark_interpretation=benchmark_interpretation.output,
            source_ids=source_ids,
        )
    )
    for result in section_outputs:
        traces.append(result.trace)
        payloads.append(result.output)
    section_payloads = tuple(result.output for result in section_outputs)

    draft_sections = _sections_by_brief_field(section_payloads)
    reviewer_critique = _run_typed_agent(
        agent_name="RegulatoryCriticAgent",
        instructions=_regulatory_critic_instructions(),
        output_type=ProtocolReviewerCritique,
        run_id=run_id,
        input_summary=f"Red-team Agent 5 draft sections for {target_trial.nct_id}.",
        payload={
            "sections": [section.model_dump(mode="json") for output in section_payloads for section in output.sections],
            "benchmark_interpretation": benchmark_interpretation.output.model_dump(mode="json"),
            "agent3_risks": agent3_output.failure_mode_classification.model_dump(mode="json"),
            "agent4_red_flags": [flag.model_dump(mode="json") for flag in agent4_output.red_flags],
            "missing_data_flags": [flag.model_dump(mode="json") for flag in missing_data_flags],
        },
        fallback_output=review_protocol_design(
            run_id=run_id,
            source_ids=source_ids,
            analog_limitations=benchmark_bundle.limitations,
            agent3_output=agent3_output,
            agent4_output=agent4_output,
        ),
        source_ids=source_ids,
        confidence=0.6,
        config=runtime_config,
        rationale_summary="Critique only; do not approve or invent facts.",
    )
    traces.append(reviewer_critique.trace)
    payloads.append(reviewer_critique.output)

    fallback_brief = build_protocol_design_brief(
        run_id=run_id,
        target_trial=target_trial,
        strategy_sections=draft_sections["strategy_sections"],
        eligibility_sections=draft_sections["eligibility_sections"],
        reviewer_critique=reviewer_critique.output,
        benchmark_bundle=benchmark_bundle,
        claims=claims,  # type: ignore[arg-type]
        assumptions=assumptions,  # type: ignore[arg-type]
        missing_data_flags=missing_data_flags,
        source_ids=source_ids,
    )
    brief = _run_typed_agent(
        agent_name="ProtocolBriefWriterAgent",
        instructions=_brief_writer_instructions(),
        output_type=ProtocolDesignBrief,
        run_id=run_id,
        input_summary=f"Assemble draft ProtocolDesignBrief for {target_trial.nct_id}.",
        payload={
            "sections": [section.model_dump(mode="json") for output in section_payloads for section in output.sections],
            "reviewer_critique": reviewer_critique.output.model_dump(mode="json"),
            "benchmark_interpretation": benchmark_interpretation.output.model_dump(mode="json"),
            "claims": [getattr(claim, "model_dump", lambda **_: claim)(mode="json") if hasattr(claim, "model_dump") else str(claim) for claim in claims],
            "missing_data_flags": [flag.model_dump(mode="json") for flag in missing_data_flags],
        },
        fallback_output=fallback_brief,
        source_ids=source_ids,
        confidence=fallback_brief.confidence,
        config=runtime_config,
        rationale_summary="Assemble the draft strategy brief with human-review framing.",
    )
    traces.append(brief.trace)
    payloads.append(brief.output)

    return ProtocolDesignManagerResult(
        manager_plan=manager_plan.output,
        search_plan=search_plan.output,
        analog_candidates=analog_candidates,
        analog_sources=analog_sources,
        retrieval_flags=retrieval_flags,
        selection=selection.output,
        benchmark_bundle=benchmark_bundle,
        benchmark_interpretation=benchmark_interpretation.output,
        section_outputs=section_payloads,
        reviewer_critique=reviewer_critique.output,
        protocol_design_brief=brief.output,
        traces=tuple(traces),
        subagent_payloads=tuple(payloads),
    )


def _agent5_runtime_config(config: AgentRuntimeConfig | None) -> AgentRuntimeConfig:
    if config is not None:
        return config
    return runtime_config_for_live_agents(disabled_provenance="pharma_os.agents.protocol_design")


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


def _run_section_agent(
    *,
    run_id: str,
    agent_name: str,
    instructions: str,
    fallback_output: ProtocolSectionAgentOutput,
    target_trial: ClinicalTrialRecord,
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
    benchmark_bundle: AnalogBenchmarkBundle,
    benchmark_interpretation: BenchmarkInterpretation,
    source_ids: tuple[str, ...],
    config: AgentRuntimeConfig,
) -> StructuredAgentResult:
    return _run_typed_agent(
        agent_name=agent_name,
        instructions=instructions,
        output_type=ProtocolSectionAgentOutput,
        run_id=run_id,
        input_summary=f"Draft source-grounded strategy sections for {target_trial.nct_id}.",
        payload={
            **_base_payload(target_trial, agent3_output, agent4_output, source_ids),
            "benchmark_bundle": benchmark_bundle.model_dump(mode="json"),
            "benchmark_interpretation": benchmark_interpretation.model_dump(mode="json"),
        },
        fallback_output=fallback_output,
        source_ids=fallback_output.source_ids or source_ids,
        confidence=fallback_output.confidence,
        config=config,
        rationale_summary=f"{agent_name} produced draft strategy sections and review questions.",
    )


def _base_payload(
    target_trial: ClinicalTrialRecord,
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
    source_ids: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "target_trial": target_trial.model_dump(mode="json"),
        "agent3_context": {
            "trial_identity": agent3_output.trial_identity.model_dump(mode="json"),
            "asset_identity": agent3_output.asset_identity.model_dump(mode="json"),
            "trial_design_features": agent3_output.trial_design_features.model_dump(mode="json"),
            "endpoint_risk_assessment": agent3_output.endpoint_risk_assessment.model_dump(mode="json"),
            "enrollment_duration_risk": agent3_output.enrollment_duration_risk.model_dump(mode="json"),
            "failure_mode_classification": agent3_output.failure_mode_classification.model_dump(mode="json"),
            "safety_context": agent3_output.safety_context.model_dump(mode="json"),
            "missing_data_flags": [flag.model_dump(mode="json") for flag in agent3_output.missing_data_flags],
        },
        "agent4_context": {
            "asset_identity": agent4_output.asset_identity.model_dump(mode="json"),
            "clinical_risk_summary": agent4_output.clinical_risk_summary.model_dump(mode="json"),
            "competitive_landscape": agent4_output.competitive_landscape.model_dump(mode="json"),
            "safety_label_summary": agent4_output.safety_label_summary.model_dump(mode="json"),
            "red_flags": [flag.model_dump(mode="json") for flag in agent4_output.red_flags],
            "missing_data_flags": [flag.model_dump(mode="json") for flag in agent4_output.missing_data_flags],
        },
        "source_ids": source_ids,
        "guardrails": (
            "Draft strategy artifact only. Do not write final protocol language, IRB-ready language, "
            "submission-ready language, enrollment plans, final approval, go/no-go decisions, invented sample size, "
            "power, effect size, efficacy, safety, patient, site, or enrollment facts."
        ),
    }


def _fallback_manager_plan(
    run_id: str,
    target_trial: ClinicalTrialRecord,
    source_ids: tuple[str, ...],
    missing_data_flags: tuple[MissingDataFlag, ...],
) -> ProtocolDesignManagerPlan:
    return ProtocolDesignManagerPlan(
        output_id=f"protocol-design-manager-plan-{run_id}",
        target_nct_id=target_trial.nct_id,
        ordered_steps=(
            "plan_ctgov_analog_search",
            "execute_ctgov_search_deterministically",
            "select_analogs",
            "calculate_benchmark_deterministically",
            "interpret_benchmark",
            "draft_strategy_sections",
            "red_team_sections",
            "assemble_draft_brief",
        ),
        source_ids=source_ids,
        missing_data_flags=missing_data_flags,
        guardrail_summary="Agent 5 produces a draft strategy brief only and escalates missing or ambiguous evidence to flags and human-review questions.",
        rationale_summary="Coordinate ambiguous protocol reasoning through scoped subagents while preserving deterministic retrieval, math, validation, persistence, and gates.",
        confidence=0.7 if not missing_data_flags else 0.55,
    )


def _fallback_benchmark_interpretation(
    run_id: str,
    target_trial: ClinicalTrialRecord,
    benchmark_bundle: AnalogBenchmarkBundle,
) -> BenchmarkInterpretation:
    common_patterns = []
    if benchmark_bundle.randomized_frequency:
        common_patterns.append(f"Randomization pattern most often detected as {benchmark_bundle.randomized_frequency[0].label}.")
    if benchmark_bundle.comparator_categories:
        common_patterns.append(f"Comparator category most often detected as {benchmark_bundle.comparator_categories[0].label}.")
    if benchmark_bundle.primary_endpoint_family_frequency:
        common_patterns.append(f"Primary endpoint family most often detected as {benchmark_bundle.primary_endpoint_family_frequency[0].label}.")
    weak = tuple(benchmark_bundle.limitations) or ("Benchmark interpretation is limited by available CT.gov fields.",)
    return BenchmarkInterpretation(
        output_id=f"benchmark-interpretation-{run_id}",
        target_nct_id=target_trial.nct_id,
        common_design_patterns=tuple(common_patterns) or ("No dominant analog design pattern was confidently identified.",),
        target_alignment=("Target-trial alignment should be reviewed against selected analog endpoint, comparator, and population patterns.",),
        target_misalignment=("No source-backed target misalignment should be treated as final without human review.",),
        strategy_implications=("Use benchmark patterns as protocol-team review inputs, not as final design recommendations.",),
        weak_or_incomplete_findings=weak,
        human_review_questions=(
            "Which benchmark patterns are strong enough to influence protocol strategy?",
            "Which analog limitations should reduce confidence in protocol design choices?",
        ),
        source_ids=benchmark_bundle.source_ids,
        confidence=benchmark_bundle.confidence,
    )


def _section_agent_specs(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
    benchmark_bundle: AnalogBenchmarkBundle,
    benchmark_interpretation: BenchmarkInterpretation,
    source_ids: tuple[str, ...],
) -> tuple[tuple[str, str, ProtocolSectionAgentOutput], ...]:
    del benchmark_interpretation
    strategy_sections = build_protocol_strategy_sections(
        run_id=run_id,
        target_trial=target_trial,
        source_ids=source_ids,
        benchmark_summary=build_benchmark_summary(benchmark_bundle),
        agent3_output=agent3_output,
        agent4_output=agent4_output,
    )
    eligibility_sections = build_eligibility_and_schedule_sections(
        run_id=run_id,
        source_ids=benchmark_bundle.source_ids or source_ids,
        inclusion_themes=benchmark_bundle.inclusion_themes,
        exclusion_themes=benchmark_bundle.exclusion_themes,
        safety_themes=benchmark_bundle.safety_exclusion_themes,
    )
    return (
        (
            "EndpointStrategyAgent",
            _endpoint_strategy_instructions(),
            ProtocolSectionAgentOutput(
                output_id=f"endpoint-strategy-output-{run_id}",
                agent_name="EndpointStrategyAgent",
                sections=(strategy_sections["analog_trial_benchmark_summary"], strategy_sections["endpoint_strategy"]),
                human_review_questions=(
                    "Does the endpoint family align with selected analog precedent?",
                    "Which endpoint hierarchy questions require statistical review?",
                ),
                source_ids=source_ids,
                confidence=0.65,
            ),
        ),
        (
            "PopulationEligibilityAgent",
            _population_eligibility_instructions(),
            ProtocolSectionAgentOutput(
                output_id=f"population-eligibility-output-{run_id}",
                agent_name="PopulationEligibilityAgent",
                sections=(
                    strategy_sections["target_population"],
                    eligibility_sections["draft_eligibility_framework"],
                    eligibility_sections["draft_schedule_of_assessments_framework"],
                ),
                human_review_questions=(
                    "Which eligibility themes are source-backed enough to carry into protocol drafting?",
                    "Which schedule-burden concerns need clinical operations review?",
                ),
                missing_data_flags=benchmark_bundle.missing_data_flags,
                source_ids=source_ids,
                confidence=0.6,
            ),
        ),
        (
            "ComparatorDesignAgent",
            _comparator_design_instructions(),
            ProtocolSectionAgentOutput(
                output_id=f"comparator-design-output-{run_id}",
                agent_name="ComparatorDesignAgent",
                sections=(
                    strategy_sections["executive_synopsis"],
                    strategy_sections["strategic_rationale"],
                    strategy_sections["study_design"],
                    strategy_sections["comparator_and_landscape_rationale"],
                ),
                human_review_questions=(
                    "Which comparator/control options should remain human-review alternatives?",
                    "Where is Agent 4 landscape context insufficient for comparator strategy?",
                ),
                source_ids=source_ids,
                confidence=0.6,
            ),
        ),
        (
            "SafetyMonitoringAgent",
            _safety_monitoring_instructions(),
            ProtocolSectionAgentOutput(
                output_id=f"safety-monitoring-output-{run_id}",
                agent_name="SafetyMonitoringAgent",
                sections=(strategy_sections["safety_monitoring_outline"],),
                human_review_questions=("Which openFDA and analog safety signals need clinical safety review?",),
                source_ids=source_ids,
                confidence=0.55,
            ),
        ),
        (
            "StatisticalSkeletonAgent",
            _statistical_skeleton_instructions(),
            ProtocolSectionAgentOutput(
                output_id=f"statistical-skeleton-output-{run_id}",
                agent_name="StatisticalSkeletonAgent",
                sections=(
                    strategy_sections["statistical_analysis_skeleton"],
                    strategy_sections["operational_feasibility_risks"],
                    strategy_sections["regulatory_standards_considerations"],
                ),
                human_review_questions=(
                    "Which estimand, analysis population, multiplicity, interim, and missing-data questions require biostatistics review?",
                    "Which benchmark limitations should be carried into regulatory review?",
                ),
                source_ids=source_ids,
                confidence=0.55,
            ),
        ),
    )


def _sections_by_brief_field(outputs: tuple[ProtocolSectionAgentOutput, ...]) -> dict[str, dict[str, ProtocolSectionDraft]]:
    by_title = {section.title: section for output in outputs for section in output.sections}
    return {
        "strategy_sections": {
            "executive_synopsis": by_title["Executive Synopsis"],
            "strategic_rationale": by_title["Strategic Rationale"],
            "analog_trial_benchmark_summary": by_title["Analog Trial Benchmark Summary"],
            "target_population": by_title["Target Population"],
            "study_design": by_title["Study Design"],
            "comparator_and_landscape_rationale": by_title["Comparator And Landscape Rationale"],
            "endpoint_strategy": by_title["Endpoint Strategy"],
            "safety_monitoring_outline": by_title["Safety Monitoring Outline"],
            "statistical_analysis_skeleton": by_title["Statistical Analysis Skeleton"],
            "operational_feasibility_risks": by_title["Operational Feasibility Risks"],
            "regulatory_standards_considerations": by_title["Regulatory Standards Considerations"],
        },
        "eligibility_sections": {
            "draft_eligibility_framework": by_title["Draft Eligibility Framework"],
            "draft_schedule_of_assessments_framework": by_title["Draft Schedule Of Assessments Framework"],
        },
    }


def _shared_guardrails() -> str:
    return (
        "You are part of PharmaOS Agent 5. Return only the requested strict structured output. "
        "Use only supplied source IDs and facts. Do not invent missing facts. Missing or ambiguous evidence becomes "
        "missing_data_flags, weak findings, confidence reductions, limitations, or human-review questions. "
        "This is a draft protocol strategy brief, not a full protocol, IRB-ready protocol, submission-ready protocol, "
        "enrollment plan, approval decision, go/no-go decision, or final design recommendation. Do not invent sample size, "
        "power, effect size, efficacy, safety, patient, site, or enrollment numbers."
    )


def _manager_instructions() -> str:
    return _shared_guardrails() + (
        " Coordinate the Agent 5 workflow after Agent 3 and Agent 4 handoffs. "
        "List ordered steps, guardrails, missing context, and a short rationale for the typed subagent sequence."
    )


def _search_planner_instructions() -> str:
    return _shared_guardrails() + (
        " Create bounded ClinicalTrials.gov analog search queries only. Do not execute searches or select analogs. "
        "Cover same indication/phase, adjacent phase when justified, modality or target/MOA when known, endpoint family, "
        "comparator/control, biomarker-defined population, and line/prior-treatment setting when detectable. "
        "If a dimension is unknown, expose it as unknown rather than guessing."
    )


def _analog_selection_instructions() -> str:
    return _shared_guardrails() + (
        " Select analog trials from retrieved CT.gov candidates. Select and exclude with reasons. "
        "Use match/mismatch/unknown dimensions across indication, phase, modality/target/MOA, endpoint family, comparator, "
        "biomarker/line, population, study design, and data completeness. Do not choose analogs only because they appeared in search results."
    )


def _benchmark_interpreter_instructions() -> str:
    return _shared_guardrails() + (
        " Interpret deterministic benchmark metrics. Do not invent new benchmark numbers. Explain common design patterns, "
        "target alignment/misalignment, findings that matter for strategy, weak/incomplete findings, and human-review questions."
    )


def _endpoint_strategy_instructions() -> str:
    return _shared_guardrails() + (
        " Interpret endpoint precedent, endpoint family, hierarchy issues, surrogate versus clinical endpoint risk, and Agent 3 endpoint risk. "
        "Output draft endpoint strategy considerations and statistical review questions, not final endpoints."
    )


def _population_eligibility_instructions() -> str:
    return _shared_guardrails() + (
        " Interpret analog inclusion/exclusion themes and target trial eligibility. Output a draft eligibility framework, missing evidence, "
        "schedule-burden considerations, and review questions. Do not create final inclusion/exclusion criteria."
    )


def _comparator_design_instructions() -> str:
    return _shared_guardrails() + (
        " Interpret comparator/control precedent from analogs and Agent 4 landscape context. Output comparator rationale questions and tradeoffs. "
        "Do not declare a comparator acceptable."
    )


def _safety_monitoring_instructions() -> str:
    return _shared_guardrails() + (
        " Use openFDA/Agent 4 safety context and analog safety exclusion themes. Output safety monitoring considerations only. "
        "Do not create a final safety monitoring plan."
    )


def _statistical_skeleton_instructions() -> str:
    return _shared_guardrails() + (
        " Draft statistical strategy questions around estimand, analysis population, multiplicity, interim analysis, missing data, "
        "and endpoint hierarchy. Do not invent sample size, power, effect size, alpha allocation, or final SAP language."
    )


def _regulatory_critic_instructions() -> str:
    return _shared_guardrails() + (
        " Red-team the draft strategy. Identify clinical weaknesses, statistical weaknesses, regulatory questions, source gaps, "
        "overclaiming risk, places that sound too final, and areas requiring human clinical/statistical/regulatory review. Do not approve."
    )


def _brief_writer_instructions() -> str:
    return _shared_guardrails() + (
        " Assemble the final ProtocolDesignBrief from supplied typed sections and critique. Preserve source IDs, human-review framing, "
        "missing evidence, and review questions. Avoid final protocol language and avoid saying recommended design unless framed as a human-review option."
    )



def build_search_strategy(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
) -> AnalogSearchPlanOutput:
    """Build a structured CT.gov search plan without calling the API."""

    indication = (
        agent4_output.asset_identity.normalized_indication
        or agent3_output.asset_identity.normalized_indication
        or (target_trial.conditions[0] if target_trial.conditions else None)
    )
    if not indication:
        indication = "unknown condition"
    phase = target_trial.phases[0] if target_trial.phases else agent3_output.historical_pos_estimate.current_phase
    asset = agent4_output.asset_identity.asset_name or agent3_output.asset_identity.asset_name
    modality = agent4_output.asset_identity.modality or agent3_output.asset_identity.modality
    endpoint_family = _endpoint_family(target_trial)
    comparator = _comparator_hint(target_trial)
    biomarker_or_line = _biomarker_or_line(target_trial)

    queries = [
        CTGovSearchQuery(
            query_id=f"pdq-{run_id}-condition-phase",
            condition=indication,
            phase=phase,
            endpoint_family=endpoint_family,
            comparator=comparator,
            biomarker_or_line=biomarker_or_line,
            limit=25,
            expected_analog_dimension="same indication and phase",
            rationale="Primary analog search anchored to target indication and phase from CT.gov and upstream handoffs.",
        )
    ]
    if modality and modality != "unknown":
        queries.append(
            CTGovSearchQuery(
                query_id=f"pdq-{run_id}-modality",
                condition=indication,
                phase=phase,
                target_or_moa=modality,
                endpoint_family=endpoint_family,
                limit=25,
                expected_analog_dimension="same indication, phase, and modality",
                rationale="Secondary analog search adds modality when upstream asset identity provides one.",
            )
        )
    if asset:
        queries.append(
            CTGovSearchQuery(
                query_id=f"pdq-{run_id}-asset",
                condition=indication,
                intervention=asset,
                phase=phase,
                endpoint_family=endpoint_family,
                limit=25,
                expected_analog_dimension="same asset or close asset-family context",
                rationale="Asset-name search captures same-product studies when public registry records exist.",
            )
        )
    return AnalogSearchPlanOutput(
        output_id=f"analog-search-plan-{run_id}",
        target_nct_id=target_trial.nct_id,
        queries=tuple(queries),
        rationale="Search plan is bounded to CT.gov and prioritizes analog dimensions detectable from Agent 3, Agent 4, and the target trial registry record.",
        expected_dimensions=tuple(
            item
            for item in (
                "indication",
                "phase",
                "modality" if modality else None,
                "endpoint_family" if endpoint_family else None,
                "comparator" if comparator else None,
                "biomarker_or_line" if biomarker_or_line else None,
            )
            if item
        ),
        source_ids=tuple(
            dict.fromkeys(
                (
                    target_trial.source_id,
                    *agent3_output.trial_identity.source_ids,
                    *agent3_output.asset_identity.source_ids,
                    *agent4_output.asset_identity.source_ids,
                )
            )
        ),
        confidence=0.7 if indication != "unknown condition" else 0.4,
    )


def select_analog_trials(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    candidates: tuple[AnalogCandidateRecord, ...],
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
    search_plan: AnalogSearchPlanOutput,
    top_k: int = 10,
) -> AnalogTrialSelectionOutput:
    """Select analogs from normalized candidates without API calls."""

    del agent3_output, agent4_output, search_plan
    scored = [_score_candidate(target_trial, candidate) for candidate in candidates if candidate.trial.nct_id != target_trial.nct_id]
    scored.sort(key=lambda item: (-item[0].match_score, item[0].nct_id))
    selected = tuple(item[0] for item in scored[:top_k])
    selected_ids = {item.nct_id for item in selected}
    excluded = [
        ExcludedAnalogTrial(
            nct_id=target_trial.nct_id,
            reason="Target trial was excluded from analog benchmarking.",
            source_ids=(target_trial.source_id,),
        )
    ]
    for selection, candidate in scored[top_k:]:
        excluded.append(
            ExcludedAnalogTrial(
                nct_id=selection.nct_id,
                reason="Candidate ranked below selected analog cutoff.",
                source_ids=candidate.source_ids,
            )
        )
    for candidate in candidates:
        if candidate.trial.nct_id != target_trial.nct_id or candidate.trial.nct_id in selected_ids:
            continue
        if not any(item.nct_id == candidate.trial.nct_id for item in excluded):
            excluded.append(
                ExcludedAnalogTrial(
                    nct_id=candidate.trial.nct_id,
                    reason="Target trial was excluded from analog benchmarking.",
                    source_ids=candidate.source_ids,
                )
            )
    return AnalogTrialSelectionOutput(
        output_id=f"analog-selection-{run_id}",
        target_nct_id=target_trial.nct_id,
        selected_analogs=selected,
        excluded_candidates=tuple(excluded),
        source_ids=tuple(dict.fromkeys(source_id for candidate in candidates for source_id in candidate.source_ids)),
        confidence=0.75 if selected else 0.25,
    )


def build_protocol_strategy_sections(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    source_ids: tuple[str, ...],
    benchmark_summary: str,
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
) -> dict[str, ProtocolSectionDraft]:
    """Draft source-grounded strategy sections."""

    asset = agent4_output.asset_identity.asset_name or agent3_output.asset_identity.asset_name or "the investigational asset"
    indication = agent4_output.asset_identity.normalized_indication or ", ".join(target_trial.conditions) or "the target indication"
    phase = ", ".join(target_trial.phases) or "the target phase"
    endpoint = _endpoint_family(target_trial) or "endpoint family not clearly classified"
    risk = agent4_output.clinical_risk_summary.endpoint_risk_level or "unknown"
    population = _population_summary(target_trial)
    sections = {
        "executive_synopsis": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-executive-synopsis",
            title="Executive Synopsis",
            body=f"Draft strategy brief for {asset} in {indication}. The target registry record is {target_trial.nct_id}, {phase}, and this artifact requires human review before protocol use.",
            source_ids=source_ids,
            confidence=0.7,
        ),
        "strategic_rationale": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-strategic-rationale",
            title="Strategic Rationale",
            body=f"Rationale is grounded in Agent 3 clinical-risk context, Agent 4 diligence findings, and public analog trial benchmarks. Endpoint risk is {risk}; missing or low-confidence upstream items are carried as flags.",
            source_ids=source_ids,
            confidence=0.65,
        ),
        "analog_trial_benchmark_summary": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-analog-benchmark",
            title="Analog Trial Benchmark Summary",
            body=benchmark_summary,
            source_ids=source_ids,
            confidence=0.75,
        ),
        "target_population": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-target-population",
            title="Target Population",
            body=f"Draft target population follows the public target trial record: {population}. Human review should confirm biomarker, line-of-therapy, organ function, and safety exclusions.",
            source_ids=(target_trial.source_id,),
            confidence=0.65,
        ),
        "study_design": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-study-design",
            title="Study Design",
            body="Draft design should be benchmarked against selected analog trials for randomization, blinding, arm count, duration, and enrollment burden; no final protocol design decision is made by Agent 5.",
            source_ids=source_ids,
            confidence=0.6,
        ),
        "comparator_and_landscape_rationale": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-comparator-landscape",
            title="Comparator And Landscape Rationale",
            body="Comparator rationale is based on named comparators and control categories detected in selected CT.gov analog trials plus Agent 4 competitive-landscape context.",
            source_ids=source_ids,
            confidence=0.6,
        ),
        "endpoint_strategy": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-endpoint-strategy",
            title="Endpoint Strategy",
            body=f"Draft endpoint strategy should align with analog endpoint-family frequencies and the target trial primary endpoint family: {endpoint}. Statistical review is required before endpoint hierarchy or powering assumptions are used.",
            source_ids=source_ids,
            confidence=0.65,
        ),
        "safety_monitoring_outline": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-safety-monitoring",
            title="Safety Monitoring Outline",
            body="Safety monitoring should incorporate openFDA label context when available and safety exclusion themes from analog trial eligibility criteria. This is a review prompt, not a final safety plan.",
            source_ids=source_ids,
            confidence=0.55,
        ),
        "statistical_analysis_skeleton": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-stats-skeleton",
            title="Statistical Analysis Skeleton",
            body="Statistical skeleton should define estimand, analysis population, primary analysis method, multiplicity, interim review, and missing-data handling after biostatistician review; Agent 5 does not invent powering assumptions.",
            source_ids=source_ids,
            confidence=0.5,
        ),
        "operational_feasibility_risks": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-feasibility-risks",
            title="Operational Feasibility Risks",
            body="Operational risks should be reviewed against analog enrollment, duration, site/country distribution, biomarker testing, prior-treatment restrictions, and schedule burden.",
            source_ids=source_ids,
            confidence=0.65,
        ),
        "regulatory_standards_considerations": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-regulatory-standards",
            title="Regulatory Standards Considerations",
            body="Regulatory considerations are limited to questions for human review, including endpoint acceptability, comparator justification, eligibility defensibility, safety monitoring, and statistical analysis alignment.",
            source_ids=source_ids,
            confidence=0.5,
        ),
    }
    return sections


def build_eligibility_and_schedule_sections(
    *,
    run_id: str,
    source_ids: tuple[str, ...],
    inclusion_themes: tuple[str, ...],
    exclusion_themes: tuple[str, ...],
    safety_themes: tuple[str, ...],
) -> dict[str, ProtocolSectionDraft]:
    """Draft eligibility and schedule frameworks from analog themes."""

    inclusion = "; ".join(inclusion_themes) if inclusion_themes else "No common inclusion themes were confidently extracted."
    exclusion = "; ".join(exclusion_themes) if exclusion_themes else "No common exclusion themes were confidently extracted."
    safety = "; ".join(safety_themes) if safety_themes else "No recurring safety exclusion theme was confidently extracted."
    return {
        "draft_eligibility_framework": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-eligibility-framework",
            title="Draft Eligibility Framework",
            body=f"Draft eligibility framework should start from analog themes. Inclusion themes: {inclusion}. Exclusion themes: {exclusion}. Safety themes: {safety}.",
            source_ids=source_ids,
            confidence=0.6 if inclusion_themes or exclusion_themes else 0.35,
        ),
        "draft_schedule_of_assessments_framework": ProtocolSectionDraft(
            section_id=f"pd-{run_id}-schedule-framework",
            title="Draft Schedule Of Assessments Framework",
            body="Draft schedule should cover screening, baseline, treatment visits, response/safety assessments, biomarker or diagnostic testing when applicable, end-of-treatment, and follow-up. Visit timing remains a human-reviewed design choice.",
            source_ids=source_ids,
            confidence=0.45,
        ),
    }


def review_protocol_design(
    *,
    run_id: str,
    source_ids: tuple[str, ...],
    analog_limitations: tuple[str, ...],
    agent3_output: ClinicalOutcomePredictionOutput,
    agent4_output: DueDiligenceOutput,
) -> ProtocolReviewerCritique:
    """Review only; do not approve or add unsupported facts."""

    missing = []
    if agent3_output.missing_data_flags:
        missing.append("Agent 3 missing-data flags require clinical review.")
    if agent4_output.missing_data_flags:
        missing.append("Agent 4 missing-data flags require diligence review.")
    if analog_limitations:
        missing.append("Analog benchmark limitations require protocol team review.")
    return ProtocolReviewerCritique(
        critique_id=f"protocol-reviewer-critique-{run_id}",
        missing_elements=tuple(missing),
        statistical_questions=(
            "What estimand and analysis population should govern the primary endpoint?",
            "What sample-size and multiplicity assumptions are justified by source evidence?",
        ),
        regulatory_questions=(
            "Is the comparator or control strategy acceptable for the target population?",
            "Do eligibility restrictions align with safety evidence and intended-use rationale?",
        ),
        limitations=analog_limitations,
        source_ids=source_ids,
        confidence=0.6,
    )


def _score_candidate(target: ClinicalTrialRecord, candidate: AnalogCandidateRecord) -> tuple[SelectedAnalogTrial, AnalogCandidateRecord]:
    trial = candidate.trial
    matched: list[str] = []
    mismatched: list[str] = []
    unknown: list[str] = []
    score = 0.0
    if set(map(norm, target.conditions)) & set(map(norm, trial.conditions)):
        matched.append("indication")
        score += 0.3
    elif trial.conditions:
        mismatched.append("indication")
    else:
        unknown.append("indication")
    if set(_norm_phase_values(target.phases)) & set(_norm_phase_values(trial.phases)):
        matched.append("phase")
        score += 0.2
    elif trial.phases:
        mismatched.append("phase")
    else:
        unknown.append("phase")
    if _endpoint_family(target) and _endpoint_family(target) == _endpoint_family(trial):
        matched.append("endpoint_family")
        score += 0.2
    elif _endpoint_family(trial):
        mismatched.append("endpoint_family")
    else:
        unknown.append("endpoint_family")
    if _comparator_hint(target) and _comparator_hint(target) == _comparator_hint(trial):
        matched.append("comparator")
        score += 0.15
    elif _comparator_hint(trial):
        mismatched.append("comparator")
    else:
        unknown.append("comparator")
    if _biomarker_or_line(target) and _biomarker_or_line(target) == _biomarker_or_line(trial):
        matched.append("biomarker_or_line")
        score += 0.15
    elif _biomarker_or_line(trial):
        mismatched.append("biomarker_or_line")
    else:
        unknown.append("biomarker_or_line")
    confidence = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    return (
        SelectedAnalogTrial(
            nct_id=trial.nct_id,
            match_score=round(score, 3),
            match_confidence=confidence,
            matched_dimensions=tuple(matched),
            mismatched_dimensions=tuple(mismatched),
            unknown_dimensions=tuple(unknown),
            reasoning=f"Matched {len(matched)} dimensions; mismatched {len(mismatched)}; unknown {len(unknown)}.",
            source_ids=candidate.source_ids,
        ),
        candidate,
    )


def _endpoint_family(trial: ClinicalTrialRecord) -> str | None:
    text = " ".join(endpoint.measure for endpoint in (*trial.primary_endpoints, *trial.secondary_endpoints)).casefold()
    if not text:
        return None
    if any(term in text for term in ("overall survival", "mortality", "death")):
        return "survival"
    if any(term in text for term in ("progression-free", "time to", "duration")):
        return "time_to_event"
    if any(term in text for term in ("response", "orr", "remission")):
        return "response"
    if any(term in text for term in ("safety", "adverse", "toxicity")):
        return "safety"
    if any(term in text for term in ("biomarker", "pharmacodynamic", "marker")):
        return "biomarker"
    return "other"


def _comparator_hint(trial: ClinicalTrialRecord) -> str | None:
    text = " ".join(intervention.name for intervention in trial.interventions).casefold()
    if not text:
        return None
    if "placebo" in text:
        return "placebo"
    if "standard of care" in text or "best supportive" in text:
        return "standard_of_care"
    if "control" in text:
        return "control"
    return "active_or_single_arm"


def _biomarker_or_line(trial: ClinicalTrialRecord) -> str | None:
    text = (trial.eligibility_criteria or "").casefold()
    if "biomarker" in text or "mutation" in text or "expression" in text:
        return "biomarker_defined"
    if "prior therapy" in text or "previous treatment" in text or "line of therapy" in text:
        return "prior_treatment_defined"
    return None


def _population_summary(trial: ClinicalTrialRecord) -> str:
    parts = [
        f"condition {', '.join(trial.conditions) or 'unknown'}",
        f"sex {trial.sex or 'unknown'}",
        f"minimum age {trial.minimum_age or 'unknown'}",
        f"maximum age {trial.maximum_age or 'unknown'}",
    ]
    return "; ".join(parts)


def _norm_phase_values(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        text = value.casefold().replace(" ", "").replace("_", "")
        text = text.replace("phaseiii", "phase3").replace("phaseii", "phase2").replace("phasei", "phase1")
        normalized.append(text)
    return tuple(normalized)
