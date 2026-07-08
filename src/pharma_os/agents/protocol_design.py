"""Bounded helper subagents for the Protocol Design Brief workflow."""

from __future__ import annotations

from pharma_os.schemas import (
    AnalogCandidateRecord,
    AnalogSearchPlanOutput,
    AnalogTrialSelectionOutput,
    CTGovSearchQuery,
    ClinicalOutcomePredictionOutput,
    ClinicalTrialRecord,
    DueDiligenceOutput,
    ExcludedAnalogTrial,
    ProtocolReviewerCritique,
    ProtocolSectionDraft,
    SelectedAnalogTrial,
)
from pharma_os.tools._due_diligence_common import norm


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
