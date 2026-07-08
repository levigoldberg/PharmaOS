"""Deterministic tools for Agent 5 Protocol Design Briefs."""

from __future__ import annotations

from datetime import date
from statistics import mean, median

from pharma_os.schemas import (
    AnalogBenchmarkBundle,
    AnalogCandidateRecord,
    AnalogSearchPlanOutput,
    AnalogTrialSelectionOutput,
    AssumptionRecord,
    BenchmarkFrequency,
    BenchmarkNumericSummary,
    ClinicalOutcomePredictionOutput,
    ClinicalTrialIntelligenceInput,
    ClinicalTrialRecord,
    DueDiligenceOutput,
    EvidenceClaim,
    MissingDataFlag,
    ProtocolDesignBrief,
    ProtocolReviewerCritique,
    ProtocolSectionDraft,
    SourceMetadata,
)
from pharma_os.tools._due_diligence_common import missing, slug
from pharma_os.tools.clinicaltrials import ClinicalTrialsGovClient, ClinicalTrialsGovError


def execute_ctgov_search_plan(
    *,
    search_plan: AnalogSearchPlanOutput,
    target_nct_id: str,
    client: ClinicalTrialsGovClient | None = None,
) -> tuple[tuple[AnalogCandidateRecord, ...], tuple[SourceMetadata, ...], tuple[MissingDataFlag, ...]]:
    """Execute structured CT.gov analog searches and deduplicate by NCT ID."""

    ctgov = client or ClinicalTrialsGovClient()
    records_by_nct: dict[str, ClinicalTrialRecord] = {}
    query_ids_by_nct: dict[str, list[str]] = {}
    sources_by_id: dict[str, SourceMetadata] = {}
    flags: list[MissingDataFlag] = []
    for query in search_plan.queries:
        search_source = SourceMetadata(
            source_id=f"ctgov_search:protocol_design:{slug(query.query_id)}",
            title=f"Protocol design CT.gov analog search {query.query_id}",
            provenance=f"ClinicalTrials.gov API v2 search; expected analog dimension: {query.expected_analog_dimension}",
            source_type="clinical_trial_registry_search",
            version="v2",
        )
        sources_by_id[search_source.source_id] = search_source
        try:
            result = ctgov.search_trials(_clinical_trials_input(query))
        except (ClinicalTrialsGovError, ValueError) as exc:
            flags.append(
                missing(
                    f"protocol-design-ctgov-search-{slug(query.query_id)}",
                    "analog_benchmark",
                    "ctgov_search",
                    f"ClinicalTrials.gov analog query {query.query_id} failed: {exc.__class__.__name__}.",
                    "medium",
                )
            )
            continue
        for source in result.sources:
            sources_by_id[source.source_id] = source
        for trial in result.trials:
            records_by_nct[trial.nct_id] = trial
            query_ids_by_nct.setdefault(trial.nct_id, []).append(query.query_id)

    candidates = tuple(
        AnalogCandidateRecord(
            candidate_id=f"analog-candidate-{nct_id}",
            trial=trial,
            query_ids=tuple(dict.fromkeys(query_ids_by_nct.get(nct_id, ()))),
            source_ids=tuple(
                dict.fromkeys(
                    (
                        trial.source_id,
                        *(
                            f"ctgov_search:protocol_design:{slug(query_id)}"
                            for query_id in query_ids_by_nct.get(nct_id, ())
                        ),
                    )
                )
            ),
            provenance="protocol_design.ctgov_search_plan_execution",
        )
        for nct_id, trial in sorted(records_by_nct.items())
        if nct_id != target_nct_id
    )
    if not candidates:
        flags.append(
            missing(
                "protocol-design-no-analog-candidates",
                "analog_benchmark",
                "analog_candidates",
                "CT.gov searches returned no non-target analog candidates.",
                "high",
            )
        )
    return candidates, tuple(sources_by_id.values()), tuple(flags)


def calculate_analog_benchmark(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    candidates: tuple[AnalogCandidateRecord, ...],
    selection: AnalogTrialSelectionOutput,
    search_plan: AnalogSearchPlanOutput,
) -> AnalogBenchmarkBundle:
    """Calculate deterministic analog trial benchmark metrics."""

    candidate_by_id = {candidate.trial.nct_id: candidate for candidate in candidates}
    selected = tuple(candidate_by_id[item.nct_id] for item in selection.selected_analogs if item.nct_id in candidate_by_id)
    source_ids = tuple(
        dict.fromkeys(
            (
                *search_plan.source_ids,
                *selection.source_ids,
                *(source_id for candidate in selected for source_id in candidate.source_ids),
            )
        )
    )
    flags = list(_benchmark_missing_flags(selected))
    enrollment_values = [candidate.trial.enrollment_count for candidate in selected if candidate.trial.enrollment_count is not None]
    duration_values = [
        duration
        for candidate in selected
        if (duration := _planned_duration_months(candidate.trial.start_date, candidate.trial.primary_completion_date or candidate.trial.completion_date)) is not None
    ]
    site_values = [len(candidate.trial.locations) for candidate in selected if candidate.trial.locations]
    return AnalogBenchmarkBundle(
        bundle_id=f"analog-benchmark-{run_id}",
        target_nct_id=target_trial.nct_id,
        selected_analog_ids=tuple(item.nct_id for item in selection.selected_analogs),
        excluded_analog_ids=tuple(item.nct_id for item in selection.excluded_candidates),
        search_plan=search_plan,
        selection=selection,
        enrollment=_numeric_summary(
            values=enrollment_values,
            selected_count=len(selected),
            missing_count=len(selected) - len(enrollment_values),
            unit="participants",
            source_ids=source_ids,
        ),
        planned_duration_months=_numeric_summary(
            values=duration_values,
            selected_count=len(selected),
            missing_count=len(selected) - len(duration_values),
            unit="months",
            source_ids=source_ids,
        ),
        randomized_frequency=_frequency(_design_labels(selected, "randomized"), source_ids),
        blinding_frequency=_frequency(_design_labels(selected, "blinding"), source_ids),
        arm_count_distribution=_frequency(tuple(str(_arm_count(candidate.trial)) for candidate in selected), source_ids),
        primary_endpoint_family_frequency=_frequency(
            tuple(_endpoint_family(endpoint.measure) for candidate in selected for endpoint in candidate.trial.primary_endpoints),
            source_ids,
        ),
        secondary_endpoint_family_frequency=_frequency(
            tuple(_endpoint_family(endpoint.measure) for candidate in selected for endpoint in candidate.trial.secondary_endpoints),
            source_ids,
        ),
        comparator_categories=_frequency(tuple(_comparator_category(candidate.trial) for candidate in selected), source_ids),
        named_comparators=_named_comparators(selected),
        inclusion_themes=_theme_counts(selected, section="inclusion"),
        exclusion_themes=_theme_counts(selected, section="exclusion"),
        biomarker_testing_themes=_theme_keyword(selected, ("biomarker", "mutation", "expression", "testing")),
        prior_treatment_themes=_theme_keyword(selected, ("prior therapy", "previous treatment", "line of therapy", "refractory")),
        safety_exclusion_themes=_theme_keyword(selected, ("cardiac", "infection", "organ function", "pregnant", "toxicity", "qt")),
        country_distribution=_frequency(tuple(location.country for candidate in selected for location in candidate.trial.locations if location.country), source_ids),
        site_count=_numeric_summary(
            values=site_values,
            selected_count=len(selected),
            missing_count=len(selected) - len(site_values),
            unit="sites",
            source_ids=source_ids,
        ),
        status_distribution=_frequency(tuple(candidate.trial.overall_status or "unknown" for candidate in selected), source_ids),
        results_availability=_frequency(tuple("results_available" if candidate.trial.results_available else "results_not_available" for candidate in selected), source_ids),
        limitations=_limitations(selected, flags),
        source_ids=source_ids,
        missing_data_flags=tuple(flags),
        confidence=_benchmark_confidence(len(selected), flags),
    )


def build_benchmark_summary(bundle: AnalogBenchmarkBundle) -> str:
    """Return concise source-grounded benchmark prose."""

    selected = len(bundle.selected_analog_ids)
    enrollment = _stat_phrase(bundle.enrollment)
    duration = _stat_phrase(bundle.planned_duration_months)
    endpoints = ", ".join(f"{item.label}: {item.count}" for item in bundle.primary_endpoint_family_frequency[:4]) or "no classified primary endpoint families"
    controls = ", ".join(f"{item.label}: {item.count}" for item in bundle.comparator_categories[:4]) or "no comparator categories detected"
    return (
        f"Analog benchmark uses {selected} selected CT.gov analog trials. "
        f"Enrollment {enrollment}; planned duration {duration}. "
        f"Primary endpoint families: {endpoints}. Comparator categories: {controls}. "
        "Limitations and missing fields are carried as flags."
    )


def build_protocol_design_brief(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    strategy_sections: dict[str, ProtocolSectionDraft],
    eligibility_sections: dict[str, ProtocolSectionDraft],
    reviewer_critique: ProtocolReviewerCritique,
    benchmark_bundle: AnalogBenchmarkBundle,
    claims: tuple[EvidenceClaim, ...],
    assumptions: tuple[AssumptionRecord, ...],
    missing_data_flags: tuple[MissingDataFlag, ...],
    source_ids: tuple[str, ...],
) -> ProtocolDesignBrief:
    """Assemble the typed ProtocolDesignBrief."""

    questions = tuple(
        dict.fromkeys(
            (
                *reviewer_critique.statistical_questions,
                *reviewer_critique.regulatory_questions,
                "Which analog trials should be accepted or rejected by the clinical team?",
                "Which missing CT.gov, PubMed, or label fields must be resolved before protocol drafting?",
            )
        )
    )
    return ProtocolDesignBrief(
        brief_id=f"protocol-design-brief-{run_id}",
        title=f"Draft Protocol Design Brief for {target_trial.nct_id}",
        executive_synopsis=strategy_sections["executive_synopsis"],
        strategic_rationale=strategy_sections["strategic_rationale"],
        analog_trial_benchmark_summary=strategy_sections["analog_trial_benchmark_summary"],
        target_population=strategy_sections["target_population"],
        study_design=strategy_sections["study_design"],
        comparator_and_landscape_rationale=strategy_sections["comparator_and_landscape_rationale"],
        endpoint_strategy=strategy_sections["endpoint_strategy"],
        draft_eligibility_framework=eligibility_sections["draft_eligibility_framework"],
        draft_schedule_of_assessments_framework=eligibility_sections["draft_schedule_of_assessments_framework"],
        safety_monitoring_outline=strategy_sections["safety_monitoring_outline"],
        statistical_analysis_skeleton=strategy_sections["statistical_analysis_skeleton"],
        operational_feasibility_risks=strategy_sections["operational_feasibility_risks"],
        regulatory_standards_considerations=strategy_sections["regulatory_standards_considerations"],
        human_review_questions=questions,
        source_backed_claim_ids=tuple(claim.claim_id for claim in claims),
        assumptions=assumptions,
        missing_data_flags=missing_data_flags,
        reviewer_critique=reviewer_critique,
        source_ids=source_ids,
        confidence=min(0.75, benchmark_bundle.confidence),
    )


def build_protocol_design_claims(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    benchmark_bundle: AnalogBenchmarkBundle,
    source_ids: tuple[str, ...],
) -> tuple[EvidenceClaim, ...]:
    """Build source-backed factual claims for Agent 5 output."""

    claims = [
        EvidenceClaim(
            claim_id=f"claim-{run_id}-protocol-target",
            claim_text=f"Target trial {target_trial.nct_id} has CT.gov status {target_trial.overall_status or 'unknown'}.",
            source_ids=(target_trial.source_id,),
            provenance="protocol_design.ctgov_target_trial",
            confidence=0.95,
            confidence_level="very_high",
        ),
        EvidenceClaim(
            claim_id=f"claim-{run_id}-analog-count",
            claim_text=f"Protocol design benchmarking selected {len(benchmark_bundle.selected_analog_ids)} CT.gov analog trials.",
            source_ids=source_ids,
            provenance="protocol_design.analog_benchmark",
            confidence=benchmark_bundle.confidence,
            confidence_level="medium" if benchmark_bundle.confidence >= 0.5 else "low",
        ),
    ]
    if benchmark_bundle.enrollment.median is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-analog-enrollment",
                claim_text=f"Selected analog median enrollment is {benchmark_bundle.enrollment.median:.1f} participants.",
                source_ids=benchmark_bundle.enrollment.source_ids,
                provenance="protocol_design.analog_benchmark.calculated",
                confidence=benchmark_bundle.confidence,
                confidence_level="medium",
            )
        )
    if benchmark_bundle.planned_duration_months.median is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-analog-duration",
                claim_text=f"Selected analog median planned duration is {benchmark_bundle.planned_duration_months.median:.1f} months.",
                source_ids=benchmark_bundle.planned_duration_months.source_ids,
                provenance="protocol_design.analog_benchmark.calculated",
                confidence=benchmark_bundle.confidence,
                confidence_level="medium",
            )
        )
    return tuple(claims)


def _clinical_trials_input(query: object) -> ClinicalTrialIntelligenceInput:
    terms = " ".join(
        str(value)
        for value in (
            getattr(query, "target_or_moa", None),
            getattr(query, "endpoint_family", None),
            getattr(query, "comparator", None),
            getattr(query, "biomarker_or_line", None),
            getattr(query, "term", None),
        )
        if value
    )
    return ClinicalTrialIntelligenceInput(
        disease=getattr(query, "condition"),
        drug=getattr(query, "intervention", None),
        target=terms or None,
        phase=getattr(query, "phase", None),
        limit=getattr(query, "limit", 25),
    )


def _numeric_summary(
    *,
    values: list[int | float],
    selected_count: int,
    missing_count: int,
    unit: str,
    source_ids: tuple[str, ...],
) -> BenchmarkNumericSummary:
    if not values:
        return BenchmarkNumericSummary(missing_count=selected_count, unit=unit, source_ids=source_ids)
    sorted_values = sorted(float(value) for value in values)
    q1 = _percentile(sorted_values, 25)
    q3 = _percentile(sorted_values, 75)
    return BenchmarkNumericSummary(
        observed_count=len(sorted_values),
        missing_count=missing_count,
        mean=round(mean(sorted_values), 2),
        median=round(median(sorted_values), 2),
        minimum=round(min(sorted_values), 2),
        maximum=round(max(sorted_values), 2),
        iqr=round(q3 - q1, 2) if q1 is not None and q3 is not None else None,
        unit=unit,
        source_ids=source_ids,
    )


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * (percentile / 100)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _frequency(labels: tuple[str | None, ...], source_ids: tuple[str, ...]) -> tuple[BenchmarkFrequency, ...]:
    filtered = tuple(label or "unknown" for label in labels)
    if not filtered:
        return ()
    total = len(filtered)
    counts: dict[str, int] = {}
    for label in filtered:
        counts[label] = counts.get(label, 0) + 1
    return tuple(
        BenchmarkFrequency(label=label, count=count, frequency=round(count / total, 4), source_ids=source_ids)
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _planned_duration_months(start: str | None, end: str | None) -> float | None:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    if start_date is None or end_date is None or end_date < start_date:
        return None
    return round((end_date - start_date).days / 30.4375, 1)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parts = [int(part) for part in text.split("-")]
        except ValueError:
            return None
        if fmt == "%Y-%m-%d" and len(parts) == 3:
            return date(parts[0], parts[1], parts[2])
        if fmt == "%Y-%m" and len(parts) == 2:
            return date(parts[0], parts[1], 1)
        if fmt == "%Y" and len(parts) == 1:
            return date(parts[0], 1, 1)
    return None


def _arm_count(trial: ClinicalTrialRecord) -> int:
    return max(1, len(trial.interventions)) if trial.interventions else 0


def _design_labels(selected: tuple[AnalogCandidateRecord, ...], field: str) -> tuple[str, ...]:
    labels = []
    for candidate in selected:
        text = _trial_text(candidate.trial)
        if field == "randomized":
            labels.append("randomized" if "randomized" in text else "single_arm_or_not_detected")
        else:
            if "open label" in text or "open-label" in text:
                labels.append("open_label")
            elif "blind" in text or "masked" in text:
                labels.append("blinded_or_masked")
            else:
                labels.append("not_detected")
    return tuple(labels)


def _endpoint_family(measure: str) -> str:
    text = measure.casefold()
    if any(term in text for term in ("overall survival", "mortality", "death")):
        return "survival"
    if any(term in text for term in ("progression-free", "time to", "duration")):
        return "time_to_event"
    if any(term in text for term in ("response", "orr", "remission")):
        return "response"
    if any(term in text for term in ("safety", "adverse", "toxicity")):
        return "safety"
    if any(term in text for term in ("biomarker", "marker", "pharmacodynamic")):
        return "biomarker"
    return "other"


def _comparator_category(trial: ClinicalTrialRecord) -> str:
    text = " ".join(intervention.name for intervention in trial.interventions).casefold()
    if "placebo" in text:
        return "placebo_control"
    if "standard of care" in text or "best supportive" in text:
        return "standard_of_care"
    if "control" in text:
        return "control"
    if len(trial.interventions) > 1:
        return "active_comparator_or_combination"
    return "single_arm_or_uncontrolled"


def _named_comparators(selected: tuple[AnalogCandidateRecord, ...]) -> tuple[str, ...]:
    names = []
    for candidate in selected:
        for intervention in candidate.trial.interventions:
            name = intervention.name.strip()
            if name and name.casefold() not in {"unknown intervention"}:
                names.append(name)
    return tuple(dict.fromkeys(names))[:25]


def _theme_counts(selected: tuple[AnalogCandidateRecord, ...], *, section: str) -> tuple[str, ...]:
    patterns = {
        "inclusion": ("diagnosis", "age", "performance status", "measurable disease", "consent", "biomarker"),
        "exclusion": ("pregnant", "infection", "cardiac", "prior therapy", "organ function", "metastases"),
    }
    return _theme_keyword(selected, patterns[section])


def _theme_keyword(selected: tuple[AnalogCandidateRecord, ...], keywords: tuple[str, ...]) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for candidate in selected:
        criteria = (candidate.trial.eligibility_criteria or "").casefold()
        for keyword in keywords:
            if keyword in criteria:
                counts[keyword] = counts.get(keyword, 0) + 1
    return tuple(label for label, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])))[:10]


def _trial_text(trial: ClinicalTrialRecord) -> str:
    return " ".join(
        item
        for item in (
            trial.brief_title,
            trial.official_title,
            trial.study_type,
            trial.eligibility_criteria,
            *(endpoint.measure for endpoint in trial.primary_endpoints),
            *(endpoint.measure for endpoint in trial.secondary_endpoints),
        )
        if item
    ).casefold()


def _benchmark_missing_flags(selected: tuple[AnalogCandidateRecord, ...]) -> tuple[MissingDataFlag, ...]:
    if not selected:
        return (
            missing(
                "protocol-design-benchmark-selected-empty",
                "analog_benchmark",
                "selected_analog_ids",
                "No selected analog trials were available for benchmark calculations.",
                "high",
            ),
        )
    flags = []
    if any(candidate.trial.enrollment_count is None for candidate in selected):
        flags.append(missing("protocol-design-analog-enrollment-missing", "analog_benchmark", "enrollment", "One or more selected analog trials lack enrollment count.", "medium"))
    if any(_planned_duration_months(candidate.trial.start_date, candidate.trial.primary_completion_date or candidate.trial.completion_date) is None for candidate in selected):
        flags.append(missing("protocol-design-analog-duration-missing", "analog_benchmark", "planned_duration_months", "One or more selected analog trials lack calculable planned duration.", "medium"))
    if any(not candidate.trial.primary_endpoints for candidate in selected):
        flags.append(missing("protocol-design-analog-primary-endpoints-missing", "analog_benchmark", "primary_endpoint_family_frequency", "One or more selected analog trials lack primary endpoints.", "medium"))
    if any(not candidate.trial.eligibility_criteria for candidate in selected):
        flags.append(missing("protocol-design-analog-eligibility-missing", "analog_benchmark", "eligibility_themes", "One or more selected analog trials lack eligibility criteria text.", "medium"))
    return tuple(flags)


def _limitations(selected: tuple[AnalogCandidateRecord, ...], flags: list[MissingDataFlag]) -> tuple[str, ...]:
    limitations = [flag.reason for flag in flags]
    if selected and len(selected) < 5:
        limitations.append("Fewer than five selected analog trials were available, limiting benchmark stability.")
    return tuple(dict.fromkeys(limitations))


def _benchmark_confidence(selected_count: int, flags: list[MissingDataFlag]) -> float:
    if selected_count == 0:
        return 0.15
    high = sum(1 for flag in flags if flag.severity == "high")
    medium = sum(1 for flag in flags if flag.severity == "medium")
    return max(0.2, min(0.85, 0.55 + min(selected_count, 10) * 0.03 - high * 0.15 - medium * 0.04))


def _stat_phrase(summary: BenchmarkNumericSummary) -> str:
    if summary.observed_count == 0:
        return f"not calculable with {summary.missing_count} missing values"
    return (
        f"median {summary.median} {summary.unit or ''}, mean {summary.mean}, "
        f"range {summary.minimum}-{summary.maximum}, IQR {summary.iqr}, missing {summary.missing_count}"
    )
