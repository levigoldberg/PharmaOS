"""Deterministic tools for Agent 5 Protocol Design Briefs."""

from __future__ import annotations

from datetime import date
import re
from statistics import mean, median

from pharma_os.schemas import (
    AnalogDerivedDesignDecision,
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
    ClinicalTrialsSearchResult,
    DueDiligenceOutput,
    EvidenceClaim,
    FollowOnCandidateRecord,
    FollowOnTrialAdjudication,
    FollowOnTrialAdjudicationOutput,
    MissingDataFlag,
    NextStudyIntent,
    ProtocolDesignBrief,
    QualitativeProtocolSynthesis,
    ProtocolReviewerCritique,
    ProtocolSectionDraft,
    SelectedAnalogTrial,
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
        attempt_notes: list[str] = []
        result = None
        used_attempt_label: str | None = None
        for attempt_label, input_data in _clinical_trials_input_variants(query):
            try:
                attempt_result = ctgov.search_trials(input_data)
            except (ClinicalTrialsGovError, ValueError) as exc:
                attempt_notes.append(f"{attempt_label}:{exc.__class__.__name__}")
                continue
            if not attempt_result.trials:
                attempt_notes.append(f"{attempt_label}:no_results")
                continue
            result = attempt_result
            used_attempt_label = attempt_label
            break
        source_provenance = search_source.provenance
        if used_attempt_label:
            source_provenance = f"{source_provenance}; executed attempt={used_attempt_label}"
        if attempt_notes:
            source_provenance = f"{source_provenance}; earlier attempts={'; '.join(attempt_notes[:4])}"
        sources_by_id[search_source.source_id] = search_source.model_copy(update={"provenance": source_provenance})
        if result is None:
            flags.append(
                missing(
                    f"protocol-design-ctgov-search-{slug(query.query_id)}",
                    "analog_benchmark",
                    "ctgov_search",
                    f"ClinicalTrials.gov analog query {query.query_id} returned no usable trials after API-safe retries: {'; '.join(attempt_notes[:6]) or 'no attempts'}.",
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


def hydrate_selected_analog_candidates(
    *,
    target_trial: ClinicalTrialRecord,
    selection: AnalogTrialSelectionOutput,
    candidates: tuple[AnalogCandidateRecord, ...],
    client: ClinicalTrialsGovClient | None = None,
) -> tuple[tuple[AnalogCandidateRecord, ...], tuple[SourceMetadata, ...], tuple[MissingDataFlag, ...]]:
    """Fetch full CT.gov records for selected analog NCTs missing from retrieved candidates."""

    ctgov = client or ClinicalTrialsGovClient()
    by_nct = {candidate.trial.nct_id: candidate for candidate in candidates}
    ordered: list[AnalogCandidateRecord] = list(candidates)
    sources: list[SourceMetadata] = []
    flags: list[MissingDataFlag] = []
    for selected in selection.selected_analogs:
        try:
            nct_id = ClinicalTrialsGovClient.normalize_nct_id(selected.nct_id)
        except ValueError:
            flags.append(
                missing(
                    f"protocol-design-selected-analog-invalid-{slug(selected.nct_id)}",
                    "analog_selection",
                    "selected_analog_nct_id",
                    f"AnalogSelectionAgent selected invalid NCT ID {selected.nct_id}; it was not carried forward.",
                    "high",
                )
            )
            continue
        if nct_id == target_trial.nct_id or nct_id in by_nct:
            continue
        try:
            trial = ctgov.fetch_trial(nct_id)
        except (ClinicalTrialsGovError, ValueError) as exc:
            flags.append(
                missing(
                    f"protocol-design-selected-analog-fetch-{slug(nct_id)}",
                    "analog_selection",
                    "selected_analog_full_ctgov_record",
                    f"AnalogSelectionAgent selected {nct_id}, but its full CT.gov record could not be fetched ({exc.__class__.__name__}); it was not carried forward.",
                    "high",
                )
            )
            continue
        source = _source_for_trial(trial)
        sources.append(source)
        candidate = AnalogCandidateRecord(
            candidate_id=f"analog-candidate-{nct_id}",
            trial=trial,
            query_ids=("selected_analog_hydration",),
            similarity_features={
                "recovered_from_selection": True,
                "selection_match_score": selected.match_score,
                "selection_match_confidence": selected.match_confidence,
            },
            source_ids=(trial.source_id,),
            provenance="protocol_design.selected_analog_hydration",
        )
        by_nct[nct_id] = candidate
        ordered.append(candidate)
    return tuple(ordered), tuple(sources), tuple(flags)


def annotate_analog_similarity_features(
    *,
    target_trial: ClinicalTrialRecord,
    candidates: tuple[AnalogCandidateRecord, ...],
) -> tuple[AnalogCandidateRecord, ...]:
    """Attach deterministic current-trial similarity features to analog candidates."""

    return tuple(
        candidate.model_copy(update={"similarity_features": _similarity_features(target_trial, candidate.trial)})
        for candidate in candidates
    )


def retrieve_follow_on_candidates(
    *,
    selected_analogs: tuple[AnalogCandidateRecord, ...],
    client: ClinicalTrialsGovClient | None = None,
) -> tuple[tuple[FollowOnCandidateRecord, ...], tuple[SourceMetadata, ...], tuple[MissingDataFlag, ...]]:
    """Search CT.gov for same-asset/same-indication/same-sponsor follow-on candidates."""

    ctgov = client or ClinicalTrialsGovClient()
    candidates_by_key: dict[tuple[str, str], FollowOnCandidateRecord] = {}
    sources_by_id: dict[str, SourceMetadata] = {}
    flags: list[MissingDataFlag] = []
    for analog in selected_analogs:
        aliases = _expanded_asset_aliases(analog.trial)
        sponsors = _sponsor_names(analog.trial)
        indication_terms = _follow_on_indication_terms(analog.trial)
        if not aliases or not sponsors or not indication_terms:
            flags.append(
                missing(
                    f"protocol-design-follow-on-lineage-inputs-{slug(analog.trial.nct_id)}",
                    "follow_on_lineage",
                    "same_asset_same_indication_same_sponsor",
                    f"Analog {analog.trial.nct_id} lacks asset, indication, or sponsor fields needed for lineage-constrained follow-on search.",
                    "medium",
                )
            )
            continue
        alias_failures: list[MissingDataFlag] = []
        analog_had_result = False
        for alias in aliases[:10]:
            query_source = SourceMetadata(
                source_id=f"ctgov_search:protocol_design_follow_on:{slug(analog.trial.nct_id)}:{slug(alias)}",
                title=f"Follow-on search for {analog.trial.nct_id} using {alias}",
                provenance="ClinicalTrials.gov API v2 search constrained post hoc to same asset, indication, and sponsor",
                source_type="clinical_trial_registry_search",
                version="v2",
            )
            attempt_notes: list[str] = []
            results: list[ClinicalTrialsSearchResult] = []
            for attempt_label, input_data in _follow_on_search_inputs(alias=alias, indication_terms=indication_terms):
                try:
                    attempt_result = ctgov.search_trials(input_data)
                except (ClinicalTrialsGovError, ValueError) as exc:
                    attempt_notes.append(f"{attempt_label}:{exc.__class__.__name__}")
                    continue
                if not attempt_result.trials:
                    attempt_notes.append(f"{attempt_label}:no_results")
                    continue
                attempt_notes.append(f"{attempt_label}:ok:{len(attempt_result.trials)}")
                results.append(attempt_result)
            provenance = query_source.provenance
            if attempt_notes:
                provenance = f"{provenance}; attempts={'; '.join(attempt_notes[:8])}"
            sources_by_id[query_source.source_id] = query_source.model_copy(update={"provenance": provenance})
            if not results:
                alias_failures.append(
                    missing(
                        f"protocol-design-follow-on-search-{slug(analog.trial.nct_id)}-{slug(alias)}",
                        "follow_on_lineage",
                        "ctgov_search",
                        f"Follow-on CT.gov query for {analog.trial.nct_id} using {alias} returned no usable trials after retries: {'; '.join(attempt_notes[:6]) or 'no attempts'}.",
                        "medium",
                    )
                )
                continue
            analog_had_result = True
            for result in results:
                for source in result.sources:
                    sources_by_id[source.source_id] = source
                for trial in result.trials:
                    key = (analog.trial.nct_id, trial.nct_id)
                    features = _lineage_features(anchor=analog.trial, candidate=trial, alias=alias)
                    exclusion = _follow_on_exclusion_reason(features)
                    record = FollowOnCandidateRecord(
                        candidate_id=f"follow-on-candidate-{analog.trial.nct_id}-{trial.nct_id}",
                        anchor_analog_nct_id=analog.trial.nct_id,
                        trial=trial,
                        lineage_features=features,
                        exclusion_reason=exclusion,
                        source_ids=(trial.source_id, query_source.source_id),
                        provenance=(
                            "protocol_design.follow_on_lineage_search.excluded"
                            if exclusion is not None
                            else "protocol_design.follow_on_lineage_search"
                        ),
                    )
                    existing = candidates_by_key.get(key)
                    if existing is None or (existing.exclusion_reason is not None and record.exclusion_reason is None):
                        candidates_by_key[key] = record
                    if exclusion is not None:
                        continue
        if not analog_had_result:
            flags.extend(alias_failures[:3])
    candidates = tuple(candidates_by_key.values())
    if not any(candidate.exclusion_reason is None for candidate in candidates):
        flags.append(
            missing(
                "protocol-design-no-follow-on-candidates",
                "follow_on_lineage",
                "selected_follow_on_nct_ids",
                "No same-asset, same-indication, same-sponsor follow-on candidates survived deterministic lineage filters.",
                "high",
            )
        )
    return candidates, tuple(sources_by_id.values()), tuple(flags)


def adjudicate_follow_on_trials_deterministically(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    selected_analogs: tuple[AnalogCandidateRecord, ...],
    follow_on_candidates: tuple[FollowOnCandidateRecord, ...],
) -> FollowOnTrialAdjudicationOutput:
    """Fallback adjudication that does not force a follow-on when lineage evidence is weak."""

    adjudications: list[FollowOnTrialAdjudication] = []
    source_ids: list[str] = []
    for analog in selected_analogs:
        candidates = tuple(
            candidate
            for candidate in follow_on_candidates
            if candidate.anchor_analog_nct_id == analog.trial.nct_id
        )
        viable = tuple(candidate for candidate in candidates if candidate.exclusion_reason is None)
        excluded = tuple(candidate.trial.nct_id for candidate in candidates if candidate.exclusion_reason)
        source_ids.extend(source_id for candidate in candidates for source_id in candidate.source_ids)
        if not viable:
            adjudications.append(
                FollowOnTrialAdjudication(
                    anchor_analog_nct_id=analog.trial.nct_id,
                    status="no_plausible_follow_on",
                    excluded_follow_on_nct_ids=excluded,
                    rationale="No same-asset, same-indication, same-sponsor candidate survived deterministic lineage filters.",
                    ambiguity_flags=("no_viable_follow_on_candidates",),
                    source_ids=tuple(dict.fromkeys(source_id for candidate in candidates for source_id in candidate.source_ids)),
                    confidence=0.35,
                )
            )
            continue
        ranked = sorted(viable, key=lambda item: (_phase_progression_rank(item), _date_sort_key(item.trial.start_date), item.trial.nct_id), reverse=True)
        if len(ranked) == 1:
            adjudications.append(
                FollowOnTrialAdjudication(
                    anchor_analog_nct_id=analog.trial.nct_id,
                    status="clear_follow_on",
                    selected_follow_on_nct_ids=(ranked[0].trial.nct_id,),
                    excluded_follow_on_nct_ids=excluded,
                    rationale="One candidate survived same-asset, same-indication, same-sponsor lineage filters.",
                    ambiguity_flags=(),
                    source_ids=ranked[0].source_ids,
                    confidence=0.75,
                )
            )
            continue
        selected = tuple(candidate.trial.nct_id for candidate in ranked[:3])
        adjudications.append(
            FollowOnTrialAdjudication(
                anchor_analog_nct_id=analog.trial.nct_id,
                status="multiple_plausible_branches",
                selected_follow_on_nct_ids=selected,
                alternative_follow_on_nct_ids=tuple(candidate.trial.nct_id for candidate in ranked[3:]),
                excluded_follow_on_nct_ids=excluded,
                rationale="Multiple same-asset, same-indication, same-sponsor candidates survived filters; these may represent expansion, optimization, pivotal progression, confirmatory development, label expansion, or parallel branches.",
                ambiguity_flags=("multiple_plausible_lineage_branches",),
                source_ids=tuple(dict.fromkeys(source_id for candidate in ranked for source_id in candidate.source_ids)),
                confidence=0.6,
            )
        )
    return FollowOnTrialAdjudicationOutput(
        output_id=f"follow-on-adjudication-{run_id}",
        target_nct_id=target_trial.nct_id,
        adjudications=tuple(adjudications),
        rationale_summary="Follow-on adjudication used same asset, same indication, same sponsor, chronology, phase progression, and study-role continuity without forcing a single successor.",
        source_ids=tuple(dict.fromkeys(source_ids)),
        confidence=min((item.confidence for item in adjudications), default=0.25),
    )


def fetch_selected_follow_on_trials(
    *,
    adjudication: FollowOnTrialAdjudicationOutput,
    follow_on_candidates: tuple[FollowOnCandidateRecord, ...],
    client: ClinicalTrialsGovClient | None = None,
) -> tuple[tuple[ClinicalTrialRecord, ...], tuple[SourceMetadata, ...], tuple[MissingDataFlag, ...]]:
    """Fetch complete CT.gov records for selected follow-on IDs."""

    ctgov = client or ClinicalTrialsGovClient()
    selected_ids = tuple(
        dict.fromkeys(
            nct_id
            for item in adjudication.adjudications
            for nct_id in item.selected_follow_on_nct_ids
        )
    )
    candidate_by_id = {candidate.trial.nct_id: candidate.trial for candidate in follow_on_candidates}
    records: list[ClinicalTrialRecord] = []
    sources: list[SourceMetadata] = []
    flags: list[MissingDataFlag] = []
    for nct_id in selected_ids:
        try:
            trial = ctgov.fetch_trial(nct_id)
        except (ClinicalTrialsGovError, ValueError):
            trial = candidate_by_id.get(nct_id)
            if trial is None:
                flags.append(
                    missing(
                        f"protocol-design-follow-on-fetch-{slug(nct_id)}",
                        "follow_on_trials",
                        "full_ctgov_record",
                        f"Could not fetch full CT.gov record for selected follow-on {nct_id}.",
                        "medium",
                    )
                )
                continue
        records.append(trial)
        sources.append(_source_for_trial(trial))
    return tuple(records), tuple(sources), tuple(flags)


def calculate_follow_on_benchmark(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    search_plan: AnalogSearchPlanOutput,
    follow_on_trials: tuple[ClinicalTrialRecord, ...],
    adjudication: FollowOnTrialAdjudicationOutput,
) -> AnalogBenchmarkBundle:
    """Reuse analog benchmark math over selected follow-on trials."""

    candidates = tuple(
        AnalogCandidateRecord(
            candidate_id=f"follow-on-benchmark-candidate-{trial.nct_id}",
            trial=trial,
            query_ids=("follow_on_lineage",),
            similarity_features={"benchmark_subject": "selected_follow_on_trial"},
            source_ids=(trial.source_id,),
            provenance="protocol_design.follow_on_benchmark",
        )
        for trial in follow_on_trials
    )
    selection = AnalogTrialSelectionOutput(
        output_id=f"follow-on-benchmark-selection-{run_id}",
        target_nct_id=target_trial.nct_id,
        selected_analogs=tuple(
            SelectedAnalogTrial(
                nct_id=trial.nct_id,
                match_score=1.0,
                match_confidence="high",
                matched_dimensions=("adjudicated_follow_on_precedent",),
                reasoning="Selected follow-on trial from analog lineage adjudication.",
                source_ids=(trial.source_id,),
            )
            for trial in follow_on_trials
        ),
        excluded_candidates=(),
        source_ids=tuple(dict.fromkeys((*adjudication.source_ids, *(trial.source_id for trial in follow_on_trials)))),
        confidence=adjudication.confidence,
    )
    bundle = calculate_analog_benchmark(
        run_id=f"follow-on-{run_id}",
        target_trial=target_trial,
        candidates=candidates,
        selection=selection,
        search_plan=search_plan,
    )
    limitations = tuple(dict.fromkeys(("Benchmark calculated from selected follow-on trials, not initial analog trials.", *bundle.limitations)))
    return bundle.model_copy(update={"limitations": limitations})


def build_qualitative_protocol_synthesis(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    follow_on_trials: tuple[ClinicalTrialRecord, ...],
    benchmark_bundle: AnalogBenchmarkBundle,
) -> QualitativeProtocolSynthesis:
    """Deterministic fallback synthesis of recurring follow-on protocol patterns."""

    source_ids = tuple(dict.fromkeys((*benchmark_bundle.source_ids, *(trial.source_id for trial in follow_on_trials))))
    comparator = tuple(f"{item.label}: {item.count}/{sum(row.count for row in benchmark_bundle.comparator_categories) or item.count}" for item in benchmark_bundle.comparator_categories[:4])
    randomized = tuple(f"{item.label}: {item.count}" for item in benchmark_bundle.randomized_frequency[:4])
    endpoint = tuple(f"{item.label}: {item.count}" for item in benchmark_bundle.primary_endpoint_family_frequency[:4])
    phases = tuple(f"{item.label}: {item.count}" for item in _frequency(tuple(", ".join(trial.phases) or "unknown" for trial in follow_on_trials), source_ids)[:4])
    insufficient = []
    if len(follow_on_trials) < 2:
        insufficient.append("Fewer than two selected follow-on trials; qualitative precedent is sparse.")
    if not endpoint:
        insufficient.append("No recurring primary endpoint pattern was available from follow-on trials.")
    return QualitativeProtocolSynthesis(
        output_id=f"qualitative-protocol-synthesis-{run_id}",
        target_nct_id=target_trial.nct_id,
        study_role_patterns=phases or ("No phase/study-role pattern was confidently identified.",),
        target_population_patterns=tuple(dict.fromkeys(_population_pattern(trial) for trial in follow_on_trials if _population_pattern(trial)))[:6],
        study_design_patterns=randomized or ("Randomization/design pattern insufficiently evidenced.",),
        comparator_control_patterns=comparator or ("Comparator/control pattern insufficiently evidenced.",),
        primary_endpoint_patterns=endpoint or ("Primary endpoint pattern insufficiently evidenced.",),
        secondary_endpoint_patterns=tuple(f"{item.label}: {item.count}" for item in benchmark_bundle.secondary_endpoint_family_frequency[:4]),
        inclusion_criteria_themes=benchmark_bundle.inclusion_themes,
        exclusion_criteria_themes=benchmark_bundle.exclusion_themes,
        biomarker_requirement_patterns=benchmark_bundle.biomarker_testing_themes,
        prior_treatment_patterns=benchmark_bundle.prior_treatment_themes,
        safety_monitoring_concepts=benchmark_bundle.safety_exclusion_themes,
        assessment_schedule_concepts=("Use follow-on CT.gov visit/endpoint schedules as precedent; exact visit timing requires human protocol authoring.",),
        treatment_duration_patterns=_duration_patterns(follow_on_trials),
        follow_up_patterns=("Follow-up approach must be human-reviewed against endpoint timing and safety context.",),
        dominant_patterns=tuple(dict.fromkeys((*randomized[:1], *comparator[:1], *endpoint[:1]))),
        minority_patterns=tuple(dict.fromkeys((*randomized[1:3], *comparator[1:3], *endpoint[1:3]))),
        conflicting_precedent=_conflicting_precedent(benchmark_bundle),
        insufficient_evidence=tuple(insufficient),
        human_review_questions=(
            "Which follow-on precedent patterns are clinically appropriate for the target asset and indication?",
            "Which sparse or conflicting precedent should be excluded before protocol drafting?",
        ),
        source_ids=source_ids,
        confidence=benchmark_bundle.confidence if follow_on_trials else 0.2,
    )


def build_analog_derived_design_decisions(
    *,
    run_id: str,
    follow_on_trials: tuple[ClinicalTrialRecord, ...],
    benchmark_bundle: AnalogBenchmarkBundle,
    qualitative_synthesis: QualitativeProtocolSynthesis,
) -> tuple[AnalogDerivedDesignDecision, ...]:
    """Create explicit protocol choices derived from observed follow-on precedent."""

    total = len(follow_on_trials)
    follow_on_ids = tuple(trial.nct_id for trial in follow_on_trials)
    decisions: list[AnalogDerivedDesignDecision] = []
    if benchmark_bundle.enrollment.median is not None:
        decisions.append(
            AnalogDerivedDesignDecision(
                decision_id=f"analog-derived-decision-{run_id}-target-enrollment",
                field_name="proposed_enrollment",
                proposed_value=f"{benchmark_bundle.enrollment.median:g} participants",
                derivation_method="median",
                supporting_follow_on_nct_ids=follow_on_ids,
                observed_count=benchmark_bundle.enrollment.observed_count,
                total_eligible_follow_on_trials=total,
                rationale="Target enrollment derived from median enrollment across selected follow-on trials.",
                source_ids=benchmark_bundle.enrollment.source_ids,
                confidence=benchmark_bundle.confidence,
            )
        )
    if benchmark_bundle.site_count.median is not None:
        decisions.append(
            AnalogDerivedDesignDecision(
                decision_id=f"analog-derived-decision-{run_id}-site-footprint",
                field_name="proposed_site_footprint",
                proposed_value=f"{benchmark_bundle.site_count.median:g} sites",
                derivation_method="median",
                supporting_follow_on_nct_ids=follow_on_ids,
                observed_count=benchmark_bundle.site_count.observed_count,
                total_eligible_follow_on_trials=total,
                rationale="Site footprint derived from median site count across selected follow-on trials.",
                source_ids=benchmark_bundle.site_count.source_ids,
                confidence=benchmark_bundle.confidence,
            )
        )
    for field_name, rows, label in (
        ("randomized_design", benchmark_bundle.randomized_frequency, "Randomization"),
        ("masking_strategy", benchmark_bundle.blinding_frequency, "Masking"),
        ("comparator_control", benchmark_bundle.comparator_categories, "Comparator/control"),
        ("endpoint_strategy", benchmark_bundle.primary_endpoint_family_frequency, "Primary endpoint family"),
    ):
        if not rows:
            continue
        top = rows[0]
        decisions.append(
            AnalogDerivedDesignDecision(
                decision_id=f"analog-derived-decision-{run_id}-{slug(field_name)}",
                field_name=field_name,
                proposed_value=top.label,
                derivation_method="frequency",
                supporting_follow_on_nct_ids=follow_on_ids,
                observed_count=top.count,
                total_eligible_follow_on_trials=total,
                rationale=f"{label} choice derived from the most frequent observed category across selected follow-on trials.",
                source_ids=top.source_ids,
                confidence=min(benchmark_bundle.confidence, top.frequency),
            )
        )
    for field_name, patterns in (
        ("target_population", qualitative_synthesis.target_population_patterns),
        ("eligibility_framework", qualitative_synthesis.inclusion_criteria_themes),
        ("safety_monitoring_framework", qualitative_synthesis.safety_monitoring_concepts),
        ("schedule_of_assessments_framework", qualitative_synthesis.assessment_schedule_concepts),
    ):
        if not patterns:
            continue
        decisions.append(
            AnalogDerivedDesignDecision(
                decision_id=f"analog-derived-decision-{run_id}-{slug(field_name)}",
                field_name=field_name,
                proposed_value="; ".join(patterns[:4]),
                derivation_method="qualitative_consensus",
                supporting_follow_on_nct_ids=follow_on_ids,
                observed_count=total,
                total_eligible_follow_on_trials=total,
                rationale="Qualitative design choice derived from recurring selected follow-on trial patterns.",
                source_ids=qualitative_synthesis.source_ids,
                confidence=qualitative_synthesis.confidence,
            )
        )
    return tuple(decisions)


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
    subject = "selected follow-on CT.gov trials" if any("follow-on" in item.casefold() for item in bundle.limitations) else "selected CT.gov analog trials"
    enrollment = _stat_phrase(bundle.enrollment)
    duration = _stat_phrase(bundle.planned_duration_months)
    endpoints = ", ".join(f"{item.label}: {item.count}" for item in bundle.primary_endpoint_family_frequency[:4]) or "no classified primary endpoint families"
    controls = ", ".join(f"{item.label}: {item.count}" for item in bundle.comparator_categories[:4]) or "no comparator categories detected"
    return (
        f"Benchmark uses {selected} {subject}. "
        f"Enrollment {enrollment}; planned duration {duration}. "
        f"Primary endpoint families: {endpoints}. Comparator categories: {controls}. "
        "Limitations and missing fields are carried as flags."
    )


def build_protocol_design_brief(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    next_study_intent: NextStudyIntent,
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
                f"Is the proposed next study ({next_study_intent.proposed_next_stage}; {next_study_intent.study_role}) the right next development step?",
                f"What evidence is still needed to resolve the key clinical question: {next_study_intent.key_clinical_question}",
                "Which analog trials should be accepted or rejected by the clinical team?",
                "Which missing CT.gov, PubMed, or label fields must be resolved before protocol drafting?",
            )
        )
    )
    return ProtocolDesignBrief(
        brief_id=f"protocol-design-brief-{run_id}",
        title=f"Draft {next_study_intent.proposed_next_stage} Protocol Design Brief for {target_trial.nct_id}",
        next_study_intent=next_study_intent,
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
    return _clinical_trials_input_variants(query)[0][1]


def _clinical_trials_input_variants(query: object) -> tuple[tuple[str, ClinicalTrialIntelligenceInput], ...]:
    """Convert an AI search query into specific but API-safe CT.gov attempts."""

    condition = str(getattr(query, "condition", "") or "").strip()
    phase = str(getattr(query, "phase", "") or "").strip() or None
    raw_intervention = str(getattr(query, "intervention", "") or "").strip() or None
    safe_intervention = raw_intervention if raw_intervention and not _is_generic_or_control_intervention(raw_intervention) else None
    terms = _ai_query_terms(query, raw_intervention=raw_intervention, safe_intervention=safe_intervention)
    limit = min(int(getattr(query, "limit", 25) or 25), 50)
    attempts: list[tuple[str, ClinicalTrialIntelligenceInput]] = []

    def add(label: str, *, drug: str | None, target: str | None, phase_value: str | None) -> None:
        if not condition:
            return
        input_data = ClinicalTrialIntelligenceInput(
            disease=condition,
            drug=drug,
            target=target,
            phase=phase_value,
            limit=limit,
        )
        signature = (input_data.disease, input_data.drug, input_data.target, input_data.phase, input_data.limit)
        if not any((existing.disease, existing.drug, existing.target, existing.phase, existing.limit) == signature for _, existing in attempts):
            attempts.append((label, input_data))

    add("ai_specific", drug=safe_intervention, target=terms, phase_value=phase)
    if safe_intervention:
        add("ai_specific_no_intervention_filter", drug=None, target=terms, phase_value=phase)
    add("ai_terms_no_phase", drug=safe_intervention, target=terms, phase_value=None)
    add("condition_phase", drug=None, target=None, phase_value=phase)
    add("condition_only", drug=None, target=None, phase_value=None)
    return tuple(attempts)


def _ai_query_terms(query: object, *, raw_intervention: str | None, safe_intervention: str | None) -> str | None:
    values: list[str] = []
    if raw_intervention and raw_intervention != safe_intervention and not _is_empty_generic_intervention(raw_intervention):
        values.append(raw_intervention)
    for value in (
        getattr(query, "target_or_moa", None),
        getattr(query, "endpoint_family", None),
        getattr(query, "comparator", None),
        getattr(query, "biomarker_or_line", None),
        getattr(query, "term", None),
    ):
        if value:
            values.append(str(value))
    terms = " ".join(" ".join(values).split())
    return terms or None


def _is_generic_or_control_intervention(value: str) -> bool:
    normalized = slug(value)
    if not normalized:
        return True
    exact = {
        "placebo",
        "drug",
        "therapy",
        "treatment",
        "investigationaldrug",
        "investigationalproduct",
        "standardofcare",
        "bestsupportivecare",
        "supportivecare",
        "control",
        "vehicle",
    }
    if normalized in exact:
        return True
    return any(token in normalized for token in ("placebo", "standardofcare", "bestsupportivecare", "vehicle", "control"))


def _is_empty_generic_intervention(value: str) -> bool:
    return slug(value) in {"drug", "therapy", "treatment", "investigationaldrug", "investigationalproduct"}


def _source_for_trial(trial: ClinicalTrialRecord) -> SourceMetadata:
    return SourceMetadata(
        source_id=trial.source_id,
        title=trial.brief_title or trial.official_title or trial.nct_id,
        url=f"https://clinicaltrials.gov/study/{trial.nct_id}",
        authors=tuple(sponsor.name for sponsor in (trial.lead_sponsor, *trial.collaborators) if sponsor is not None),
        provenance="ClinicalTrials.gov API v2 protocolSection",
        source_type="clinical_trial_registry",
        version="v2",
    )


def _similarity_features(target: ClinicalTrialRecord, candidate: ClinicalTrialRecord) -> dict[str, object]:
    target_endpoint = _endpoint_family_from_trial(target)
    candidate_endpoint = _endpoint_family_from_trial(candidate)
    target_comparator = _comparator_category(target)
    candidate_comparator = _comparator_category(candidate)
    return {
        "same_indication": _indication_matches(target.conditions, candidate.conditions),
        "same_or_comparable_phase": bool(set(_norm_phase_values(target.phases)) & set(_norm_phase_values(candidate.phases))),
        "same_endpoint_family": bool(target_endpoint and target_endpoint == candidate_endpoint),
        "same_comparator_structure": bool(target_comparator and target_comparator == candidate_comparator),
        "same_modality": _intervention_type_set(target) == _intervention_type_set(candidate) if _intervention_type_set(target) and _intervention_type_set(candidate) else None,
        "similar_population": _population_context(target) == _population_context(candidate) if _population_context(target) and _population_context(candidate) else None,
        "similar_biomarker_or_line": _has_any_keyword_overlap(target.eligibility_criteria, candidate.eligibility_criteria, ("biomarker", "mutation", "refractory", "prior", "line")),
        "target_or_moa_overlap": _has_any_keyword_overlap(_intervention_text(target), _intervention_text(candidate), ("inhibitor", "antibody", "agonist", "antagonist", "kinase", "receptor")),
        "current_trial_anchor": target.nct_id,
    }


def _asset_aliases(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    aliases: list[str] = []
    excluded = {"placebo", "standard of care", "best supportive care", "control", "vehicle"}
    excluded_tokens = ("placebo", "control", "standardofcare", "bestsupportivecare", "vehicle")
    for intervention in trial.interventions:
        names = (intervention.name, *intervention.other_names)
        for name in names:
            cleaned = _clean_intervention_name(name)
            cleaned_slug = slug(cleaned) if cleaned else ""
            if cleaned and cleaned.casefold() not in excluded and not any(token in cleaned_slug for token in excluded_tokens):
                aliases.append(cleaned)
    return tuple(dict.fromkeys(aliases))


def _expanded_asset_aliases(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    aliases: list[str] = []
    for alias in _asset_aliases(trial):
        aliases.append(alias)
        base = _strip_dose_suffix(alias)
        if base and base != alias:
            aliases.append(base)
        aliases.extend(_known_asset_aliases(alias))
    return tuple(dict.fromkeys(alias for alias in aliases if alias and not _is_generic_or_control_intervention(alias)))


def _strip_dose_suffix(value: str) -> str | None:
    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|%)\b.*$", "", value, flags=re.I).strip(" -/,:;")
    return text or None


def _known_asset_aliases(value: str) -> tuple[str, ...]:
    alias_sets = {
        "cp-690-550": ("CP-690,550", "CP 690,550", "CP-690550", "tofacitinib", "tasocitinib"),
        "cp-690550": ("CP-690,550", "CP 690,550", "CP-690550", "tofacitinib", "tasocitinib"),
        "cp690550": ("CP-690,550", "CP 690,550", "CP-690550", "tofacitinib", "tasocitinib"),
        "tofacitinib": ("tofacitinib", "CP-690,550", "CP 690,550", "CP-690550", "tasocitinib"),
        "tasocitinib": ("tasocitinib", "tofacitinib", "CP-690,550", "CP 690,550", "CP-690550"),
        "ain457": ("AIN457", "secukinumab"),
        "secukinumab": ("secukinumab", "AIN457"),
    }
    return tuple(alias_sets.get(slug(value), ()))


def _equivalent_asset_slugs(value: str) -> set[str]:
    values = {value, *_known_asset_aliases(value)}
    stripped = _strip_dose_suffix(value)
    if stripped:
        values.add(stripped)
        values.update(_known_asset_aliases(stripped))
    return {slug(item) for item in values if item}


def _follow_on_indication_terms(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    terms: list[str] = []
    for condition in trial.conditions:
        terms.extend(_condition_variants(condition))
    for title in (trial.brief_title, trial.official_title):
        if not title:
            continue
        match = re.search(r"(?:subjects|patients|adults?) with (?P<phrase>.+?)(?:\s+\([^)]*\)|$)", title, re.I)
        if match:
            terms.extend(_condition_variants(match.group("phrase").strip(" .,:;")))
    return tuple(dict.fromkeys(term for term in terms if term))


def _condition_variants(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    text = " ".join(str(value).split()).strip(" .,:;")
    variants = [text]
    lowered = text.casefold()
    for prefix in ("moderate to severe ", "moderate-to-severe ", "severe "):
        if lowered.startswith(prefix):
            variants.append(text[len(prefix) :].strip())
    if "plaque psoriasis" in lowered:
        variants.append("Psoriasis")
    if "atopic dermatitis" in lowered:
        variants.append("Atopic Dermatitis")
    return tuple(dict.fromkeys(item for item in variants if item))


def _follow_on_search_inputs(
    *,
    alias: str,
    indication_terms: tuple[str, ...],
) -> tuple[tuple[str, ClinicalTrialIntelligenceInput], ...]:
    attempts: list[tuple[str, ClinicalTrialIntelligenceInput]] = []

    def add(label: str, *, disease: str, drug: str | None, target: str | None = None) -> None:
        input_data = ClinicalTrialIntelligenceInput(disease=disease, drug=drug, target=target, limit=50)
        signature = (input_data.disease, input_data.drug, input_data.target, input_data.phase, input_data.limit)
        if not any((existing.disease, existing.drug, existing.target, existing.phase, existing.limit) == signature for _, existing in attempts):
            attempts.append((label, input_data))

    for index, disease in enumerate(indication_terms[:4], start=1):
        add(f"condition_drug_{index}", disease=disease, drug=alias)
        add(f"condition_term_{index}", disease=disease, drug=None, target=alias)
    return tuple(attempts)


def _clean_intervention_name(name: str | None) -> str | None:
    if not name:
        return None
    text = str(name).strip()
    for prefix in ("Drug:", "Biological:", "Biologic:", "Device:", "Procedure:", "Radiation:", "Combination Product:"):
        if text.casefold().startswith(prefix.casefold()):
            text = text[len(prefix) :].strip()
    return text or None


def _sponsor_names(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    return tuple(dict.fromkeys(sponsor.name for sponsor in (trial.lead_sponsor, *trial.collaborators) if sponsor and sponsor.name))


def _lineage_features(*, anchor: ClinicalTrialRecord, candidate: ClinicalTrialRecord, alias: str) -> dict[str, object]:
    same_asset = _asset_matches(candidate, alias)
    same_indication = _indication_matches(anchor.conditions, candidate.conditions)
    same_sponsor = bool(set(map(slug, _sponsor_names(anchor))) & set(map(slug, _sponsor_names(candidate))))
    candidate_start = _parse_date(candidate.start_date)
    anchor_start = _parse_date(anchor.start_date)
    chronology = None
    if candidate_start and anchor_start:
        chronology = "later_start" if candidate_start >= anchor_start else "earlier_start"
    return {
        "anchor_nct_id": anchor.nct_id,
        "candidate_nct_id": candidate.nct_id,
        "asset_alias_used": alias,
        "same_normalized_asset": same_asset,
        "same_indication": same_indication,
        "same_sponsor": same_sponsor,
        "chronology": chronology,
        "anchor_phase": anchor.phases,
        "candidate_phase": candidate.phases,
        "phase_progression_rank": _phase_progression_rank_for_trials(anchor, candidate),
        "anchor_study_type": anchor.study_type,
        "candidate_study_type": candidate.study_type,
        "population_continuity": _population_context(anchor) == _population_context(candidate) if _population_context(anchor) and _population_context(candidate) else None,
        "endpoint_continuity": _endpoint_family_from_trial(anchor) == _endpoint_family_from_trial(candidate) if _endpoint_family_from_trial(anchor) and _endpoint_family_from_trial(candidate) else None,
        "comparator_change": _comparator_category(anchor) != _comparator_category(candidate) if _comparator_category(anchor) and _comparator_category(candidate) else None,
    }


def _asset_matches(trial: ClinicalTrialRecord, alias: str) -> bool:
    alias_slugs = _equivalent_asset_slugs(alias)
    for name in _expanded_asset_aliases(trial):
        name_slugs = _equivalent_asset_slugs(name)
        if alias_slugs & name_slugs:
            return True
        if any(left in right or right in left for left in alias_slugs for right in name_slugs):
            return True
    return False


def _indication_matches(left_terms: tuple[str, ...], right_terms: tuple[str, ...]) -> bool:
    left = {slug(term) for term in left_terms if term}
    right = {slug(term) for term in right_terms if term}
    if left & right:
        return True
    return any(
        (len(left_term) >= 5 and left_term in right_term)
        or (len(right_term) >= 5 and right_term in left_term)
        for left_term in left
        for right_term in right
    )


def _follow_on_exclusion_reason(features: dict[str, object]) -> str | None:
    if features.get("anchor_nct_id") == features.get("candidate_nct_id"):
        return "analog_itself"
    if not features.get("same_normalized_asset"):
        return "asset_mismatch"
    if not features.get("same_indication"):
        return "indication_mismatch"
    if not features.get("same_sponsor"):
        return "sponsor_mismatch"
    if features.get("chronology") == "earlier_start":
        return "clearly_earlier_trial"
    anchor_type = str(features.get("anchor_study_type") or "").casefold()
    candidate_type = str(features.get("candidate_study_type") or "").casefold()
    if "interventional" in anchor_type and "observational" in candidate_type:
        return "observational_candidate_for_interventional_program"
    return None


def _phase_progression_rank(candidate: FollowOnCandidateRecord) -> int:
    return int(candidate.lineage_features.get("phase_progression_rank") or 0)


def _phase_progression_rank_for_trials(anchor: ClinicalTrialRecord, candidate: ClinicalTrialRecord) -> int:
    anchor_rank = max((_phase_rank(value) for value in anchor.phases), default=0)
    candidate_rank = max((_phase_rank(value) for value in candidate.phases), default=0)
    return candidate_rank - anchor_rank


def _phase_rank(value: str | None) -> int:
    text = slug(value or "")
    if "phase4" in text:
        return 4
    if "phase3" in text:
        return 3
    if "phase2" in text:
        return 2
    if "phase1" in text:
        return 1
    return 0


def _norm_phase_values(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        text = slug(value)
        if "phase4" in text:
            normalized.append("phase4")
        elif "phase3" in text:
            normalized.append("phase3")
        elif "phase2" in text:
            normalized.append("phase2")
        elif "phase1" in text:
            normalized.append("phase1")
    return tuple(dict.fromkeys(normalized))


def _date_sort_key(value: str | None) -> str:
    parsed = _parse_date(value)
    return parsed.isoformat() if parsed else ""


def _population_context(trial: ClinicalTrialRecord) -> str | None:
    parts = [trial.minimum_age, trial.maximum_age, trial.sex]
    text = " ".join(part for part in parts if part)
    return slug(text) if text else None


def _population_pattern(trial: ClinicalTrialRecord) -> str | None:
    items = [trial.minimum_age, trial.maximum_age, trial.sex]
    if not any(items):
        return None
    return " / ".join(item for item in items if item)


def _duration_patterns(trials: tuple[ClinicalTrialRecord, ...]) -> tuple[str, ...]:
    durations = [
        duration
        for trial in trials
        if (duration := _planned_duration_months(trial.start_date, trial.primary_completion_date or trial.completion_date)) is not None
    ]
    if not durations:
        return ()
    return (f"Observed planned duration median {round(median(durations), 1)} months across {len(durations)} follow-on trials.",)


def _conflicting_precedent(bundle: AnalogBenchmarkBundle) -> tuple[str, ...]:
    conflicts = []
    for label, rows in (
        ("randomization", bundle.randomized_frequency),
        ("masking", bundle.blinding_frequency),
        ("comparator/control", bundle.comparator_categories),
        ("primary endpoint family", bundle.primary_endpoint_family_frequency),
    ):
        if len(rows) > 1 and rows[0].frequency < 0.7:
            conflicts.append(f"No dominant {label} precedent; top category {rows[0].label} has frequency {rows[0].frequency}.")
    return tuple(conflicts)


def _endpoint_family_from_trial(trial: ClinicalTrialRecord) -> str | None:
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


def _intervention_type_set(trial: ClinicalTrialRecord) -> set[str]:
    return {slug(item.type or "") for item in trial.interventions if item.type}


def _intervention_text(trial: ClinicalTrialRecord) -> str:
    return " ".join(
        item
        for intervention in trial.interventions
        for item in (intervention.name, intervention.description, *intervention.other_names)
        if item
    )


def _has_any_keyword_overlap(left: str | None, right: str | None, keywords: tuple[str, ...]) -> bool | None:
    if not left or not right:
        return None
    left_text = left.casefold()
    right_text = right.casefold()
    return any(keyword in left_text and keyword in right_text for keyword in keywords)


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
    if trial.number_of_arms is not None:
        return trial.number_of_arms
    if trial.arm_groups:
        return len(trial.arm_groups)
    return max(1, len(trial.interventions)) if trial.interventions else 0


def _design_labels(selected: tuple[AnalogCandidateRecord, ...], field: str) -> tuple[str, ...]:
    labels = []
    for candidate in selected:
        structured_label = _structured_design_label(candidate.trial, field)
        if structured_label:
            labels.append(structured_label)
            continue
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


def _structured_design_label(trial: ClinicalTrialRecord, field: str) -> str | None:
    if field == "randomized":
        allocation = (trial.allocation or "").casefold()
        if not allocation:
            return None
        if "non" in allocation or "not random" in allocation:
            return "non_randomized"
        if "random" in allocation:
            return "randomized"
        return None

    masking = (trial.masking or "").casefold()
    if not masking:
        return None
    if "none" in masking or "open" in masking:
        return "open_label"
    if any(term in masking for term in ("single", "double", "triple", "quadruple", "mask", "blind")):
        return "blinded_or_masked"
    return None


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
    structured = _structured_comparator_category(trial)
    if structured:
        return structured
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


def _structured_comparator_category(trial: ClinicalTrialRecord) -> str | None:
    arm_texts = [
        " ".join(
            value
            for value in (arm.label, arm.type or "", arm.description or "", " ".join(arm.intervention_names))
            if value
        ).casefold()
        for arm in trial.arm_groups
    ]
    mapped_interventions = {
        intervention.name.casefold()
        for intervention in trial.interventions
        if intervention.arm_group_labels
    }
    mapping_texts = [
        " ".join((intervention.name, " ".join(intervention.arm_group_labels))).casefold()
        for intervention in trial.interventions
        if intervention.arm_group_labels
    ]
    texts = tuple(text for text in (*arm_texts, *mapping_texts) if text.strip())
    if not texts and not trial.arm_groups:
        return None
    combined = " ".join(texts)
    if "placebo" in combined:
        return "placebo_control"
    if "standard of care" in combined or "best supportive" in combined:
        return "standard_of_care"
    if "control" in combined or "comparator" in combined:
        return "control"
    if len(trial.arm_groups) == 1 or trial.number_of_arms == 1:
        return "single_arm_or_uncontrolled"
    if len(trial.arm_groups) > 1 or len(mapped_interventions) > 1 or (trial.number_of_arms or 0) > 1:
        return "active_comparator_or_combination"
    return None


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
