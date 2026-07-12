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
    ExcludedAnalogTrial,
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
    UnevaluableAnalogTrial,
)
from pharma_os.tools._due_diligence_common import missing, slug
from pharma_os.tools.clinical_semantics import (
    active_intervention_text,
    asset_aliases as semantic_asset_aliases,
    asset_matches as semantic_asset_matches,
    comparable_modality,
    condition_variants as semantic_condition_variants,
    endpoint_family as semantic_endpoint_family,
    endpoint_family_from_trial as semantic_endpoint_family_from_trial,
    expanded_asset_aliases as semantic_expanded_asset_aliases,
    normalized_sponsor_names,
    route_set,
    same_endpoint_domain,
    same_indication as semantic_same_indication,
    strip_dose_suffix as semantic_strip_dose_suffix,
)
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


def score_current_trial_candidate(
    target_trial: ClinicalTrialRecord,
    candidate: AnalogCandidateRecord,
) -> tuple[SelectedAnalogTrial, AnalogCandidateRecord]:
    """Score a candidate against the current target trial using normalized semantics."""

    trial = candidate.trial
    features = candidate.similarity_features or _similarity_features(target_trial, trial)
    matched: list[str] = []
    mismatched: list[str] = []
    unknown: list[str] = []
    weights = {
        "same_indication": 0.28,
        "same_clinical_endpoint_domain": 0.16,
        "same_endpoint_family": 0.12,
        "same_comparator_structure": 0.1,
        "same_modality": 0.1,
        "same_or_comparable_phase": 0.08,
        "randomized_placebo_controlled": 0.07,
        "similar_population": 0.04,
        "similar_biomarker_or_line": 0.03,
        "target_or_moa_overlap": 0.02,
    }
    score = 0.0
    for key, weight in weights.items():
        value = features.get(key)
        if value is True:
            matched.append(key)
            score += weight
        elif value is False:
            mismatched.append(key)
        else:
            unknown.append(key)
    if features.get("primary_endpoint_is_pk_or_safety") is True:
        mismatched.append("primary_endpoint_is_pk_or_safety")
        score -= 0.12
    if features.get("wrong_active_route") is True:
        mismatched.append("wrong_active_route")
        score -= 0.08
    score = max(0.0, min(1.0, score))
    confidence = "high" if score >= 0.7 else "medium" if score >= 0.45 else "low"
    return (
        SelectedAnalogTrial(
            nct_id=trial.nct_id,
            match_score=round(score, 3),
            match_confidence=confidence,
            matched_dimensions=tuple(dict.fromkeys(matched)),
            mismatched_dimensions=tuple(dict.fromkeys(mismatched)),
            unknown_dimensions=tuple(dict.fromkeys(unknown)),
            reasoning="Selected by deterministic semantic similarity repair over indication, modality, endpoint family, design, comparator, and population.",
            source_ids=candidate.source_ids,
        ),
        candidate,
    )


def normalize_analog_selection_output(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    selection: AnalogTrialSelectionOutput,
    candidates: tuple[AnalogCandidateRecord, ...],
    top_k: int,
) -> AnalogTrialSelectionOutput:
    """Repair live/fallback selection so every retrieved candidate has one disposition."""

    candidate_by_id = {candidate.trial.nct_id: candidate for candidate in candidates}
    scored = [
        score_current_trial_candidate(target_trial, candidate)
        for candidate in candidates
        if candidate.trial.nct_id != target_trial.nct_id
    ]
    scored.sort(key=lambda item: (-item[0].match_score, item[0].nct_id))
    selected_rows = tuple(row for row in scored[:top_k] if row[0].match_score >= 0.35)
    selected_by_id = {row[0].nct_id: row[0] for row in selected_rows}
    if not selected_by_id:
        for selected in selection.selected_analogs:
            if selected.nct_id in candidate_by_id and selected.nct_id != target_trial.nct_id:
                selected_by_id[selected.nct_id] = selected
    selected_ids = set(selected_by_id)
    live_exclusions = {
        item.nct_id: item
        for item in selection.excluded_candidates
        if item.nct_id in candidate_by_id and item.nct_id not in selected_ids
    }
    unevaluable: list[UnevaluableAnalogTrial] = []
    excluded: list[ExcludedAnalogTrial] = []
    score_by_id = {row[0].nct_id: row[0] for row in scored}
    for candidate in candidates:
        nct_id = candidate.trial.nct_id
        if nct_id in selected_ids:
            continue
        if nct_id == target_trial.nct_id:
            excluded.append(
                ExcludedAnalogTrial(
                    nct_id=nct_id,
                    reason="Target trial was retrieved and excluded from analog benchmarking.",
                    source_ids=candidate.source_ids,
                )
            )
            continue
        if _candidate_unevaluable_reason(candidate.trial):
            unevaluable.append(
                UnevaluableAnalogTrial(
                    nct_id=nct_id,
                    reason=_candidate_unevaluable_reason(candidate.trial) or "Candidate lacked required structured fields for semantic analog adjudication.",
                    source_ids=candidate.source_ids,
                )
            )
            continue
        if nct_id in live_exclusions:
            excluded.append(live_exclusions[nct_id])
            continue
        score = score_by_id.get(nct_id)
        reason = (
            "Candidate ranked below the deterministic top-k analog cutoff after semantic normalization."
            if score and score.match_score >= 0.35
            else "Candidate did not meet the deterministic semantic similarity threshold after normalization."
        )
        excluded.append(
            ExcludedAnalogTrial(
                nct_id=nct_id,
                reason=reason,
                source_ids=candidate.source_ids,
            )
        )
    selected = tuple(selected_by_id[nct_id] for nct_id in selected_by_id)
    return selection.model_copy(
        update={
            "output_id": selection.output_id or f"analog-selection-{run_id}",
            "selected_analogs": selected,
            "excluded_candidates": tuple(_dedupe_disposition_rows(excluded)),
            "unevaluable_candidates": tuple(_dedupe_disposition_rows(unevaluable)),
            "source_ids": tuple(
                dict.fromkeys(
                    (
                        *selection.source_ids,
                        *(source_id for candidate in candidates for source_id in candidate.source_ids),
                    )
                )
            ),
            "confidence": max(selection.confidence, 0.75 if selected else 0.25),
        }
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
        for alias in aliases[:16]:
            query_source = SourceMetadata(
                source_id=f"ctgov_search:protocol_design_follow_on:{slug(analog.trial.nct_id)}:{slug(alias)}",
                title=f"Follow-on search for {analog.trial.nct_id} using {alias}",
                provenance="ClinicalTrials.gov API v2 search constrained post hoc to same asset, indication, and sponsor",
                source_type="clinical_trial_registry_search",
                version="v2",
            )
            attempt_notes: list[str] = []
            results: list[ClinicalTrialsSearchResult] = []
            for attempt_label, input_data in _follow_on_search_inputs(alias=alias, indication_terms=indication_terms, sponsors=sponsors):
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
            status = "retrieval_failed" if not candidates else "no_plausible_follow_on"
            rationale = (
                "No candidate records were retrieved for this analog despite bounded alias and sponsor follow-on searches."
                if not candidates
                else "Candidates were retrieved, but none survived deterministic same-program lineage filters."
            )
            ambiguity = ("follow_on_retrieval_empty",) if not candidates else ("no_viable_follow_on_candidates",)
            adjudications.append(
                FollowOnTrialAdjudication(
                    anchor_analog_nct_id=analog.trial.nct_id,
                    status=status,
                    excluded_follow_on_nct_ids=excluded,
                    rationale=rationale,
                    ambiguity_flags=ambiguity,
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
        unevaluable_candidates=(),
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
    return bundle.model_copy(update={"evidence_mode": "follow_on", "limitations": limitations})


def build_qualitative_protocol_synthesis(
    *,
    run_id: str,
    target_trial: ClinicalTrialRecord,
    follow_on_trials: tuple[ClinicalTrialRecord, ...],
    benchmark_bundle: AnalogBenchmarkBundle,
) -> QualitativeProtocolSynthesis:
    """Deterministic fallback synthesis of recurring protocol patterns."""

    source_ids = tuple(dict.fromkeys((*benchmark_bundle.source_ids, *(trial.source_id for trial in follow_on_trials))))
    evidence_mode = benchmark_bundle.evidence_mode
    subject = "follow-on trials" if evidence_mode == "follow_on" and follow_on_trials else "direct selected analog trials"
    comparator = tuple(f"{item.label}: {item.count}/{sum(row.count for row in benchmark_bundle.comparator_categories) or item.count}" for item in benchmark_bundle.comparator_categories[:4])
    randomized = tuple(f"{item.label}: {item.count}" for item in benchmark_bundle.randomized_frequency[:4])
    endpoint = tuple(f"{item.label}: {item.count}" for item in benchmark_bundle.primary_endpoint_family_frequency[:4])
    phases = tuple(f"{item.label}: {item.count}" for item in _frequency(tuple(", ".join(trial.phases) or "unknown" for trial in follow_on_trials), source_ids)[:4])
    insufficient = []
    if evidence_mode == "follow_on" and len(follow_on_trials) < 2:
        insufficient.append("Fewer than two selected follow-on trials; qualitative precedent is sparse.")
    if evidence_mode != "follow_on":
        insufficient.append("No lineage-confirmed follow-on trials were found; synthesis uses direct selected analogs as weaker evidence.")
    if not endpoint:
        insufficient.append(f"No recurring primary endpoint pattern was available from {subject}.")
    return QualitativeProtocolSynthesis(
        output_id=f"qualitative-protocol-synthesis-{run_id}",
        target_nct_id=target_trial.nct_id,
        study_role_patterns=phases or ("No lineage-confirmed follow-on role pattern was identified.",),
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
        assessment_schedule_concepts=(f"Use {subject} endpoint timing and schedule fields as review inputs; exact visit timing requires human protocol authoring.",),
        treatment_duration_patterns=_duration_patterns(follow_on_trials),
        follow_up_patterns=("Follow-up approach must be human-reviewed against endpoint timing and safety context.",),
        dominant_patterns=tuple(dict.fromkeys((*randomized[:1], *comparator[:1], *endpoint[:1]))),
        minority_patterns=tuple(dict.fromkeys((*randomized[1:3], *comparator[1:3], *endpoint[1:3]))),
        conflicting_precedent=_conflicting_precedent(benchmark_bundle),
        insufficient_evidence=tuple(insufficient),
        human_review_questions=(
            f"Which {subject} patterns are clinically appropriate for the target asset and indication?",
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
    """Create explicit protocol choices with evidence-mode support labels."""

    total = len(follow_on_trials)
    follow_on_ids = tuple(trial.nct_id for trial in follow_on_trials)
    analog_ids = benchmark_bundle.selected_analog_ids if benchmark_bundle.evidence_mode != "follow_on" else ()
    evidence_subject = _decision_evidence_subject(benchmark_bundle.evidence_mode, total)
    decisions: list[AnalogDerivedDesignDecision] = []
    if benchmark_bundle.enrollment.median is not None:
        decisions.append(
            AnalogDerivedDesignDecision(
                decision_id=f"analog-derived-decision-{run_id}-target-enrollment",
                field_name="proposed_enrollment",
                proposed_value=f"{_rounded_count(benchmark_bundle.enrollment.median)} participants",
                derivation_method="median",
                support_source_type=_support_source_type(
                    benchmark_bundle=benchmark_bundle,
                    observed_count=benchmark_bundle.enrollment.observed_count,
                    total_follow_on=total,
                ),
                supporting_follow_on_nct_ids=follow_on_ids,
                supporting_analog_nct_ids=analog_ids,
                observed_count=benchmark_bundle.enrollment.observed_count,
                total_eligible_follow_on_trials=total,
                rationale=f"Target enrollment derived from median enrollment across {evidence_subject}; rounded to a whole participant count.",
                source_ids=benchmark_bundle.enrollment.source_ids,
                confidence=benchmark_bundle.confidence,
            )
        )
    if benchmark_bundle.site_count.median is not None:
        decisions.append(
            AnalogDerivedDesignDecision(
                decision_id=f"analog-derived-decision-{run_id}-site-footprint",
                field_name="proposed_site_footprint",
                proposed_value=f"{_rounded_count(benchmark_bundle.site_count.median)} sites",
                derivation_method="median",
                support_source_type=_support_source_type(
                    benchmark_bundle=benchmark_bundle,
                    observed_count=benchmark_bundle.site_count.observed_count,
                    total_follow_on=total,
                ),
                supporting_follow_on_nct_ids=follow_on_ids,
                supporting_analog_nct_ids=analog_ids,
                observed_count=benchmark_bundle.site_count.observed_count,
                total_eligible_follow_on_trials=total,
                rationale=f"Site footprint derived from median site count across {evidence_subject}; rounded to a whole site count.",
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
                support_source_type=_support_source_type(
                    benchmark_bundle=benchmark_bundle,
                    observed_count=top.count,
                    total_follow_on=total,
                    frequency=top.frequency,
                ),
                supporting_follow_on_nct_ids=follow_on_ids,
                supporting_analog_nct_ids=analog_ids,
                observed_count=top.count,
                total_eligible_follow_on_trials=total,
                rationale=f"{label} choice derived from the most frequent observed category across {evidence_subject}.",
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
        qualitative_observed = total if benchmark_bundle.evidence_mode == "follow_on" else len(benchmark_bundle.selected_analog_ids)
        decisions.append(
            AnalogDerivedDesignDecision(
                decision_id=f"analog-derived-decision-{run_id}-{slug(field_name)}",
                field_name=field_name,
                proposed_value="; ".join(patterns[:4]),
                derivation_method="qualitative_consensus",
                support_source_type=_support_source_type(
                    benchmark_bundle=benchmark_bundle,
                    observed_count=qualitative_observed,
                    total_follow_on=total,
                ),
                supporting_follow_on_nct_ids=follow_on_ids,
                supporting_analog_nct_ids=analog_ids,
                observed_count=qualitative_observed,
                total_eligible_follow_on_trials=total,
                rationale=f"Qualitative design choice derived from recurring patterns across {evidence_subject}.",
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
    primary_completion_interval_values = [
        duration
        for candidate in selected
        if (duration := _planned_duration_months(candidate.trial.start_date, candidate.trial.primary_completion_date or candidate.trial.completion_date)) is not None
    ]
    total_execution_values = [
        duration
        for candidate in selected
        if (duration := _planned_duration_months(candidate.trial.start_date, candidate.trial.completion_date)) is not None
    ]
    treatment_duration_values = [
        value
        for candidate in selected
        if (value := _trial_treatment_duration_weeks(candidate.trial)) is not None
    ]
    primary_endpoint_timing_values = [
        value
        for candidate in selected
        if (value := _primary_endpoint_timing_weeks(candidate.trial)) is not None
    ]
    follow_up_duration_values = [
        value
        for candidate in selected
        if (value := _trial_follow_up_duration_weeks(candidate.trial)) is not None
    ]
    site_values = [len(candidate.trial.locations) for candidate in selected if candidate.trial.locations]
    return AnalogBenchmarkBundle(
        bundle_id=f"analog-benchmark-{run_id}",
        target_nct_id=target_trial.nct_id,
        evidence_mode="direct_analog",
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
            values=primary_completion_interval_values,
            selected_count=len(selected),
            missing_count=len(selected) - len(primary_completion_interval_values),
            unit="months",
            source_ids=source_ids,
        ),
        treatment_duration_weeks=_numeric_summary(
            values=treatment_duration_values,
            selected_count=len(selected),
            missing_count=len(selected) - len(treatment_duration_values),
            unit="weeks",
            source_ids=source_ids,
        ),
        primary_endpoint_timing_weeks=_numeric_summary(
            values=primary_endpoint_timing_values,
            selected_count=len(selected),
            missing_count=len(selected) - len(primary_endpoint_timing_values),
            unit="weeks",
            source_ids=source_ids,
        ),
        follow_up_duration_weeks=_numeric_summary(
            values=follow_up_duration_values,
            selected_count=len(selected),
            missing_count=len(selected) - len(follow_up_duration_values),
            unit="weeks",
            source_ids=source_ids,
        ),
        enrollment_period_months=BenchmarkNumericSummary(missing_count=len(selected), unit="months", source_ids=source_ids),
        total_study_execution_duration_months=_numeric_summary(
            values=total_execution_values,
            selected_count=len(selected),
            missing_count=len(selected) - len(total_execution_values),
            unit="months",
            source_ids=source_ids,
        ),
        randomized_frequency=_frequency(_design_labels(selected, "randomized"), source_ids),
        blinding_frequency=_frequency(_design_labels(selected, "blinding"), source_ids),
        arm_count_distribution=_frequency(tuple(str(_arm_count(candidate.trial)) for candidate in selected), source_ids),
        primary_endpoint_family_frequency=_frequency(
            tuple(
                semantic_endpoint_family(endpoint.measure, endpoint.description, endpoint.time_frame)
                for candidate in selected
                for endpoint in candidate.trial.primary_endpoints
            ),
            source_ids,
        ),
        secondary_endpoint_family_frequency=_frequency(
            tuple(
                semantic_endpoint_family(endpoint.measure, endpoint.description, endpoint.time_frame)
                for candidate in selected
                for endpoint in candidate.trial.secondary_endpoints
            ),
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
    subject = "selected follow-on CT.gov trials" if bundle.evidence_mode == "follow_on" else "selected CT.gov analog trials"
    enrollment = _stat_phrase(bundle.enrollment)
    primary_interval = _stat_phrase(bundle.planned_duration_months)
    endpoint_timing = _stat_phrase(bundle.primary_endpoint_timing_weeks)
    treatment_duration = _stat_phrase(bundle.treatment_duration_weeks)
    endpoints = ", ".join(f"{item.label}: {item.count}" for item in bundle.primary_endpoint_family_frequency[:4]) or "no classified primary endpoint families"
    controls = ", ".join(f"{item.label}: {item.count}" for item in bundle.comparator_categories[:4]) or "no comparator categories detected"
    return (
        f"Benchmark uses {selected} {subject}. "
        f"Enrollment {enrollment}; primary-completion interval {primary_interval}; "
        f"primary endpoint timing {endpoint_timing}; treatment duration {treatment_duration}. "
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
                claim_text=f"Selected analog median enrollment is {_rounded_count(benchmark_bundle.enrollment.median)} participants.",
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
                claim_text=f"Selected analog median start-to-primary-completion interval is {benchmark_bundle.planned_duration_months.median:.1f} months.",
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
    target_routes = route_set(target)
    candidate_routes = route_set(candidate)
    endpoint_domain = same_endpoint_domain(target_endpoint, candidate_endpoint)
    primary_family = candidate_endpoint
    return {
        "same_indication": _indication_matches(target.conditions, candidate.conditions),
        "same_or_comparable_phase": bool(set(_norm_phase_values(target.phases)) & set(_norm_phase_values(candidate.phases))),
        "same_endpoint_family": bool(target_endpoint and target_endpoint == candidate_endpoint),
        "same_clinical_endpoint_domain": endpoint_domain,
        "same_comparator_structure": bool(target_comparator and target_comparator == candidate_comparator),
        "randomized_placebo_controlled": _structured_design_label(candidate, "randomized") == "randomized" and candidate_comparator == "placebo_control",
        "same_modality": comparable_modality(target, candidate),
        "target_active_routes": tuple(sorted(target_routes)),
        "candidate_active_routes": tuple(sorted(candidate_routes)),
        "wrong_active_route": bool(target_routes and candidate_routes and not (target_routes & candidate_routes)),
        "primary_endpoint_is_pk_or_safety": primary_family in {"pk", "safety"} if primary_family else None,
        "similar_population": _population_context(target) == _population_context(candidate) if _population_context(target) and _population_context(candidate) else None,
        "similar_biomarker_or_line": _has_any_keyword_overlap(target.eligibility_criteria, candidate.eligibility_criteria, ("biomarker", "mutation", "refractory", "prior", "line")),
        "target_or_moa_overlap": _has_any_keyword_overlap(_intervention_text(target), _intervention_text(candidate), ("inhibitor", "antibody", "agonist", "antagonist", "kinase", "receptor")),
        "target_primary_endpoint_family": target_endpoint,
        "candidate_primary_endpoint_family": candidate_endpoint,
        "current_trial_anchor": target.nct_id,
    }


def _asset_aliases(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    return semantic_asset_aliases(trial)


def _expanded_asset_aliases(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    return semantic_expanded_asset_aliases(trial)


def _strip_dose_suffix(value: str) -> str | None:
    return semantic_strip_dose_suffix(value)


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
    return semantic_condition_variants(value)


def _follow_on_search_inputs(
    *,
    alias: str,
    indication_terms: tuple[str, ...],
    sponsors: tuple[str, ...] = (),
) -> tuple[tuple[str, ClinicalTrialIntelligenceInput], ...]:
    attempts: list[tuple[str, ClinicalTrialIntelligenceInput]] = []

    def add(label: str, *, disease: str, drug: str | None, target: str | None = None, sponsor: str | None = None) -> None:
        input_data = ClinicalTrialIntelligenceInput(disease=disease, drug=drug, target=target, sponsor=sponsor, limit=50)
        signature = (input_data.disease, input_data.drug, input_data.target, input_data.sponsor, input_data.phase, input_data.limit)
        if not any((existing.disease, existing.drug, existing.target, existing.sponsor, existing.phase, existing.limit) == signature for _, existing in attempts):
            attempts.append((label, input_data))

    for index, disease in enumerate(indication_terms[:4], start=1):
        add(f"condition_drug_{index}", disease=disease, drug=alias)
        add(f"condition_term_{index}", disease=disease, drug=None, target=alias)
        for sponsor_index, sponsor in enumerate(sponsors[:3], start=1):
            add(f"condition_sponsor_alias_{index}_{sponsor_index}", disease=disease, drug=None, target=alias, sponsor=sponsor)
            add(f"condition_sponsor_rescue_{index}_{sponsor_index}", disease=disease, drug=None, target=None, sponsor=sponsor)
    for sponsor_index, sponsor in enumerate(sponsors[:3], start=1):
        add(f"sponsor_alias_rescue_{sponsor_index}", disease=indication_terms[0], drug=None, target=alias, sponsor=sponsor)
    return tuple(attempts)


def _sponsor_names(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    return normalized_sponsor_names(trial)


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
    return semantic_asset_matches(trial, alias)


def _indication_matches(left_terms: tuple[str, ...], right_terms: tuple[str, ...]) -> bool:
    return semantic_same_indication(left_terms, right_terms)


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
    return (f"Observed start-to-primary-completion interval median {round(median(durations), 1)} months across {len(durations)} trials.",)


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
    return semantic_endpoint_family_from_trial(trial)


def _intervention_text(trial: ClinicalTrialRecord) -> str:
    return active_intervention_text(trial)


def _has_any_keyword_overlap(left: str | None, right: str | None, keywords: tuple[str, ...]) -> bool | None:
    if not left or not right:
        return None
    left_text = left.casefold()
    right_text = right.casefold()
    return any(keyword in left_text and keyword in right_text for keyword in keywords)


def _candidate_unevaluable_reason(trial: ClinicalTrialRecord) -> str | None:
    if not trial.conditions:
        return "Candidate lacks condition fields needed for indication similarity adjudication."
    if not trial.interventions:
        return "Candidate lacks intervention fields needed for active asset modality adjudication."
    if not trial.primary_endpoints and not trial.secondary_endpoints:
        return "Candidate lacks endpoint fields needed for endpoint-family adjudication."
    return None


def _dedupe_disposition_rows(rows: list[ExcludedAnalogTrial] | list[UnevaluableAnalogTrial]) -> tuple[ExcludedAnalogTrial, ...] | tuple[UnevaluableAnalogTrial, ...]:
    by_nct = {}
    for row in rows:
        by_nct.setdefault(row.nct_id, row)
    return tuple(by_nct.values())


def _rounded_count(value: float) -> int:
    return int(value + 0.5)


def _decision_evidence_subject(evidence_mode: str, total_follow_on: int) -> str:
    if evidence_mode == "follow_on" and total_follow_on:
        return "selected follow-on trials"
    if evidence_mode == "direct_analog":
        return "direct selected analog trials"
    if evidence_mode == "target_only":
        return "target-trial evidence"
    return "sparse or unresolved evidence"


def _support_source_type(
    *,
    benchmark_bundle: AnalogBenchmarkBundle,
    observed_count: int,
    total_follow_on: int,
    frequency: float | None = None,
) -> str:
    if benchmark_bundle.evidence_mode == "follow_on":
        if total_follow_on and observed_count:
            return "follow_on_supported"
        return "unresolved"
    if benchmark_bundle.evidence_mode == "direct_analog":
        if not observed_count:
            return "unresolved"
        if frequency is not None and frequency < 0.5:
            return "minority_precedent"
        if observed_count >= max(2, len(benchmark_bundle.selected_analog_ids) // 2 + 1):
            return "analog_majority_supported"
        return "minority_precedent"
    if benchmark_bundle.evidence_mode == "target_only":
        return "target_trial_supported"
    return "human_decision_required"


def _primary_endpoint_timing_weeks(trial: ClinicalTrialRecord) -> float | None:
    values = [
        value
        for endpoint in trial.primary_endpoints
        if (value := _duration_weeks_from_text(" ".join(item for item in (endpoint.measure, endpoint.time_frame, endpoint.description) if item))) is not None
    ]
    return min(values) if values else None


def _trial_treatment_duration_weeks(trial: ClinicalTrialRecord) -> float | None:
    text = " ".join(
        item
        for item in (
            active_intervention_text(trial),
            *(endpoint.time_frame or "" for endpoint in trial.primary_endpoints),
        )
        if item
    )
    return _duration_weeks_from_text(text)


def _trial_follow_up_duration_weeks(trial: ClinicalTrialRecord) -> float | None:
    text = _trial_text(trial)
    if "follow" not in text and "safety" not in text:
        return None
    return _duration_weeks_from_text(text)


def _duration_weeks_from_text(text: str | None) -> float | None:
    if not text:
        return None
    lowered = text.casefold()
    candidates: list[float] = []
    for match in re.finditer(r"\b(?:week|wk)s?\s*(?:up\s*to|through|to|at)?\s*(\d+(?:\.\d+)?)\b", lowered):
        candidates.append(float(match.group(1)))
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*(?:week|wk)s?\b", lowered):
        candidates.append(float(match.group(1)))
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*months?\b", lowered):
        candidates.append(float(match.group(1)) * 4.345)
    for match in re.finditer(r"\b(\d+(?:\.\d+)?)\s*days?\b", lowered):
        candidates.append(float(match.group(1)) / 7)
    if not candidates:
        return None
    return round(max(candidates), 1)


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
    return semantic_endpoint_family(measure)


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
        flags.append(missing("protocol-design-analog-duration-missing", "analog_benchmark", "planned_duration_months", "One or more selected analog trials lack calculable start-to-primary-completion interval.", "medium"))
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
        f"median {_format_summary_value(summary.median, summary.unit)} {summary.unit or ''}, "
        f"mean {_format_summary_value(summary.mean, summary.unit)}, "
        f"range {_format_summary_value(summary.minimum, summary.unit)}-{_format_summary_value(summary.maximum, summary.unit)}, "
        f"IQR {_format_summary_value(summary.iqr, summary.unit)}, missing {summary.missing_count}"
    )


def _format_summary_value(value: float | None, unit: str | None) -> str:
    if value is None:
        return "NA"
    if unit in {"participants", "patients", "subjects", "sites"}:
        return str(_rounded_count(value))
    return f"{value:g}"
