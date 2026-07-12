"""Commercial market sizing and deterministic revenue model for Agent 4."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import re
from typing import Any

from pydantic import BaseModel, Field

from pharma_os.agent_runtime import AgentRuntimeConfig, run_structured_llm_call, runtime_config_for_route
from pharma_os.schemas import (
    AssetIdentityOutput,
    ClinicalEvidenceSummary,
    ClinicalTrialRecord,
    CommercialAssumptionLedgerRecord,
    CommercialAssumptionTriplet,
    CommercialCaseOutput,
    CommercialInputBundle,
    CommercialModelOutput,
    CommercialPenetrationCase,
    CommercialPricingCase,
    MarketSizingInterpretation,
    MissingDataFlag,
    PatientFunnel,
    PricingOutput,
    RevenueForecastYear,
    SelectedPopulationMeasure,
    SourceMetadata,
    ValueTriplet,
)
from pharma_os.tools._due_diligence_common import (
    assumption,
    missing,
    select_assumption_value,
    to_float,
    triplet_base,
)
from pharma_os.tools.census import CensusPopulationClient, CensusPopulationError
from pharma_os.tools.pubmed import PubMedArticle, PubMedClient, PubMedError
from pharma_os.tools.rules import config_provenance, config_source_id, load_config


class CommercialModelRunResult(BaseModel):
    """Commercial model output plus the optional market-sizing agent trace."""

    output: CommercialModelOutput
    agent_trace: object | None = None
    agent_traces: tuple[object, ...] = ()
    sources: tuple[SourceMetadata, ...] = ()
    trace_metadata: dict[str, str | int | float | bool | None] = {}


class EpidemiologyPopulationCandidate(BaseModel):
    """Candidate market-sizing denominator extracted from epidemiology evidence."""

    value: float | None = None
    unit: str | None = None
    measure_type: str | None = None
    condition: str | None = None
    geography: str | None = None
    source_type: str = "source_derived"
    evidence_reference: str | None = None
    source_id: str | None = None
    rationale: str = ""
    confidence_score: int = Field(default=0, ge=0, le=10)
    human_review_required: bool = True


class EpidemiologyEvidenceExtraction(BaseModel):
    """AI-extracted epidemiology inputs for the commercial market-sizing bundle."""

    evidence_status: str = "not_run"
    selected_population_measure: SelectedPopulationMeasure | None = None
    selected_source_id: str | None = None
    candidate_population_measures: tuple[EpidemiologyPopulationCandidate, ...] = ()
    assumption_flags: tuple[str, ...] = ()
    human_review_questions: tuple[str, ...] = ()
    confidence_score: int = Field(default=0, ge=0, le=10)


MARKET_SIZING_INSTRUCTIONS = """You are a market-sizing interpretation subagent for PharmaOS Agent 4 due diligence.
Use only the supplied commercial input bundle. Do not invent source facts.
Do not calculate revenue, treated patients, launch ramp, peak sales, rNPV, discounted revenue, probability-adjusted revenue, or LOE-adjusted revenue.
Your job is only to select or infer market-sizing assumptions for a deterministic Python calculator.
If no usable population measure exists, set calculable false, set the selected population measure value to null, and explain the missing evidence.
You may infer missing diagnosed, treated, eligibility, or commercially addressable fractions only when the source_type is model_inferred, default_assumption, or fallback.
Return structured output matching the schema only."""


EPIDEMIOLOGY_EXTRACTION_INSTRUCTIONS = """You are an epidemiology evidence extraction subagent for PharmaOS Agent 4 due diligence.
Use only the supplied PubMed titles and abstract snippets. Do not use outside knowledge.
Extract only explicit numeric epidemiology estimates relevant to the target indication and geography.
Prefer United States-specific patient-count estimates for the same disease or indicated severity segment.
If a source gives a prevalence percentage/rate, you may convert it to a patient-count denominator only when the input payload provides a US population denominator. Choose total, adult, or pediatric denominator to match the article population when possible, state the formula in your rationale, cite both the PubMed source and the population source, and mark human_review_required true.
Do not use clinical trial enrollment counts as market denominators.
Return a selected_population_measure only when a usable patient-count denominator or source-backed converted denominator exists.
Every selected or candidate value must include the PMID/source_id and a short rationale tied to the evidence."""


def build_commercial_model(
    *,
    annual_patients: float | None,
    peak_penetration: float | None,
    gross_to_net: float | None,
    pricing: PricingOutput,
    trial: ClinicalTrialRecord | None = None,
    asset: AssetIdentityOutput | None = None,
    clinical_evidence: ClinicalEvidenceSummary | None = None,
    run_id: str | None = None,
    config: AgentRuntimeConfig | None = None,
) -> CommercialModelOutput:
    """Build a commercial model while preserving the legacy output contract."""

    return build_commercial_model_with_trace(
        annual_patients=annual_patients,
        peak_penetration=peak_penetration,
        gross_to_net=gross_to_net,
        pricing=pricing,
        trial=trial,
        asset=asset,
        clinical_evidence=clinical_evidence,
        run_id=run_id,
        config=config,
    ).output


def build_commercial_model_with_trace(
    *,
    annual_patients: float | None,
    peak_penetration: float | None,
    gross_to_net: float | None,
    pricing: PricingOutput,
    trial: ClinicalTrialRecord | None = None,
    asset: AssetIdentityOutput | None = None,
    clinical_evidence: ClinicalEvidenceSummary | None = None,
    run_id: str | None = None,
    config: AgentRuntimeConfig | None = None,
) -> CommercialModelRunResult:
    """Build a Panoptic-style commercial model and retain market-sizing trace metadata."""

    default_config = load_config("default_archetypes.yaml", section="due_diligence")
    config_id = config_source_id("default_archetypes.yaml", section="due_diligence")
    market_evidence = (
        _collect_market_population_evidence(
            trial=trial,
            asset=asset,
            clinical_evidence=clinical_evidence,
            default_config=default_config,
            run_id=run_id,
            config=config,
        )
        if annual_patients is None and trial is not None
        else _empty_market_evidence()
    )
    bundle = assemble_commercial_input_bundle(
        trial=trial,
        asset=asset,
        clinical_evidence=clinical_evidence,
        pricing=pricing,
        annual_patients=annual_patients,
        peak_penetration=peak_penetration,
        gross_to_net=gross_to_net,
        default_archetypes=default_config,
        market_evidence=market_evidence,
    )
    if annual_patients is not None:
        interpretation = _interpretation_from_reviewed_population(
            annual_patients=annual_patients,
            default_config=default_config,
        )
        agent_trace = None
        agent_traces: tuple[object, ...] = ()
        trace_metadata: dict[str, str | int | float | bool | None] = {}
    else:
        fallback = _fallback_interpretation(default_config, reason="No source-backed population measure was available.")
        result = run_structured_llm_call(
            agent_name="CommercialMarketSizingAgent",
            instructions=MARKET_SIZING_INSTRUCTIONS,
            payload=bundle,
            output_type=MarketSizingInterpretation,
            run_id=run_id or f"commercial-market-sizing-{datetime.now(timezone.utc).isoformat()}",
            input_summary="Select market-sizing assumptions for Agent 4 commercial forecast.",
            config=runtime_config_for_route(
                model_route="agent4_subagent",
                disabled_provenance="pharma_os.tools.commercial_model.market_sizing",
                config=config,
            ),
            offline_output=fallback,
            source_ids=tuple(pricing.source_ids),
            confidence=None,
            rationale_summary="Market-sizing interpretation feeds deterministic commercial calculations.",
        )
        interpretation = result.output
        agent_trace = result.trace
        agent_traces = tuple(item for item in (market_evidence.get("trace"), result.trace) if item is not None)
        trace_metadata = _combined_trace_metadata(
            {
                "epidemiology_extraction": market_evidence.get("trace_metadata"),
                "market_sizing": result.trace_metadata,
            }
        )

    output = calculate_commercial_model(
        bundle=bundle,
        interpretation=interpretation,
        default_config=default_config,
        pricing=pricing,
        config_id=config_id,
    )
    source_ids = tuple(dict.fromkeys((*output.source_ids, *_market_source_ids(market_evidence))))
    if source_ids != output.source_ids:
        output = output.model_copy(update={"source_ids": source_ids})
    return CommercialModelRunResult(
        output=output,
        agent_trace=agent_trace,
        agent_traces=agent_traces,
        sources=tuple(market_evidence.get("sources") or ()),
        trace_metadata=trace_metadata,
    )


def assemble_commercial_input_bundle(
    *,
    trial: ClinicalTrialRecord | None,
    asset: AssetIdentityOutput | None,
    clinical_evidence: ClinicalEvidenceSummary | None,
    pricing: PricingOutput,
    annual_patients: float | None,
    peak_penetration: float | None,
    gross_to_net: float | None,
    default_archetypes: dict[str, Any],
    market_evidence: dict[str, Any] | None = None,
) -> CommercialInputBundle:
    """Assemble a compact commercial evidence bundle from existing Agent 4 context."""

    market_evidence = market_evidence or _empty_market_evidence()
    user_overrides = {
        key: value
        for key, value in {
            "annual_patients": annual_patients,
            "peak_penetration": peak_penetration,
            "gross_to_net": gross_to_net,
        }.items()
        if value is not None
    }
    population_evidence: list[dict[str, Any]] = []
    if annual_patients is not None:
        population_evidence.append(
            {
                "value": annual_patients,
                "unit": "patients",
                "measure_type": "reviewed_annual_eligible_patients",
                "condition": asset.normalized_indication if asset else None,
                "geography": "United States",
                "source_type": "user_override",
                "evidence_reference": "DueDiligenceInput.annual_patients",
                "rationale": "User-reviewed annual eligible patient assumption supplied to PharmaOS.",
            }
        )
    for item in market_evidence.get("population_evidence") or ():
        if isinstance(item, dict):
            population_evidence.append(item)
    pubmed_titles = tuple(clinical_evidence.pubmed_titles if clinical_evidence else ())
    segmentation = [
        {
            "value": None,
            "unit": None,
            "measure_type": "literature_metadata",
            "condition": asset.normalized_indication if asset else None,
            "geography": "United States",
            "source_type": "source_derived",
            "evidence_reference": "clinical_evidence.pubmed_titles",
            "rationale": title,
        }
        for title in pubmed_titles
    ]
    segmentation.extend(item for item in market_evidence.get("segmentation_evidence") or () if isinstance(item, dict))
    missing_inputs = list(pricing.missing_data_flags)
    market_flags = [str(item) for item in market_evidence.get("confidence_flags") or () if item]
    missing_inputs.extend(market_flags)
    has_numeric_population = any(_float_or_none(item.get("value")) is not None for item in population_evidence if isinstance(item, dict))
    if annual_patients is None and not has_numeric_population:
        missing_inputs.append(missing("commercial-population-measure-missing", "commercial_model", "selected_population_measure", "No source-backed or reviewed population measure is available.", "high"))
    missing_labels = tuple(
        dict.fromkeys(
            flag.flag_id if isinstance(flag, MissingDataFlag) else str(flag)
            for flag in missing_inputs
        )
    )
    return CommercialInputBundle(
        asset_summary={
            "asset_name": asset.asset_name if asset else None,
            "sponsor": asset.sponsor if asset else None,
            "nct_id": trial.nct_id if trial else asset.nct_id if asset else None,
            "indication": asset.normalized_indication if asset else None,
            "condition_terms": list(trial.conditions) if trial else [],
            "phase": (trial.phases[0] if trial and trial.phases else None),
            "status": trial.overall_status if trial else None,
            "modality": asset.modality if asset else None,
        },
        us_population_denominator=dict(market_evidence.get("us_population_denominator") or {}),
        disease_population_evidence=tuple(population_evidence),
        prevalence_evidence=tuple(item for item in population_evidence if "preval" in str(item.get("measure_type", "")).casefold() or item.get("measure_type") == "reviewed_annual_eligible_patients"),
        incidence_evidence=tuple(item for item in population_evidence if "incidence" in str(item.get("measure_type", "")).casefold()),
        segmentation_evidence=tuple(segmentation),
        trial_eligibility_criteria={
            "eligibility_criteria": trial.eligibility_criteria if trial else None,
            "enrollment_count": trial.enrollment_count if trial else None,
            "primary_endpoints": list(trial.primary_endpoints) if trial else [],
        },
        pricing_benchmark=_pricing_benchmark(pricing, default_archetypes),
        market_query_diagnostics=tuple(item for item in market_evidence.get("query_diagnostics") or () if isinstance(item, dict)),
        missing_inputs=missing_labels,
        user_overrides={**user_overrides, **_market_user_overrides(market_evidence)},
        predefined_archetype_assumptions=default_archetypes,
    )


def _collect_market_population_evidence(
    *,
    trial: ClinicalTrialRecord | None,
    asset: AssetIdentityOutput | None,
    clinical_evidence: ClinicalEvidenceSummary | None,
    default_config: dict[str, Any],
    run_id: str | None,
    config: AgentRuntimeConfig | None,
) -> dict[str, Any]:
    us_population, population_source, population_flags, population_questions = _us_population_denominator(default_config)
    if trial is None:
        return _empty_market_evidence(
            sources=(population_source,) if population_source else (),
            confidence_flags=tuple((*population_flags, "market_trial_missing")),
            human_review_questions=population_questions,
            us_population_denominator=us_population,
        )
    query_config = load_config("market_query_templates.yaml", section="due_diligence")
    queries = _generate_market_queries(trial=trial, asset=asset, query_config=query_config)
    if not queries:
        return _empty_market_evidence(
            sources=(population_source,) if population_source else (),
            confidence_flags=tuple((*population_flags, "market_queries_missing")),
            human_review_questions=population_questions,
            us_population_denominator=us_population,
        )
    articles: list[PubMedArticle] = []
    flags: list[str] = []
    query_diagnostics: list[dict[str, Any]] = []
    client = PubMedClient()
    for query in queries[: _env_int("PHARMA_OS_MARKET_MAX_QUERIES", 8, minimum=1, maximum=40)]:
        try:
            results = client.search(
                query,
                max_results=_env_int("PHARMA_OS_MARKET_PUBMED_RESULTS_PER_QUERY", 5, minimum=1, maximum=25),
            )
            query_diagnostics.append({"query": query, "status": "ok", "article_count": len(results)})
            articles.extend(results)
        except PubMedError as exc:
            flags.append(f"pubmed_market_query_failed:{_slug(query)}:{exc.__class__.__name__}")
            query_diagnostics.append({"query": query, "status": "failed", "error_type": exc.__class__.__name__, "error": str(exc)})
    articles = _dedupe_articles(articles)
    sources = _dedupe_sources((*((population_source,) if population_source else ()), *(article.source() for article in articles)))
    if not articles:
        return _empty_market_evidence(
            sources=sources,
            confidence_flags=tuple((*population_flags, *flags, "market_pubmed_no_articles")),
            human_review_questions=population_questions,
            us_population_denominator=us_population,
            query_diagnostics=tuple(query_diagnostics),
        )
    selected = _select_epi_articles(trial=trial, asset=asset, articles=articles)
    payload = {
        "target_indication": _target_indication(trial, asset),
        "condition_terms": list(trial.conditions),
        "geography": "United States",
        "nct_id": trial.nct_id,
        "clinical_evidence_query": clinical_evidence.pubmed_query if clinical_evidence else None,
        "us_population_denominator": us_population,
        "articles": [
            {
                "pmid": article.pmid,
                "source_id": article.source_id,
                "title": article.title,
                "journal": article.journal,
                "year": article.year,
                "abstract_snippet": article.abstract_snippet,
            }
            for article in selected
        ],
    }
    fallback = EpidemiologyEvidenceExtraction(
        evidence_status="ai_unavailable",
        selected_population_measure=None,
        candidate_population_measures=(),
        assumption_flags=tuple((*flags, "ai_epi_extraction_unavailable")),
        human_review_questions=("Confirm the source-backed disease population or provide reviewed annual eligible patients.",),
        confidence_score=0,
    )
    result = run_structured_llm_call(
        agent_name="CommercialEpidemiologyEvidenceAgent",
        instructions=EPIDEMIOLOGY_EXTRACTION_INSTRUCTIONS,
        payload=payload,
        output_type=EpidemiologyEvidenceExtraction,
        run_id=run_id or f"commercial-epi-{datetime.now(timezone.utc).isoformat()}",
        input_summary="Extract source-backed epidemiology market denominators for Agent 4 commercial sizing.",
        config=runtime_config_for_route(
            model_route="agent4_subagent",
            disabled_provenance="pharma_os.tools.commercial_model.epidemiology_extraction",
            config=config,
        ),
        offline_output=fallback,
        source_ids=tuple(source.source_id for source in sources),
        confidence=None,
        rationale_summary="Epidemiology evidence extraction supplies market-sizing denominator candidates.",
    )
    population_evidence = _population_evidence_from_extraction(result.output)
    segmentation = tuple(
        candidate.model_dump(mode="json")
        for candidate in result.output.candidate_population_measures
    )
    confidence_flags = tuple(
        dict.fromkeys(
            (
                *population_flags,
                *flags,
                *result.output.assumption_flags,
                *("literature_population_measure_missing" for _ in [1] if not population_evidence),
            )
        )
    )
    return {
        "population_evidence": population_evidence,
        "segmentation_evidence": segmentation,
        "sources": sources,
        "source_ids": tuple(source.source_id for source in sources),
        "confidence_flags": confidence_flags,
        "human_review_questions": tuple(dict.fromkeys((*population_questions, *result.output.human_review_questions))),
        "us_population_denominator": us_population,
        "query_diagnostics": tuple(query_diagnostics),
        "trace": result.trace,
        "trace_metadata": result.trace_metadata,
    }


def _empty_market_evidence(
    *,
    sources: tuple[SourceMetadata, ...] = (),
    confidence_flags: tuple[str, ...] = (),
    human_review_questions: tuple[str, ...] = (),
    us_population_denominator: dict[str, Any] | None = None,
    query_diagnostics: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    return {
        "population_evidence": (),
        "segmentation_evidence": (),
        "sources": sources,
        "source_ids": tuple(source.source_id for source in sources),
        "confidence_flags": confidence_flags,
        "human_review_questions": human_review_questions,
        "us_population_denominator": us_population_denominator or {},
        "query_diagnostics": query_diagnostics,
        "trace": None,
        "trace_metadata": {},
    }


def _generate_market_queries(*, trial: ClinicalTrialRecord, asset: AssetIdentityOutput | None, query_config: dict[str, Any]) -> list[str]:
    templates = query_config.get("templates") if isinstance(query_config, dict) else None
    if not isinstance(templates, dict):
        return []
    geography = "United States"
    conditions = tuple(dict.fromkeys(term for term in trial.conditions if term))
    target = _target_indication(trial, asset)
    title_phrases, title_acronyms = _title_indication_phrases(trial)
    candidates: list[str] = []
    for condition in conditions[:4]:
        for key in ("disease_prevalence", "disease_epidemiology", "disease_population"):
            template = templates.get(key)
            if isinstance(template, str):
                candidates.append(template.format(condition=condition, geography=geography))
    for phrase in tuple(dict.fromkeys(term for term in (target, *title_phrases, " and ".join(conditions) if len(conditions) > 1 else None) if term))[:4]:
        for key in ("combined_indication_prevalence", "combined_indication_epidemiology", "combined_indication_population"):
            template = templates.get(key)
            if isinstance(template, str):
                candidates.append(template.format(indication_phrase=phrase, geography=geography))
    for acronym in title_acronyms[:3]:
        for key in ("indication_acronym_prevalence", "indication_acronym_epidemiology"):
            template = templates.get(key)
            if isinstance(template, str):
                candidates.append(template.format(acronym=acronym, geography=geography))
    template = templates.get("trial_indication_prevalence")
    if isinstance(template, str) and target:
        candidates.append(template.format(normalized_indication=target, geography=geography))
    title = trial.brief_title or trial.official_title
    template = templates.get("trial_eligibility_population")
    if isinstance(template, str) and title:
        candidates.append(template.format(trial_title=title))
    return list(dict.fromkeys(candidates))


def _title_indication_phrases(trial: ClinicalTrialRecord) -> tuple[list[str], list[str]]:
    phrases: list[str] = []
    acronyms: list[str] = []
    for title in (trial.brief_title, trial.official_title):
        if not title:
            continue
        match = re.search(r"patients with (?P<phrase>.+?)(?:\s+\([^)]*\)|$)", title, re.I)
        if match:
            phrases.append(match.group("phrase").strip(" .,:;"))
        acronyms.extend(
            token
            for token in re.findall(r"\(([A-Za-z][A-Za-z0-9-]{2,20})\)", title)
            if any(char.isupper() for char in token) and "study" not in token.casefold()
        )
    return list(dict.fromkeys(phrases)), list(dict.fromkeys(acronyms))


def _target_indication(trial: ClinicalTrialRecord, asset: AssetIdentityOutput | None) -> str | None:
    return (asset.normalized_indication if asset else None) or (trial.conditions[0] if trial.conditions else None)


def _dedupe_articles(articles: list[PubMedArticle]) -> list[PubMedArticle]:
    deduped: dict[str, PubMedArticle] = {}
    for article in articles:
        deduped[article.pmid] = article
    return list(deduped.values())


def _select_epi_articles(*, trial: ClinicalTrialRecord, asset: AssetIdentityOutput | None, articles: list[PubMedArticle]) -> list[PubMedArticle]:
    target = (_target_indication(trial, asset) or "").casefold()
    conditions = [condition.casefold() for condition in trial.conditions if condition]
    scored: list[tuple[int, int, PubMedArticle]] = []
    for index, article in enumerate(articles):
        text = f"{article.title} {article.abstract_snippet or ''}".casefold()
        score = 0
        if target and target in text:
            score += 75
        score += 20 * sum(1 for condition in conditions if condition and condition in text)
        for pattern, weight in (
            (r"\bprevalence\b", 10),
            (r"\bepidemiolog(?:y|ic|ical)\b", 8),
            (r"\bpopulation\b", 6),
            (r"\bUnited States\b|\bU\.?S\.?\b", 8),
            (r"\bmillion\b|\bpatients\b|\badults\b", 5),
            (r"\brandomi[sz]ed\b|\bclinical trial\b", -8),
        ):
            if re.search(pattern, text, re.I):
                score += weight
        scored.append((score, -index, article))
    scored.sort(reverse=True)
    limit = _env_int("PHARMA_OS_MARKET_EPI_MAX_ABSTRACTS", 8, minimum=1, maximum=25)
    return [article for _, _, article in scored[:limit]]


def _population_evidence_from_extraction(extraction: EpidemiologyEvidenceExtraction) -> tuple[dict[str, Any], ...]:
    items: list[dict[str, Any]] = []
    selected = extraction.selected_population_measure
    if selected and selected.value is not None and _is_patient_count_population(selected.model_dump(mode="json")):
        payload = selected.model_dump(mode="json")
        if extraction.selected_source_id:
            payload["source_id"] = extraction.selected_source_id
        items.append(payload)
    for candidate in extraction.candidate_population_measures:
        if candidate.value is None:
            continue
        payload = candidate.model_dump(mode="json")
        if not _is_patient_count_population(payload):
            continue
        if not any(item.get("value") == payload.get("value") and item.get("source_id") == payload.get("source_id") for item in items):
            items.append(payload)
    return tuple(items)


def _is_patient_count_population(payload: dict[str, Any]) -> bool:
    measure_type = str(payload.get("measure_type") or "").casefold()
    unit = str(payload.get("unit") or "").casefold()
    if any(token in measure_type for token in ("percent", "rate", "fraction")):
        return False
    if any(token in unit for token in ("%", "percent", "per ", "rate", "fraction")):
        return False
    return "patient" in measure_type or "population" in measure_type or "patient" in unit or unit in {"people", "persons"}


def _market_source_ids(market_evidence: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(source_id) for source_id in market_evidence.get("source_ids") or () if source_id)


def _market_user_overrides(market_evidence: dict[str, Any]) -> dict[str, Any]:
    del market_evidence
    return {}


def _source_ids_from_reference(reference: str | None) -> tuple[str, ...]:
    if not reference:
        return ()
    return tuple(
        dict.fromkeys(
            match.group(0).rstrip(".,;)")
            for match in re.finditer(r"\b(?:pubmed|ctgov|census|config|wac|openfda_label):[A-Za-z0-9_.:/-]+", reference)
        )
    )


def _us_population_denominator(default_config: dict[str, Any]) -> tuple[dict[str, Any], SourceMetadata | None, tuple[str, ...], tuple[str, ...]]:
    try:
        denominator = CensusPopulationClient().get_latest_us_population()
        return denominator.model_payload(), denominator.source(), (), ()
    except CensusPopulationError as exc:
        fallback = default_config.get("us_population_denominator") or {}
        if not isinstance(fallback, dict) or to_float(fallback.get("total_us_population")) is None:
            return (
                {},
                None,
                ("us_population_denominator_unavailable",),
                (f"Provide a reviewed US population denominator; live Census lookup failed: {exc}",),
            )
        payload = dict(fallback)
        payload["lookup_error"] = str(exc)
        payload["source_type"] = payload.get("source_type") or "config_fallback"
        payload["human_review_required"] = True
        source = SourceMetadata(
            source_id=str(payload.get("source_id") or "config:due_diligence:us_population_denominator"),
            title=f"US population denominator fallback ({payload.get('source_year') or 'unknown year'})",
            url=payload.get("source_url"),
            provenance="due_diligence/default_archetypes.yaml:us_population_denominator",
            source_type=str(payload.get("source_type") or "config_fallback"),
            version=str(payload.get("source_year")) if payload.get("source_year") is not None else None,
        )
        return (
            payload,
            source,
            ("us_population_denominator_config_fallback",),
            ("Confirm the US population denominator used for prevalence-to-patient conversion.",),
        )


def _dedupe_sources(sources: tuple[SourceMetadata, ...]) -> tuple[SourceMetadata, ...]:
    deduped: dict[str, SourceMetadata] = {}
    for source in sources:
        deduped[source.source_id] = source
    return tuple(deduped.values())


def _combined_trace_metadata(items: dict[str, Any]) -> dict[str, str | int | float | bool | None]:
    flattened: dict[str, str | int | float | bool | None] = {}
    for prefix, value in items.items():
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (str, int, float, bool)) or item is None:
                    flattened[f"{prefix}_{key}"] = item
    return flattened


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def calculate_commercial_model(
    *,
    bundle: CommercialInputBundle,
    interpretation: MarketSizingInterpretation,
    default_config: dict[str, Any],
    pricing: PricingOutput,
    config_id: str,
) -> CommercialModelOutput:
    """Calculate deterministic revenue cases from selected market-sizing assumptions."""

    archetype_name = interpretation.selected_market_archetype or "chronic_specialty_prevalence"
    annual_wac = _annual_wac_triplet(bundle.pricing_benchmark, default_config)
    gross_to_net, gross_source = _triplet_from_archetype(default_config, archetype_name, "gross_to_net", bundle.user_overrides)
    peak_penetration, peak_source = _triplet_from_archetype(default_config, archetype_name, "peak_penetration", bundle.user_overrides)
    launch_ramp, ramp_source = _launch_ramp_from_archetype(default_config, archetype_name, bundle.user_overrides)
    ledger = _ledger_records(
        interpretation=interpretation,
        annual_wac=annual_wac,
        gross_to_net=gross_to_net,
        gross_source=gross_source,
        peak_penetration=peak_penetration,
        peak_source=peak_source,
        launch_ramp=launch_ramp,
        ramp_source=ramp_source,
    )
    missing_values = _missing_required_values(interpretation, annual_wac, gross_to_net, peak_penetration, launch_ramp)
    flags = _commercial_missing_flags(bundle, interpretation, missing_values)
    calculable = interpretation.calculable and not missing_values
    cases = (
        _build_cases(
            interpretation=interpretation,
            annual_wac=annual_wac,
            gross_to_net=gross_to_net,
            peak_penetration=peak_penetration,
            launch_ramp=launch_ramp,
            forecast_years=int(default_config.get("forecast_years", len(launch_ramp) or 1)),
        )
        if calculable
        else {}
    )
    base = cases.get("base")
    legacy_assumptions = _legacy_assumptions(
        interpretation=interpretation,
        peak_penetration=peak_penetration,
        gross_to_net=gross_to_net,
        launch_ramp=launch_ramp,
        config_id=config_id,
        archetype_name=archetype_name,
        user_overrides=bundle.user_overrides,
    )
    source_ids = tuple(
        dict.fromkeys(
            (
                *pricing.source_ids,
                config_id,
            )
        )
    )
    confidence_flags = tuple(
        dict.fromkeys(
            (
                *bundle.missing_inputs,
                *interpretation.assumption_flags,
                *interpretation.human_review_flags,
                *(f"missing_required_commercial_input:{item}" for item in missing_values),
            )
        )
    )
    return CommercialModelOutput(
        calculable=calculable,
        annual_patients=base.patient_funnel.commercially_addressable_patients if base else _float_or_none(bundle.user_overrides.get("annual_patients")),
        peak_penetration=base.penetration.peak_penetration if base else peak_penetration.base,
        gross_to_net=base.pricing.gross_to_net if base else gross_to_net.base,
        net_price=base.pricing.net_price if base else None,
        peak_net_sales=base.peak_net_sales if base else None,
        revenue_forecast=base.revenue_forecast if base else (),
        selected_market_archetype=archetype_name,
        market_basis=interpretation.market_basis,
        selected_population_measure=interpretation.selected_population_measure,
        patient_funnel=base.patient_funnel if base else None,
        cases=cases,
        assumption_ledger=tuple(ledger),
        commercial_input_bundle_summary=_bundle_summary(bundle),
        confidence_flags=confidence_flags,
        human_review_questions=tuple(_human_review_questions(interpretation, missing_values)),
        assumptions=tuple(legacy_assumptions),
        source_ids=source_ids,
        missing_data_flags=tuple(flags),
        confidence=_confidence(calculable, interpretation.confidence_score, flags),
    )


def _interpretation_from_reviewed_population(*, annual_patients: float, default_config: dict[str, Any]) -> MarketSizingInterpretation:
    archetype_name = "chronic_specialty_prevalence"
    return MarketSizingInterpretation(
        calculable=True,
        selected_market_archetype=archetype_name,
        market_basis="prevalence_stock",
        selected_population_measure=SelectedPopulationMeasure(
            value=annual_patients,
            unit="patients",
            measure_type="reviewed_annual_eligible_patients",
            condition=None,
            geography="United States",
            source_type="user_override",
            rationale="User-reviewed annual eligible patient assumption supplied to PharmaOS.",
            evidence_reference="DueDiligenceInput.annual_patients",
            confidence_score=8,
            human_review_required=False,
        ),
        yearly_eligible_patient_logic="Use reviewed annual eligible patients directly; funnel fractions are set to 1.0 to preserve the reviewed input.",
        diagnosed_fraction=_unit_fraction("diagnosed_fraction", source_type="user_override"),
        treated_fraction=_unit_fraction("treated_fraction", source_type="user_override"),
        eligibility_fraction=_unit_fraction("eligibility_fraction", source_type="user_override"),
        commercially_addressable_fraction=_unit_fraction("commercially_addressable_fraction", source_type="user_override"),
        rationale="Reviewed annual eligible patients are authoritative for this run.",
        confidence_score=8,
        key_evidence_used=("DueDiligenceInput.annual_patients",),
        assumption_flags=(),
        human_review_flags=(),
    )


def _fallback_interpretation(default_config: dict[str, Any], *, reason: str) -> MarketSizingInterpretation:
    archetype_name = "chronic_specialty_prevalence"
    archetype = (default_config.get("archetypes") or {}).get(archetype_name, {})
    return MarketSizingInterpretation(
        calculable=False,
        selected_market_archetype=archetype_name,
        market_basis=archetype.get("market_basis") or "prevalence_stock",
        selected_population_measure=SelectedPopulationMeasure(
            value=None,
            unit="patients",
            measure_type=None,
            condition=None,
            geography="United States",
            source_type="missing",
            rationale=reason,
            evidence_reference=None,
            confidence_score=0,
            human_review_required=True,
        ),
        yearly_eligible_patient_logic="Commercial model cannot calculate until a source-backed or reviewed population measure is available.",
        diagnosed_fraction=_fraction_from_archetype(archetype, "diagnosed_fraction"),
        treated_fraction=_fraction_from_archetype(archetype, "treated_fraction"),
        eligibility_fraction=_fraction_from_archetype(archetype, "eligibility_fraction"),
        commercially_addressable_fraction=_fraction_from_archetype(archetype, "commercially_addressable_fraction"),
        rationale=reason,
        confidence_score=0,
        key_evidence_used=(),
        assumption_flags=("selected_population_measure_missing",),
        human_review_flags=("Confirm the source-backed disease population or provide reviewed annual eligible patients.",),
    )


def _unit_fraction(name: str, *, source_type: str) -> CommercialAssumptionTriplet:
    return CommercialAssumptionTriplet(
        low=1.0,
        base=1.0,
        high=1.0,
        source_type=source_type,  # type: ignore[arg-type]
        rationale=f"{name} set to 1.0 because annual_patients already represents the reviewed eligible/addressable population.",
        evidence_reference="DueDiligenceInput.annual_patients",
        confidence_score=8,
        human_review_required=False,
    )


def _fraction_from_archetype(archetype: dict[str, Any], key: str) -> CommercialAssumptionTriplet:
    value = archetype.get(key) or {}
    return CommercialAssumptionTriplet(
        low=to_float(value.get("low")),
        base=to_float(value.get("base")),
        high=to_float(value.get("high")),
        source_type="default_assumption",
        rationale=f"{key} fallback from default_archetypes.yaml.",
        evidence_reference=f"default_archetypes.yaml:{key}",
        confidence_score=5,
        human_review_required=True,
    )


def _pricing_benchmark(pricing: PricingOutput, default_config: dict[str, Any]) -> dict[str, Any]:
    base = pricing.annual_wac
    multipliers = default_config.get("price_sensitivity_multipliers") or {}
    low_multiplier = float(multipliers.get("low", 0.8))
    high_multiplier = float(multipliers.get("high", 1.2))
    return {
        "annual_gross_wac": {
            "low": base * low_multiplier if base is not None else None,
            "base": base,
            "high": base * high_multiplier if base is not None else None,
        },
        "source": "pricing_output.annual_wac" if base is not None else None,
        "primary_analog": pricing.matched_product,
        "annualization_details": pricing.annualization_details,
    }


def _annual_wac_triplet(pricing_benchmark: dict[str, Any], default_config: dict[str, Any]) -> ValueTriplet:
    del default_config
    value = pricing_benchmark.get("annual_gross_wac") or {}
    return ValueTriplet(low=_float_or_none(value.get("low")), base=_float_or_none(value.get("base")), high=_float_or_none(value.get("high")))


def _triplet_from_archetype(
    default_config: dict[str, Any],
    archetype_name: str | None,
    key: str,
    user_overrides: dict[str, Any],
) -> tuple[ValueTriplet, str]:
    override = user_overrides.get(key)
    if override is not None:
        value = _triplet(override) if isinstance(override, dict) else ValueTriplet(low=float(override), base=float(override), high=float(override))
        return value, "user_override"
    archetype = _archetype(default_config, archetype_name)
    return _triplet(archetype.get(key) or {}), "default_assumption"


def _launch_ramp_from_archetype(default_config: dict[str, Any], archetype_name: str | None, user_overrides: dict[str, Any]) -> tuple[tuple[float, ...], str]:
    override = user_overrides.get("launch_ramp")
    if override is not None:
        return tuple(float(item) for item in override), "user_override"
    value = _archetype(default_config, archetype_name).get("launch_ramp") or ()
    return tuple(float(item) for item in value if to_float(item) is not None), "default_assumption"


def _archetype(default_config: dict[str, Any], archetype_name: str | None) -> dict[str, Any]:
    archetypes = default_config.get("archetypes") or {}
    if archetype_name and archetype_name in archetypes:
        return archetypes[archetype_name]
    if "chronic_specialty_prevalence" in archetypes:
        return archetypes["chronic_specialty_prevalence"]
    return next(iter(archetypes.values()), {})


def _triplet(value: dict[str, Any]) -> ValueTriplet:
    return ValueTriplet(low=_float_or_none(value.get("low")), base=_float_or_none(value.get("base")), high=_float_or_none(value.get("high")))


def _build_cases(
    *,
    interpretation: MarketSizingInterpretation,
    annual_wac: ValueTriplet,
    gross_to_net: ValueTriplet,
    peak_penetration: ValueTriplet,
    launch_ramp: tuple[float, ...],
    forecast_years: int,
) -> dict[str, CommercialCaseOutput]:
    cases: dict[str, CommercialCaseOutput] = {}
    for case in ("downside", "base", "upside"):
        starting_population = _case_value(ValueTriplet(low=interpretation.selected_population_measure.value, base=interpretation.selected_population_measure.value, high=interpretation.selected_population_measure.value), case)
        diagnosed_fraction = _case_value(_triplet_from_ai(interpretation.diagnosed_fraction), case)
        treated_fraction = _case_value(_triplet_from_ai(interpretation.treated_fraction), case)
        eligibility_fraction = _case_value(_triplet_from_ai(interpretation.eligibility_fraction), case)
        addressable_fraction = _case_value(_triplet_from_ai(interpretation.commercially_addressable_fraction), case)
        annual_price = _case_value(annual_wac, case)
        gtn = _case_value(gross_to_net, case, inverse=True)
        peak = _case_value(peak_penetration, case)
        diagnosed = starting_population * diagnosed_fraction
        treated_or_managed = diagnosed * treated_fraction
        eligible = treated_or_managed * eligibility_fraction
        commercially_addressable = eligible * addressable_fraction
        net_price = annual_price * (1 - gtn)
        rows: list[RevenueForecastYear] = []
        for year_index in range(forecast_years):
            ramp = launch_ramp[year_index] if year_index < len(launch_ramp) else launch_ramp[-1]
            treated = commercially_addressable * peak * ramp
            rows.append(
                RevenueForecastYear(
                    year=year_index + 1,
                    treated_patients=round(treated, 2),
                    net_price=round(net_price, 2),
                    net_revenue=round(treated * net_price, 2),
                )
            )
        peak_gross = commercially_addressable * peak * annual_price
        peak_net = peak_gross * (1 - gtn)
        cases[case] = CommercialCaseOutput(
            patient_funnel=PatientFunnel(
                starting_population=round(starting_population, 4),
                diagnosed_patients=round(diagnosed, 4),
                treated_or_managed_patients=round(treated_or_managed, 4),
                eligible_patients=round(eligible, 4),
                commercially_addressable_patients=round(commercially_addressable, 4),
                diagnosed_fraction=round(diagnosed_fraction, 4),
                treated_fraction=round(treated_fraction, 4),
                eligibility_fraction=round(eligibility_fraction, 4),
                commercially_addressable_fraction=round(addressable_fraction, 4),
            ),
            pricing=CommercialPricingCase(annual_gross_wac=round(annual_price, 2), gross_to_net=round(gtn, 4), net_price=round(net_price, 2)),
            penetration=CommercialPenetrationCase(peak_penetration=round(peak, 4), launch_ramp=tuple(round(item, 4) for item in launch_ramp)),
            revenue_forecast=tuple(rows),
            peak_gross_sales=round(peak_gross, 2),
            peak_net_sales=round(peak_net, 2),
        )
    return cases


def _case_value(value: ValueTriplet, case: str, *, inverse: bool = False) -> float:
    selected = {"downside": value.high if inverse else value.low, "base": value.base, "upside": value.low if inverse else value.high}[case]
    if selected is None:
        raise ValueError(f"Missing {case} value")
    return float(selected)


def _triplet_from_ai(value: CommercialAssumptionTriplet) -> ValueTriplet:
    return ValueTriplet(low=value.low, base=value.base, high=value.high)


def _missing_required_values(
    interpretation: MarketSizingInterpretation,
    annual_wac: ValueTriplet,
    gross_to_net: ValueTriplet,
    peak_penetration: ValueTriplet,
    launch_ramp: tuple[float, ...],
) -> tuple[str, ...]:
    missing_values: list[str] = []
    if interpretation.selected_population_measure.value is None:
        missing_values.append("selected_population_measure.value")
    for name in ("diagnosed_fraction", "treated_fraction", "eligibility_fraction", "commercially_addressable_fraction"):
        value = getattr(interpretation, name)
        if value.low is None or value.base is None or value.high is None:
            missing_values.append(name)
    for name, triplet in (("annual_gross_wac", annual_wac), ("gross_to_net", gross_to_net), ("peak_penetration", peak_penetration)):
        if triplet.low is None or triplet.base is None or triplet.high is None:
            missing_values.append(name)
    if not launch_ramp:
        missing_values.append("launch_ramp")
    return tuple(missing_values)


def _commercial_missing_flags(bundle: CommercialInputBundle, interpretation: MarketSizingInterpretation, missing_values: tuple[str, ...]) -> list[MissingDataFlag]:
    flags: list[MissingDataFlag] = []
    for item in bundle.missing_inputs:
        flags.append(missing(f"commercial-{_slug(item)}", "commercial_model", item, f"Commercial sizing input gap: {item}.", "high" if "missing" in item else "medium"))
    for item in interpretation.assumption_flags:
        flags.append(missing(f"commercial-assumption-{_slug(item)}", "commercial_model", item, f"Market sizing assumption flag: {item}.", "medium"))
    for item in missing_values:
        flags.append(missing(f"commercial-required-{_slug(item)}", "commercial_model", item, f"Required commercial model input is unresolved: {item}.", "high"))
    return _dedupe_flags(flags)


def _legacy_assumptions(
    *,
    interpretation: MarketSizingInterpretation,
    peak_penetration: ValueTriplet,
    gross_to_net: ValueTriplet,
    launch_ramp: tuple[float, ...],
    config_id: str,
    archetype_name: str,
    user_overrides: dict[str, Any],
) -> list[Any]:
    selected_annual_patients = interpretation.selected_population_measure.value
    population_reference = interpretation.selected_population_measure.evidence_reference
    parsed_population_source_ids = _source_ids_from_reference(population_reference)
    population_source_ids = parsed_population_source_ids or ((config_id,) if selected_annual_patients is not None else ())
    assumptions = [
        assumption(
            "commercial-annual-patients",
            "annual_patients",
            selected_annual_patients,
            "patients",
            population_reference or "commercial_market_sizing",
            assumption_type="user_reviewed" if interpretation.selected_population_measure.source_type == "user_override" else "source_derived" if selected_annual_patients is not None else "missing",
            source_ids=() if interpretation.selected_population_measure.source_type == "user_override" else population_source_ids,
            requires_human_review=interpretation.selected_population_measure.human_review_required,
        ),
        select_assumption_value(
            "commercial-peak-penetration",
            "peak_penetration",
            user_overrides.get("peak_penetration"),
            "fraction",
            config_value=peak_penetration.base,
            config_filename="default_archetypes.yaml",
            config_field_path=f"archetypes.{archetype_name}.peak_penetration.base",
        ),
        select_assumption_value(
            "commercial-gross-to-net",
            "gross_to_net",
            user_overrides.get("gross_to_net"),
            "fraction",
            config_value=gross_to_net.base,
            config_filename="default_archetypes.yaml",
            config_field_path=f"archetypes.{archetype_name}.gross_to_net.base",
        ),
    ]
    if launch_ramp:
        assumptions.append(
            assumption(
                "commercial-launch-ramp",
                "launch_ramp",
                list(launch_ramp),
                "fraction_by_year",
                config_provenance("default_archetypes.yaml", f"archetypes.{archetype_name}.launch_ramp", section="due_diligence"),
                assumption_type="config_default",
                source_ids=(config_id,),
                requires_human_review=True,
            )
        )
    return assumptions


def _ledger_records(
    *,
    interpretation: MarketSizingInterpretation,
    annual_wac: ValueTriplet,
    gross_to_net: ValueTriplet,
    gross_source: str,
    peak_penetration: ValueTriplet,
    peak_source: str,
    launch_ramp: tuple[float, ...],
    ramp_source: str,
) -> list[CommercialAssumptionLedgerRecord]:
    population = interpretation.selected_population_measure
    records = [
        CommercialAssumptionLedgerRecord(
            assumption_name="starting_population",
            value=population.value,
            unit=population.unit,
            source_type=population.source_type,
            rationale=population.rationale,
            evidence_reference=population.evidence_reference,
            confidence_score=population.confidence_score,
            human_review_required=population.human_review_required,
        ),
        _fraction_record("diagnosed_fraction", interpretation.diagnosed_fraction),
        _fraction_record("treated_fraction", interpretation.treated_fraction),
        _fraction_record("eligibility_fraction", interpretation.eligibility_fraction),
        _fraction_record("commercially_addressable_fraction", interpretation.commercially_addressable_fraction),
        CommercialAssumptionLedgerRecord(assumption_name="annual_gross_wac", low=annual_wac.low, base=annual_wac.base, high=annual_wac.high, unit="USD/year", source_type="source_derived" if annual_wac.base is not None else "missing", rationale="Annual gross WAC from PharmaOS pricing module.", evidence_reference="pricing.annual_wac", confidence_score=8 if annual_wac.base is not None else 0, human_review_required=annual_wac.base is None),
        CommercialAssumptionLedgerRecord(assumption_name="gross_to_net", low=gross_to_net.low, base=gross_to_net.base, high=gross_to_net.high, unit="fraction", source_type=gross_source, rationale="Gross-to-net assumption used for deterministic revenue forecast.", evidence_reference="default_archetypes.yaml or user override", confidence_score=5, human_review_required=True),  # type: ignore[arg-type]
        CommercialAssumptionLedgerRecord(assumption_name="peak_penetration", low=peak_penetration.low, base=peak_penetration.base, high=peak_penetration.high, unit="fraction", source_type=peak_source, rationale="Peak penetration assumption used for deterministic forecast.", evidence_reference="default_archetypes.yaml or user override", confidence_score=5, human_review_required=True),  # type: ignore[arg-type]
        CommercialAssumptionLedgerRecord(assumption_name="launch_ramp", value=list(launch_ramp), unit="fraction_by_year", source_type=ramp_source, rationale="Launch ramp assumption used for deterministic forecast.", evidence_reference="default_archetypes.yaml or user override", confidence_score=5, human_review_required=True),  # type: ignore[arg-type]
    ]
    return records


def _fraction_record(name: str, value: CommercialAssumptionTriplet) -> CommercialAssumptionLedgerRecord:
    return CommercialAssumptionLedgerRecord(
        assumption_name=name,
        low=value.low,
        base=value.base,
        high=value.high,
        unit="fraction",
        source_type=value.source_type,
        rationale=value.rationale,
        evidence_reference=value.evidence_reference,
        confidence_score=value.confidence_score,
        human_review_required=value.human_review_required,
    )


def _bundle_summary(bundle: CommercialInputBundle) -> dict[str, Any]:
    return {
        "us_population_denominator": bundle.us_population_denominator,
        "disease_population_evidence_count": len(bundle.disease_population_evidence),
        "prevalence_evidence_count": len(bundle.prevalence_evidence),
        "incidence_evidence_count": len(bundle.incidence_evidence),
        "segmentation_evidence_count": len(bundle.segmentation_evidence),
        "has_trial_eligibility": bool(bundle.trial_eligibility_criteria),
        "pricing_source": bundle.pricing_benchmark.get("source"),
        "market_query_diagnostics": list(bundle.market_query_diagnostics),
        "missing_inputs": list(bundle.missing_inputs),
    }


def _human_review_questions(interpretation: MarketSizingInterpretation, missing_values: tuple[str, ...]) -> list[str]:
    questions = [
        "Confirm the selected population measure and market basis before using revenue forecast.",
        "Confirm diagnosed, treated, eligibility, and commercially addressable fractions.",
        "Confirm gross-to-net, peak penetration, and launch ramp assumptions.",
    ]
    if missing_values:
        questions.append(f"Resolve missing required commercial inputs: {', '.join(missing_values)}.")
    questions.extend(interpretation.human_review_flags)
    return list(dict.fromkeys(questions))


def _confidence(calculable: bool, confidence_score: int, flags: list[MissingDataFlag]) -> float:
    if not calculable:
        return 0.1 if any(flag.severity == "high" for flag in flags) else 0.25
    return max(0.2, min(0.85, confidence_score / 10 if confidence_score else 0.65))


def _dedupe_flags(flags: list[MissingDataFlag]) -> list[MissingDataFlag]:
    deduped: dict[str, MissingDataFlag] = {}
    for flag in flags:
        deduped[flag.flag_id] = flag
    return list(deduped.values())


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.casefold()).strip("-") or "unknown"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
