"""Deterministic Agent 4 due-diligence section builders."""

from __future__ import annotations

import os
from typing import Iterable

import httpx

from pharma_os.schemas import (
    AssetIdentityOutput,
    AssetMemo,
    AssumptionRecord,
    ClinicalEvidenceSummary,
    ClinicalOutcomePredictionOutput,
    ClinicalRiskSummary,
    ClinicalTrialRecord,
    CommercialModelOutput,
    CompetitiveLandscapeSummary,
    DiligenceRedFlag,
    EvidenceClaim,
    MissingDataFlag,
    PatentExclusivityOutput,
    PatentLOEReview,
    PricingOutput,
    RNPVOutput,
    SafetyLabelSummary,
    SourceMetadata,
)
from pharma_os.tools._due_diligence_common import OPENFDA_LABEL_URL, first_text, missing, slug
from pharma_os.tools.pubmed import PubMedClient, PubMedError


def build_clinical_evidence_summary(
    *,
    run_id: str,
    trial: ClinicalTrialRecord,
    asset: AssetIdentityOutput,
    pubmed_client: PubMedClient | None = None,
) -> tuple[ClinicalEvidenceSummary, tuple[SourceMetadata, ...], tuple[EvidenceClaim, ...]]:
    """Build CT.gov and PubMed evidence summary with source IDs."""

    query_terms = [trial.nct_id, asset.asset_name, *(trial.conditions[:1])]
    query = " OR ".join(f'"{term}"' for term in query_terms if term)
    flags: list[MissingDataFlag] = []
    sources: list[SourceMetadata] = []
    pubmed_titles: list[str] = []
    claims: list[EvidenceClaim] = [
        EvidenceClaim(
            claim_id=f"claim-{run_id}-clinical-evidence-ctgov",
            claim_text=f"Target trial {trial.nct_id} has status {trial.overall_status or 'unknown'} and enrollment {trial.enrollment_count if trial.enrollment_count is not None else 'unknown'}.",
            source_ids=(trial.source_id,),
            provenance="due_diligence.clinical_evidence.ctgov",
            confidence=0.95,
            confidence_level="very_high",
        )
    ]
    if query:
        try:
            articles = (pubmed_client or PubMedClient()).search(query, max_results=_env_int("PHARMA_OS_PUBMED_MAX_RESULTS", 5, minimum=0, maximum=50))
            for article in articles:
                source = article.source()
                sources.append(source)
                pubmed_titles.append(article.title)
                claims.append(
                    EvidenceClaim(
                        claim_id=f"claim-{run_id}-pubmed-{article.pmid}",
                        claim_text=f"PubMed article {article.pmid} was retrieved for the due-diligence evidence query.",
                        source_ids=(source.source_id,),
                        provenance="due_diligence.clinical_evidence.pubmed",
                        confidence=0.8,
                        confidence_level="medium",
                    )
                )
            if not articles:
                flags.append(missing("clinical-evidence-pubmed-empty", "clinical_evidence", "pubmed_titles", "PubMed returned no article metadata for the bounded evidence query.", "medium"))
        except PubMedError as exc:
            flags.append(missing("clinical-evidence-pubmed-error", "clinical_evidence", "pubmed_titles", f"PubMed retrieval failed: {exc.__class__.__name__}.", "medium"))
    else:
        flags.append(missing("clinical-evidence-query-missing", "clinical_evidence", "pubmed_query", "No asset, NCT, or condition terms were available for PubMed query construction.", "medium"))

    summary = f"CT.gov target trial evidence includes status {trial.overall_status or 'unknown'}, phase {', '.join(trial.phases) or 'unknown'}, and {len(trial.primary_endpoints)} primary endpoints."
    return (
        ClinicalEvidenceSummary(
            nct_id=trial.nct_id,
            ctgov_summary=summary,
            pubmed_query=query or None,
            pubmed_article_count=len(pubmed_titles),
            pubmed_titles=tuple(pubmed_titles),
            source_ids=tuple(dict.fromkeys((trial.source_id, *(source.source_id for source in sources)))),
            claims=tuple(claims),
            missing_data_flags=tuple(flags),
            confidence=0.75 if not flags else 0.45,
        ),
        tuple(sources),
        tuple(claims),
    )


def build_competitive_landscape_summary(agent3_output: ClinicalOutcomePredictionOutput) -> CompetitiveLandscapeSummary:
    """Build competitive landscape from Agent 3 comparator context only."""

    comparator = agent3_output.comparator_benchmarking
    return CompetitiveLandscapeSummary(
        nct_id=agent3_output.trial_identity.nct_id,
        comparator_trial_ids=comparator.comparator_trial_ids,
        matched_public_trials_count=comparator.matched_public_trials_count,
        benchmark_summary=comparator.benchmark_summary,
        status_summary=comparator.status_summary,
        phase_summary=comparator.phase_summary,
        sponsor_summary=comparator.sponsor_summary,
        endpoint_summary=comparator.endpoint_summary,
        population_summary=comparator.population_summary,
        source_ids=comparator.source_ids,
        missing_data_flags=comparator.missing_data_flags,
        confidence=comparator.confidence,
    )


def build_safety_label_summary(
    asset: AssetIdentityOutput,
    *,
    client: httpx.Client | None = None,
) -> tuple[SafetyLabelSummary, tuple[SourceMetadata, ...]]:
    """Summarize openFDA label safety fields without inference."""

    terms = tuple(dict.fromkeys(term for term in (asset.asset_name, *asset.aliases) if term))
    if not terms:
        flag = missing("safety-label-terms-missing", "safety_label_summary", "asset_name", "No asset terms were available for openFDA label lookup.", "medium")
        return SafetyLabelSummary(asset_name=asset.asset_name, missing_data_flags=(flag,), confidence=0.0), ()
    http_client = client or httpx.Client(timeout=20.0)
    for term in terms[:5]:
        try:
            payload = _openfda_label(http_client, term)
        except Exception:
            continue
        if not payload:
            continue
        source = SourceMetadata(
            source_id=f"openfda_label:{slug(term)}",
            title=f"openFDA label search for {term}",
            url=OPENFDA_LABEL_URL,
            provenance="openFDA drug label API",
            source_type="drug_label",
            version="openFDA",
        )
        return (
            SafetyLabelSummary(
                asset_name=asset.asset_name,
                label_available=True,
                warnings_summary=_trim(first_text(payload.get("warnings"))),
                adverse_reactions_summary=_trim(first_text(payload.get("adverse_reactions"))),
                source_ids=(source.source_id,),
                confidence=0.7,
            ),
            (source,),
        )
    flag = missing("safety-label-not-found", "safety_label_summary", "label_available", "openFDA returned no usable label match for asset terms.", "medium")
    return SafetyLabelSummary(asset_name=asset.asset_name, missing_data_flags=(flag,), confidence=0.15), ()


def build_patent_loe_review(patent: PatentExclusivityOutput) -> PatentLOEReview:
    """Convert Lens-only patent output into Agent 4 patent/LOE review."""

    if patent.estimated_loe_year is None:
        summary = "No reviewed LOE year is available; Agent 4 does not invent or default LOE."
    elif any(source_id.startswith("human_override:loe:") for source_id in patent.source_ids):
        summary = f"Reviewed LOE year is {patent.estimated_loe_year}; {len(patent.candidates)} Lens candidates were retrieved."
    elif patent.candidates:
        summary = f"Source-derived LOE year is {patent.estimated_loe_year} based on Lens patent-date evidence across {len(patent.candidates)} retrieved candidates; IP counsel review is still required."
    else:
        summary = f"Reviewed LOE year is {patent.estimated_loe_year}; no Lens patent family was selected from retrieved candidates."
    return PatentLOEReview(
        asset_name=patent.asset_name,
        searched_terms=patent.searched_terms,
        candidate_count=len(patent.candidates),
        estimated_loe_year=patent.estimated_loe_year,
        review_summary=summary,
        source_ids=patent.source_ids,
        missing_data_flags=patent.missing_data_flags,
        confidence=patent.confidence,
    )


def build_red_flags(
    *,
    clinical_risk: ClinicalRiskSummary,
    safety: SafetyLabelSummary,
    patent: PatentLOEReview,
    pricing: PricingOutput,
    commercial: CommercialModelOutput,
    rnpv: RNPVOutput,
    missing_data_flags: tuple[MissingDataFlag, ...],
) -> tuple[DiligenceRedFlag, ...]:
    """Build rule-based diligence red flags across Agent 4 sections."""

    flags: list[DiligenceRedFlag] = []
    if clinical_risk.endpoint_risk_level == "high" or clinical_risk.enrollment_duration_risk_level == "high":
        flags.append(_red("clinical-high-agent3-risk", "clinical", "high", "Agent 3 reports high clinical risk.", clinical_risk.source_ids))
    if safety.missing_data_flags:
        flags.append(_red("safety-label-missing", "safety", "medium", "Public openFDA safety label context is missing or unavailable.", safety.source_ids))
    if safety.warnings_summary and any(term in safety.warnings_summary.casefold() for term in ("boxed warning", "death", "fatal", "life-threatening", "serious")):
        flags.append(_red("safety-serious-label-warning", "safety", "high", "openFDA label warning text includes serious safety language requiring clinical review.", safety.source_ids))
    if patent.estimated_loe_year is None:
        flags.append(_red("ip-loe-missing", "ip_loe", "high", "LOE is missing; no fake/default LOE was assigned.", patent.source_ids))
    if patent.candidate_count == 0:
        flags.append(_red("ip-lens-no-family", "ip_loe", "high", "Lens returned no selected or retrieved patent family for review.", patent.source_ids))
    if pricing.annual_wac is None:
        flags.append(_red("pricing-missing", "pricing", "high", "Pricing is missing or lacks sourced dosing for annualization.", pricing.source_ids))
    if not commercial.calculable:
        flags.append(_red("commercial-not-calculable", "commercial", "high", "Commercial model is non-calculable under current evidence and assumptions.", commercial.source_ids))
    if not rnpv.calculable:
        flags.append(_red("rnpv-not-calculable", "rnpv", "high", "rNPV is non-calculable under current evidence and assumptions.", rnpv.source_ids))
    for flag in missing_data_flags:
        if flag.severity in {"high", "critical"}:
            flags.append(_red(f"missing-{flag.flag_id}", "source_coverage", flag.severity, flag.reason, ()))
    return tuple(_dedupe_red_flags(flags))


def build_asset_memo(
    *,
    run_id: str,
    asset: AssetIdentityOutput,
    clinical_risk: ClinicalRiskSummary,
    evidence: ClinicalEvidenceSummary,
    landscape: CompetitiveLandscapeSummary,
    safety: SafetyLabelSummary,
    patent: PatentLOEReview,
    pricing: PricingOutput,
    commercial: CommercialModelOutput,
    rnpv: RNPVOutput,
    red_flags: tuple[DiligenceRedFlag, ...],
    claims: tuple[EvidenceClaim, ...] = (),
    assumptions: tuple[AssumptionRecord, ...] = (),
    missing_data_flags: tuple[MissingDataFlag, ...] = (),
) -> AssetMemo:
    """Assemble a source-backed draft memo requiring human review."""

    title = f"Draft clinical-stage diligence memo for {asset.asset_name or asset.nct_id}"
    sections = (
        f"Asset identity: {asset.asset_name or 'unknown'}; sponsor {asset.sponsor or 'unknown'}; indication {asset.normalized_indication or 'unknown'}.",
        f"Agent 3 clinical risk: endpoint {clinical_risk.endpoint_risk_level or 'unknown'}, enrollment/duration {clinical_risk.enrollment_duration_risk_level or 'unknown'}.",
        f"Clinical evidence: {evidence.ctgov_summary}",
        f"Competitive landscape: {landscape.benchmark_summary}",
        f"Safety/label: {'openFDA label available' if safety.label_available else 'openFDA label not available'}.",
        f"Patent/LOE: {patent.review_summary}",
        f"Pricing/commercial/rNPV: annual WAC {pricing.annual_wac if pricing.annual_wac is not None else 'not available'}; peak net sales {commercial.peak_net_sales if commercial.peak_net_sales is not None else 'not calculable'}; rNPV {rnpv.rnpv if rnpv.rnpv is not None else 'not calculable'}.",
        f"Red flags: {len(red_flags)} rule-based flags require review.",
    )
    missing_evidence = tuple(dict.fromkeys(flag.reason for flag in missing_data_flags))
    assumption_lines = tuple(
        f"{assumption.name}: {assumption.value if assumption.value is not None else 'missing'} ({assumption.assumption_type})"
        for assumption in assumptions
    )
    claim_lines = tuple(claim.claim_text for claim in claims)
    questions = tuple(
        dict.fromkeys(
            [
                *("Review missing LOE or Lens patent evidence." for _ in [1] if patent.estimated_loe_year is None or patent.candidate_count == 0),
                *("Review missing pricing or dosing evidence." for _ in [1] if pricing.annual_wac is None),
                *("Review non-calculable commercial model or rNPV assumptions." for _ in [1] if not commercial.calculable or not rnpv.calculable),
                *("Review high clinical or safety risks before any business decision." for _ in [1] if any(flag.severity in {"high", "critical"} for flag in red_flags)),
            ]
        )
    )
    source_ids = tuple(
        dict.fromkeys(
            (
                *asset.source_ids,
                *clinical_risk.source_ids,
                *evidence.source_ids,
                *landscape.source_ids,
                *safety.source_ids,
                *patent.source_ids,
                *pricing.source_ids,
                *commercial.source_ids,
                *rnpv.source_ids,
                *(source_id for flag in red_flags for source_id in flag.source_ids),
            )
        )
    )
    return AssetMemo(
        memo_id=f"asset-memo-{run_id}",
        title=title,
        summary="Draft memo for human review only; it contains no final recommendations, approvals, legal conclusions, or invented values.",
        sections=sections,
        source_backed_claims=claim_lines,
        assumptions_summary=assumption_lines,
        missing_evidence=missing_evidence,
        review_questions=questions,
        source_ids=source_ids,
        requires_human_review=True,
        confidence=min(0.75, max(0.2, sum(item for item in (clinical_risk.confidence, evidence.confidence, landscape.confidence, safety.confidence, patent.confidence) if item) / 5)),
    )


def _openfda_label(client: httpx.Client, term: str) -> dict | None:
    for field in ("openfda.brand_name", "openfda.generic_name"):
        response = client.get(OPENFDA_LABEL_URL, params={"search": f'{field}:"{term}"', "limit": "1"}, timeout=20.0)
        if response.status_code == 404:
            continue
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(results, list) and results and isinstance(results[0], dict):
            return results[0]
    return None


def _red(flag_id: str, category: str, severity: str, reason: str, source_ids: tuple[str, ...]) -> DiligenceRedFlag:
    return DiligenceRedFlag(
        flag_id=flag_id,
        category=category,  # type: ignore[arg-type]
        severity=severity,  # type: ignore[arg-type]
        reason=reason,
        source_ids=source_ids,
        provenance="pharma_os.components.due_diligence_sections.red_flags",
    )


def _dedupe_red_flags(flags: Iterable[DiligenceRedFlag]) -> tuple[DiligenceRedFlag, ...]:
    deduped: dict[str, DiligenceRedFlag] = {}
    for flag in flags:
        deduped[flag.flag_id] = flag
    return tuple(deduped.values())


def _trim(value: str | None, limit: int = 1000) -> str | None:
    if value is None:
        return None
    compact = " ".join(value.split())
    return compact[:limit]


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))
