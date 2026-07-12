"""Lens patent and LOE due-diligence tool."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from typing import Any

import httpx
from pydantic import Field

from pharma_os.agent_runtime import run_structured_llm_call, runtime_config_for_route
from pharma_os.schemas import AssetIdentityOutput, MissingDataFlag, PatentCandidate, PatentExclusivityOutput, SourceMetadata, StrictSchema
from pharma_os.tools._due_diligence_common import LENS_PATENT_URL, first_text, missing, slug


PATENT_INCLUDE = [
    "lens_id",
    "jurisdiction",
    "doc_number",
    "kind",
    "date_published",
    "publication_type",
    "biblio.publication_reference",
    "biblio.application_reference",
    "biblio.application_number",
    "biblio.priority_claims",
    "biblio.invention_title",
    "biblio.parties",
    "families",
    "abstract",
    "claims",
    "legal_status",
]
MINIMAL_PATENT_INCLUDE = [
    "lens_id",
    "biblio.invention_title",
    "abstract",
    "legal_status",
]
CORPORATE_SUFFIXES = {
    "incorporated",
    "inc",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "company",
    "co",
    "therapeutics",
    "pharmaceuticals",
    "pharmaceutical",
    "pharma",
    "biosciences",
    "bioscience",
    "biotherapeutics",
    "biotech",
    "bio",
}


class PatentCandidateAdjudication(StrictSchema):
    """AI or fallback relevance decision for one Lens candidate."""

    candidate_id: str = Field(..., min_length=1)
    plausibly_covers_asset: bool = False
    confidence_score: int = Field(default=0, ge=0, le=10)
    rationale: str = Field(..., min_length=1)
    protection_type: str = "unknown"
    supporting_evidence: tuple[str, ...] = Field(default_factory=tuple)
    uncertainties: tuple[str, ...] = Field(default_factory=tuple)


class PatentCandidateReviewOutput(StrictSchema):
    """Structured relevance review over sponsor-retrieved Lens candidates."""

    selected_candidate_id: str | None = None
    confidence_score: int = Field(default=0, ge=0, le=10)
    rationale: str = Field(..., min_length=1)
    adjudications: tuple[PatentCandidateAdjudication, ...] = Field(default_factory=tuple)
    human_review_flags: tuple[str, ...] = Field(default_factory=tuple)


def search_patent_exclusivity(
    asset: AssetIdentityOutput,
    *,
    loe_year_override: int | None = None,
    client: httpx.Client | None = None,
) -> tuple[PatentExclusivityOutput, tuple[SourceMetadata, ...]]:
    """Search Lens, adjudicate relevance, and derive LOE only from the selected plausible family."""

    token = os.getenv("LENS_API_TOKEN")
    asset_context = _asset_context(asset)
    source = SourceMetadata(
        source_id=f"lens:{slug(asset.asset_name or asset.nct_id)}",
        title=f"Lens patent search for {asset.asset_name or asset.nct_id}",
        url=LENS_PATENT_URL,
        provenance="Lens Patent Search API",
        source_type="patent_search",
        version="v1",
    )
    override_source = SourceMetadata(
        source_id=f"human_override:loe:{slug(asset.nct_id)}",
        title=f"Reviewed LOE override for {asset.nct_id}",
        provenance="CLI supplied reviewed LOE year",
        source_type="human_override",
        version="local",
    )
    queries = _build_lens_queries(asset_context)
    searched_terms = tuple(term for query in queries for term in query["terms"])
    flags: list[MissingDataFlag] = []
    candidates: list[PatentCandidate] = []
    api_errors: list[str] = []
    if not token:
        flags.append(missing("patent-lens-token-missing", "patent_exclusivity", "loe_year", "LENS_API_TOKEN is missing; Lens retrieval skipped.", "high"))
    elif not queries:
        flags.append(missing("patent-search-terms-missing", "patent_exclusivity", "searched_terms", "No sponsor or asset terms were available for patent search.", "high"))
    else:
        http_client = client or httpx.Client(timeout=20.0)
        max_candidates = _env_int("PHARMA_OS_LENS_MAX_RESULTS", 25, minimum=1, maximum=50)
        deduped: dict[str, PatentCandidate] = {}
        for query in queries:
            try:
                response, fallback_note = _post_lens_query(http_client, token, query["query"], max_candidates)
                if fallback_note:
                    api_errors.append(f"{query['query_id']}: {fallback_note}")
                if response.status_code in {204, 404}:
                    continue
                if response.status_code not in range(200, 300):
                    api_errors.append(f"{query['query_id']}: Lens returned HTTP {response.status_code}.")
                    continue
                for raw in _lens_records(response.json()):
                    candidate = _normalize_lens_candidate(raw, source.source_id, tuple(query["terms"]))
                    if not candidate.candidate_id:
                        continue
                    existing = deduped.get(candidate.candidate_id)
                    if existing is None:
                        deduped[candidate.candidate_id] = candidate
                    else:
                        deduped[candidate.candidate_id] = existing.model_copy(
                            update={
                                "matched_query_terms": tuple(sorted(set((*existing.matched_query_terms, *candidate.matched_query_terms)))),
                                "matched_company_terms": tuple(sorted(set((*existing.matched_company_terms, *candidate.matched_company_terms)))),
                            }
                        )
            except Exception as exc:
                api_errors.append(f"{query['query_id']}: Lens request failed: {exc.__class__.__name__}.")
        candidates = _rank_candidates(tuple(deduped.values()))[:max_candidates]
        if not candidates:
            flags.append(missing("patent-no-candidates", "patent_exclusivity", "candidates", "No Lens patent candidates were retrieved.", "medium"))
        for index, error in enumerate(api_errors[:5], start=1):
            flags.append(missing(f"patent-lens-api-error-{index}", "patent_exclusivity", "candidates", error, "medium"))

    review = _review_patent_candidates(asset_context, tuple(candidates), source_ids=tuple(dict.fromkeys((*asset.source_ids, source.source_id))))
    selected = _selected_candidate(tuple(candidates), review.selected_candidate_id)
    selected_id = selected.candidate_id if selected is not None else None
    loe = _estimate_loe(selected)
    estimated_loe_year = loe_year_override if loe_year_override is not None else loe["estimated_loe_year"]
    loe_method = "human reviewed override" if loe_year_override is not None else loe["loe_method"]
    if selected is None and candidates:
        flags.append(
            missing(
                "patent-no-plausible-selected-family",
                "patent_exclusivity",
                "selected_candidate_id",
                "Lens retrieved candidates, but none was adjudicated as plausibly covering the target asset.",
                "high",
            )
        )
    if estimated_loe_year is None:
        flags.append(missing("patent-loe-review-required", "patent_exclusivity", "estimated_loe_year", "No credible selected patent family or reviewed LOE year was available.", "high"))
    for flag in review.human_review_flags:
        flags.append(missing(f"patent-review-{slug(flag)}", "patent_exclusivity", "selected_candidate_id", flag, "medium"))
    output_sources = []
    if token:
        output_sources.append(source)
    if loe_year_override is not None:
        output_sources.append(override_source)
    confidence = (
        0.75
        if loe_year_override is not None
        else min(0.85, max(0.35, review.confidence_score / 10))
        if selected is not None and estimated_loe_year is not None
        else 0.2
    )
    return (
        PatentExclusivityOutput(
            asset_name=asset.asset_name,
            asset_context=asset_context,
            searched_terms=searched_terms,
            candidates=tuple(candidates),
            selected_candidate_id=selected_id,
            selected_candidate_rationale=review.rationale,
            selected_candidate_confidence=review.confidence_score / 10,
            loe_method=loe_method,
            estimated_loe_year=estimated_loe_year,
            source_ids=tuple(item.source_id for item in output_sources),
            missing_data_flags=tuple(flags),
            confidence=confidence,
        ),
        tuple(output_sources),
    )


def _asset_context(asset: AssetIdentityOutput) -> dict[str, Any]:
    sponsor_aliases = _sponsor_company_terms(asset.sponsor)
    return {
        "nct_id": asset.nct_id,
        "asset_name": asset.asset_name,
        "asset_aliases": asset.aliases,
        "raw_intervention_names": asset.raw_intervention_names,
        "sponsor": asset.sponsor,
        "sponsor_aliases": tuple(term for term in sponsor_aliases if asset.sponsor and term.casefold() != asset.sponsor.casefold()),
        "indication": asset.normalized_indication,
        "therapeutic_area": asset.therapeutic_area,
        "modality": asset.modality,
        "route": getattr(asset, "route", None),
        "mechanism": getattr(asset, "mechanism", None),
        "rxnorm_match": asset.rxnorm_match.matched_name if asset.rxnorm_match else None,
        "identity_source_ids": asset.source_ids,
    }


def _build_lens_queries(asset_context: dict[str, Any]) -> list[dict[str, Any]]:
    sponsor_terms = _sponsor_company_terms(asset_context.get("sponsor"))
    if not sponsor_terms:
        sponsor_terms = _unique_terms([asset_context.get("asset_name"), *(asset_context.get("asset_aliases") or ())])
    queries = []
    for term in sponsor_terms[:6]:
        queries.append(
            {
                "query_id": f"sponsor_company_{len(queries) + 1}",
                "terms": [term],
                "query": {
                    "bool": {
                        "should": [
                            {"match_phrase": {"applicant.name": term}},
                            {"match_phrase": {"owner_all.name": term}},
                            {"match_phrase": {"biblio.parties.applicants.name": term}},
                            {"match_phrase": {"biblio.parties.owners.name": term}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
        )
    return queries


def _post_lens_query(client: httpx.Client, token: str, query: dict[str, Any], size: int) -> tuple[httpx.Response, str | None]:
    body = {
        "query": query,
        "size": size,
        "from": 0,
        "group_by": "SIMPLE_FAMILY",
        "sort": [{"relevance": "desc"}],
        "include": PATENT_INCLUDE,
    }
    response = client.post(
        LENS_PATENT_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=body,
        timeout=20.0,
    )
    if response.status_code in {300, 400, 415}:
        minimal = {**body, "include": MINIMAL_PATENT_INCLUDE}
        fallback = client.post(
            LENS_PATENT_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=minimal,
            timeout=20.0,
        )
        if fallback.status_code in range(200, 300) or fallback.status_code in {204, 404}:
            return fallback, "default Lens include list failed; retried with minimal include list"
        return fallback, f"default Lens include list failed and minimal retry returned HTTP {fallback.status_code}"
    return response, None


def _review_patent_candidates(
    asset_context: dict[str, Any],
    candidates: tuple[PatentCandidate, ...],
    *,
    source_ids: tuple[str, ...],
) -> PatentCandidateReviewOutput:
    fallback = _fallback_patent_review(asset_context, candidates)
    if not candidates:
        return fallback
    result = run_structured_llm_call(
        agent_name="PatentFamilyRelevanceReviewerAgent",
        instructions=_patent_review_instructions(),
        payload={
            "asset_context": asset_context,
            "lens_candidates": [candidate.model_dump(mode="json") for candidate in candidates[:15]],
            "selection_rules": (
                "All candidates were retrieved by sponsor/company search, so sponsor match alone is insufficient.",
                "Select no candidate when title, abstract, claims, identifiers, or dates do not plausibly connect the family to the target asset.",
                "Do not calculate LOE.",
            ),
        },
        output_type=PatentCandidateReviewOutput,
        run_id=f"patent-family-review-{slug(str(asset_context.get('nct_id') or asset_context.get('asset_name') or 'asset'))}",
        input_summary=f"Review Lens patent candidates for {asset_context.get('asset_name') or asset_context.get('nct_id')}.",
        config=runtime_config_for_route(
            model_route="agent4_subagent",
            disabled_provenance="pharma_os.tools.patents_lens",
        ),
        offline_output=fallback,
        source_ids=source_ids,
        confidence=0.5,
        rationale_summary="Adjudicate sponsor-retrieved Lens candidates before selected-family-only LOE calculation.",
    )
    selected = result.output
    if selected.selected_candidate_id and selected.confidence_score < 4:
        return selected.model_copy(update={"selected_candidate_id": None, "human_review_flags": (*selected.human_review_flags, "low_confidence_candidate_not_selected")})
    if selected.selected_candidate_id and not _selected_candidate(candidates, selected.selected_candidate_id):
        return selected.model_copy(update={"selected_candidate_id": None, "human_review_flags": (*selected.human_review_flags, "selected_candidate_missing_from_lens_results")})
    return selected


def _patent_review_instructions() -> str:
    return (
        "You are PharmaOS PatentFamilyRelevanceReviewerAgent. Use only supplied asset context and Lens candidates. "
        "Lens candidates were retrieved by sponsor/company search; do not assume sponsor patents cover the asset. "
        "Select the best candidate only when asset name, alias/code name, mechanism, modality, route, indication, title, abstract, or claims create a plausible asset-family link. "
        "Explicitly return selected_candidate_id null when no candidate plausibly covers the target asset. Do not calculate LOE or invent dates."
    )


def _fallback_patent_review(asset_context: dict[str, Any], candidates: tuple[PatentCandidate, ...]) -> PatentCandidateReviewOutput:
    decisions = tuple(_fallback_candidate_decision(asset_context, candidate) for candidate in candidates[:15])
    plausible = [decision for decision in decisions if decision.plausibly_covers_asset]
    if not plausible:
        return PatentCandidateReviewOutput(
            selected_candidate_id=None,
            confidence_score=0,
            rationale="No Lens candidate contained an asset/code-name, alias, or other strong target-asset link; sponsor ownership alone was not treated as covering evidence.",
            adjudications=decisions,
            human_review_flags=("no_plausible_covering_patent_family_selected",) if candidates else ("no_lens_candidates_found",),
        )
    selected = sorted(plausible, key=lambda item: item.confidence_score, reverse=True)[0]
    return PatentCandidateReviewOutput(
        selected_candidate_id=selected.candidate_id,
        confidence_score=selected.confidence_score,
        rationale=selected.rationale,
        adjudications=decisions,
        human_review_flags=("prototype_patent_triage_requires_ip_review",),
    )


def _fallback_candidate_decision(asset_context: dict[str, Any], candidate: PatentCandidate) -> PatentCandidateAdjudication:
    strong_terms = _unique_terms([
        asset_context.get("asset_name"),
        *(asset_context.get("asset_aliases") or ()),
        *(asset_context.get("raw_intervention_names") or ()),
        asset_context.get("rxnorm_match"),
    ])
    weak_terms = _unique_terms([
        asset_context.get("indication"),
        asset_context.get("modality"),
        asset_context.get("route"),
        asset_context.get("mechanism"),
    ])
    haystack = _word_text(
        candidate.title,
        candidate.abstract_excerpt,
        candidate.claim_excerpt,
        " ".join(candidate.publication_or_application_identifiers),
    )
    strong_hits = [term for term in strong_terms if _term_in_text(term, haystack)]
    weak_hits = [term for term in weak_terms if _term_in_text(term, haystack)]
    if strong_hits:
        score = min(8, 5 + len(strong_hits) + min(2, len(weak_hits)))
        return PatentCandidateAdjudication(
            candidate_id=candidate.candidate_id,
            plausibly_covers_asset=True,
            confidence_score=score,
            rationale=f"Candidate contains target asset/code-name evidence: {', '.join(strong_hits[:3])}.",
            protection_type=_protection_type(candidate),
            supporting_evidence=tuple(strong_hits[:5]),
            uncertainties=("Patent-family relevance still requires IP counsel review.",),
        )
    return PatentCandidateAdjudication(
        candidate_id=candidate.candidate_id,
        plausibly_covers_asset=False,
        confidence_score=0,
        rationale="No target asset/code-name or alias evidence was found in normalized title, abstract, claims, or identifiers.",
        protection_type="unknown",
        supporting_evidence=(),
        uncertainties=tuple(weak_hits[:5]) or ("Sponsor-retrieved candidate may be unrelated to the target asset.",),
    )


def _normalize_lens_candidate(raw: dict[str, Any], source_id: str, matched_terms: tuple[str, ...]) -> PatentCandidate:
    lens_id = first_text(raw.get("lens_id")) or first_text(_deep_get(raw, "biblio.lens_id"))
    identifiers = _identifiers(raw)
    candidate_id = lens_id or (identifiers[0] if identifiers else "")
    jurisdictions = _unique_terms([first_text(raw.get("jurisdiction")), *_family_jurisdictions(raw)])
    priority_date = _first_date(raw.get("earliest_priority_claim_date") or _deep_find(raw, "priority_claims"))
    application_date = _first_date(raw.get("application_reference") or _deep_get(raw, "biblio.application_reference"))
    publication_date = _first_date(raw.get("date_published") or _deep_find(raw, "date_published"))
    anticipated_term_date = _first_date(_deep_find(raw, "anticipated_term_date") or _deep_find(raw, "term_date") or _deep_find(raw, "adjusted_expiration"))
    return PatentCandidate(
        candidate_id=candidate_id,
        lens_id=lens_id,
        title=_first_title(raw),
        abstract_excerpt=_excerpt(first_text(raw.get("abstract") or _deep_find(raw, "abstract"))),
        claim_excerpt=_excerpt(first_text(raw.get("claim") or raw.get("claims") or _deep_find(raw, "claims"))),
        applicants_or_owners=tuple(_party_names(raw)),
        jurisdictions=tuple(jurisdictions),
        jurisdiction=jurisdictions[0] if jurisdictions else None,
        priority_date=priority_date,
        application_date=application_date,
        publication_date=publication_date,
        anticipated_term_date=anticipated_term_date,
        publication_or_application_identifiers=tuple(identifiers),
        legal_status=_legal_status(raw),
        matched_query_terms=matched_terms,
        matched_company_terms=matched_terms,
        source_search_strategy="sponsor_only_search",
        source_url=f"https://www.lens.org/lens/patent/{lens_id}" if lens_id else None,
        source_id=source_id,
    )


def _estimate_loe(candidate: PatentCandidate | None, *, current_year: int | None = None) -> dict[str, Any]:
    current_year = current_year or datetime.now(timezone.utc).year
    if candidate is None:
        return {"estimated_loe_year": None, "loe_method": None, "human_review_flags": ("loe_not_calculated_no_selected_candidate",)}
    term_date = _parse_date(candidate.anticipated_term_date)
    priority_date = _parse_date(candidate.priority_date)
    application_date = _parse_date(candidate.application_date)
    publication_date = _parse_date(candidate.publication_date)
    if term_date:
        original_year = term_date.year
        method = "Lens anticipated term date"
    elif priority_date:
        original_year = priority_date.year + 20
        method = "earliest priority date + 20 years"
    elif application_date:
        original_year = application_date.year + 20
        method = "application date + 20 years"
    elif publication_date:
        original_year = publication_date.year + 20
        method = "publication date + 20 years"
    else:
        return {"estimated_loe_year": None, "loe_method": None, "human_review_flags": ("loe_not_calculated_missing_patent_dates",)}
    capped_year = original_year
    if priority_date:
        capped_year = min(capped_year, priority_date.year + 25)
    capped_year = min(capped_year, current_year + 25)
    return {
        "estimated_loe_year": capped_year,
        "loe_method": method if capped_year == original_year else f"{method}; capped by plausibility guardrail",
        "human_review_flags": ("loe_cap_applied",) if capped_year != original_year else (),
    }


def _lens_records(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") or payload.get("results") or []
    if isinstance(data, dict):
        data = data.get("data") or data.get("results") or []
    return [item for item in data if isinstance(item, dict)]


def _first_title(raw: dict[str, Any]) -> str | None:
    biblio = raw.get("biblio") if isinstance(raw.get("biblio"), dict) else {}
    return first_text(raw.get("title") or biblio.get("invention_title"))


def _legal_status(raw: dict[str, Any]) -> str | None:
    parts: list[str] = []
    legal_status = raw.get("legal_status")
    if isinstance(legal_status, str):
        parts.append(legal_status)
    elif isinstance(legal_status, dict):
        for label, keys in (
            ("Legal Status", ("status", "patent_status", "grant_status")),
            ("Anticipated Term Date", ("anticipated_term_date", "term_date", "adjusted_expiration")),
            ("Priority Date", ("priority_date", "earliest_priority_claim_date")),
            ("Application Filing Date", ("application_filing_date", "filing_date", "application_date")),
        ):
            found = next((_first_date(legal_status.get(key)) for key in keys if _first_date(legal_status.get(key))), None)
            if found:
                parts.append(f"{label}: {found}")
    for label, value in (
        ("Priority Date", raw.get("earliest_priority_claim_date") or _deep_find(raw, "priority_claims")),
        ("Application Filing Date", raw.get("application_reference") or _deep_get(raw, "biblio.application_reference")),
        ("Anticipated Term Date", _deep_find(raw, "anticipated_term_date") or _deep_find(raw, "term_date")),
    ):
        found = _first_date(value)
        if found and not any(label.casefold() in part.casefold() for part in parts):
            parts.append(f"{label}: {found}")
    return "; ".join(parts) or None


def _identifiers(raw: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("doc_key", "ids", "application_number", "doc_number"):
        values.extend(_string_values(raw.get(key)))
    values.extend(_string_values(_deep_get(raw, "biblio.publication_reference.doc_number")))
    values.extend(_string_values(_deep_get(raw, "biblio.application_reference.doc_number")))
    values.extend(_string_values(_deep_get(raw, "biblio.application_number")))
    values.extend(_string_values(_deep_find(raw, "document_id")))
    return _unique_terms(values)


def _party_names(raw: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for path in (
        "applicant.name",
        "owner_all.name",
        "biblio.parties.applicants.name",
        "biblio.parties.owners.name",
        "biblio.parties.current_assignees.name",
    ):
        values.extend(_string_values(_deep_get(raw, path)))
    return _unique_terms(values)


def _family_jurisdictions(raw: dict[str, Any]) -> list[str]:
    values: list[str] = []
    family = raw.get("family") or raw.get("families")
    for family_type in ("simple", "extended", "simple_family", "extended_family"):
        members = _deep_get(family, f"{family_type}.member") or _deep_get(family, f"{family_type}.members")
        if isinstance(members, list):
            for member in members:
                values.extend(_string_values(_deep_get(member, "document_id.jurisdiction")))
    return _unique_terms(values)


def _rank_candidates(candidates: tuple[PatentCandidate, ...]) -> list[PatentCandidate]:
    return sorted(candidates, key=lambda item: (_has_pharma_terms(item), item.anticipated_term_date or item.priority_date or item.application_date or item.publication_date or ""), reverse=True)


def _has_pharma_terms(candidate: PatentCandidate) -> int:
    text = _word_text(candidate.title, candidate.abstract_excerpt, candidate.claim_excerpt)
    terms = ("compound", "inhibitor", "pharmaceutical", "composition", "method of treatment", "formulation", "antibody", "kinase", "receptor", "oral", "tablet", "injection")
    return sum(1 for term in terms if term in text)


def _selected_candidate(candidates: tuple[PatentCandidate, ...], candidate_id: str | None) -> PatentCandidate | None:
    if not candidate_id:
        return None
    return next((candidate for candidate in candidates if candidate.candidate_id == candidate_id), None)


def _protection_type(candidate: PatentCandidate) -> str:
    text = _word_text(candidate.title, candidate.abstract_excerpt, candidate.claim_excerpt)
    if "composition" in text or "compound" in text:
        return "composition_of_matter"
    if "method of treatment" in text or "treating" in text:
        return "method_of_use"
    if "formulation" in text:
        return "formulation"
    return "unknown"


def _sponsor_company_terms(sponsor_name: str | None) -> list[str]:
    terms = _unique_terms([sponsor_name])
    expanded: list[str] = []
    for term in terms:
        expanded.append(term)
        core = _normalize_sponsor_core_name(term)
        if core and core.casefold() != term.casefold():
            expanded.append(core)
    return _unique_terms(expanded)


def _normalize_sponsor_core_name(name: str | None) -> str | None:
    if not name:
        return None
    text = re.sub(r"[,.]", " ", name)
    tokens = [token for token in re.split(r"\s+", text.strip()) if token]
    while tokens and tokens[-1].casefold() in CORPORATE_SUFFIXES:
        tokens.pop()
    core = " ".join(tokens).strip()
    return core or name.strip()


def _first_date(value: Any) -> str | None:
    for item in _string_values(value):
        parsed = _parse_date(item)
        if parsed:
            return parsed.isoformat()
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    if match := re.search(r"(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?", text):
        text = match.group()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_string_values(item))
        return values
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_string_values(item))
        return values
    return []


def _deep_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            current = current[index] if index < len(current) else None
        else:
            return None
    return current


def _deep_find(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _deep_find(item, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _deep_find(item, key)
            if found is not None:
                return found
    return None


def _excerpt(value: str | None, *, max_chars: int = 900) -> str | None:
    if not value:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def _word_text(*values: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+-]+", " ", " ".join(value or "" for value in values).casefold())).strip()


def _term_in_text(term: str, haystack: str) -> bool:
    needle = _word_text(term)
    return bool(needle and needle in haystack)


def _unique_terms(values: list[Any] | tuple[Any, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        term = value.strip()
        if not term or term.casefold() in {"unknown", "n/a", "none"}:
            continue
        key = term.casefold()
        if key not in seen:
            seen.add(key)
            result.append(term)
    return result


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))
