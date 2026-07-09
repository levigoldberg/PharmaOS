"""Lens-only patent and LOE due-diligence tool."""

from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from typing import Any

import httpx

from pharma_os.schemas import AssetIdentityOutput, MissingDataFlag, PatentCandidate, PatentExclusivityOutput, SourceMetadata
from pharma_os.tools._due_diligence_common import LENS_PATENT_URL, first_text, missing, slug


def search_patent_exclusivity(
    asset: AssetIdentityOutput,
    *,
    loe_year_override: int | None = None,
    client: httpx.Client | None = None,
) -> tuple[PatentExclusivityOutput, tuple[SourceMetadata, ...]]:
    """Search Lens when configured; otherwise require human LOE review."""

    token = os.getenv("LENS_API_TOKEN")
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
    terms = tuple(item for item in (asset.asset_name, asset.sponsor, *asset.aliases[:3]) if item)
    flags: list[MissingDataFlag] = []
    candidates: list[PatentCandidate] = []
    if not token:
        flags.append(missing("patent-lens-token-missing", "patent_exclusivity", "loe_year", "LENS_API_TOKEN is missing; Lens retrieval skipped.", "high"))
    elif not terms:
        flags.append(missing("patent-search-terms-missing", "patent_exclusivity", "searched_terms", "No asset/sponsor terms were available for patent search.", "high"))
    else:
        try:
            response = (client or httpx.Client(timeout=20.0)).post(
                LENS_PATENT_URL,
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json={
                    "query": {"query_string": {"query": " OR ".join(f'"{term}"' for term in terms)}},
                    "size": _env_int("PHARMA_OS_LENS_MAX_RESULTS", 5, minimum=1, maximum=50),
                    "from": 0,
                    "include": [
                        "lens_id",
                        "doc_number",
                        "biblio.invention_title",
                        "biblio.application_reference",
                        "biblio.priority_claims",
                        "jurisdiction",
                        "date_published",
                        "legal_status",
                    ],
                },
                timeout=20.0,
            )
            if response.status_code not in {200, 204, 404}:
                flags.append(missing("patent-lens-http-error", "patent_exclusivity", "candidates", f"Lens returned HTTP {response.status_code}.", "high"))
            elif response.status_code == 200:
                payload = response.json()
                for raw in _lens_records(payload):
                    candidate_id = str(raw.get("lens_id") or raw.get("doc_number") or "")
                    if not candidate_id:
                        continue
                    candidates.append(
                        PatentCandidate(
                            candidate_id=candidate_id,
                            title=_first_title(raw),
                            jurisdiction=first_text(raw.get("jurisdiction")),
                            publication_date=_first_date(raw.get("date_published") or _deep_find(raw, "date_published")),
                            legal_status=_legal_status(raw),
                            source_id=source.source_id,
                        )
                    )
            if not candidates:
                flags.append(missing("patent-no-candidates", "patent_exclusivity", "candidates", "No Lens patent candidates were retrieved.", "medium"))
        except Exception as exc:
            flags.append(missing("patent-lens-error", "patent_exclusivity", "candidates", f"Lens request failed: {exc.__class__.__name__}.", "high"))

    estimated_loe_year = loe_year_override
    if estimated_loe_year is None and candidates:
        estimated_loe_year = _estimate_loe_year(candidates)
    if estimated_loe_year is None:
        flags.append(missing("patent-loe-review-required", "patent_exclusivity", "estimated_loe_year", "No reviewed LOE year was supplied.", "high"))
    output_sources = []
    if token:
        output_sources.append(source)
    if loe_year_override is not None:
        output_sources.append(override_source)
    return (
        PatentExclusivityOutput(
            asset_name=asset.asset_name,
            searched_terms=terms,
            candidates=tuple(candidates),
            estimated_loe_year=estimated_loe_year,
            source_ids=tuple(item.source_id for item in output_sources),
            missing_data_flags=tuple(flags),
            confidence=0.75 if candidates and loe_year_override else 0.55 if candidates and estimated_loe_year else 0.2,
        ),
        tuple(output_sources),
    )


def _lens_records(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data") or payload.get("results") or []
    if isinstance(data, dict):
        data = data.get("data") or data.get("results") or []
    return [item for item in data if isinstance(item, dict)]


def _first_title(raw: dict[str, Any]) -> str | None:
    biblio = raw.get("biblio") if isinstance(raw.get("biblio"), dict) else {}
    title = raw.get("title") or biblio.get("invention_title")
    return first_text(title)


def _estimate_loe_year(candidates: list[PatentCandidate]) -> int | None:
    estimates = [_candidate_loe(candidate) for candidate in candidates]
    estimates = [estimate for estimate in estimates if estimate is not None]
    if not estimates:
        return None
    return max(year for year, _method in estimates)


def _candidate_loe(candidate: PatentCandidate) -> tuple[int, str] | None:
    current_year = datetime.now(timezone.utc).year
    legal_status = candidate.legal_status or ""
    term_date = _date_from_text(legal_status, ("anticipated term date", "term date", "adjusted expiration", "expiration date"))
    priority_date = _date_from_text(legal_status, ("priority date", "priority claim date", "earliest priority"))
    application_date = _date_from_text(legal_status, ("application filing date", "filing date", "application date"))
    publication_date = _parse_date(candidate.publication_date)
    if term_date:
        original_year = term_date.year
        method = "Lens anticipated term/expiration date"
    elif priority_date:
        original_year = priority_date.year + 20
        method = "Lens priority date + 20 years"
    elif application_date:
        original_year = application_date.year + 20
        method = "Lens application filing date + 20 years"
    elif publication_date:
        original_year = publication_date.year + 20
        method = "Lens publication date + 20 years"
    else:
        return None
    capped_year = min(original_year, current_year + 25)
    if priority_date:
        capped_year = min(capped_year, priority_date.year + 25)
    return capped_year, method


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
    priority = _first_date(raw.get("earliest_priority_claim_date") or _deep_find(raw, "priority_claims"))
    application = _first_date(raw.get("application_reference") or _deep_get(raw, "biblio.application_reference"))
    if priority and not any("priority" in part.casefold() for part in parts):
        parts.append(f"Priority Date: {priority}")
    if application and not any("application" in part.casefold() or "filing" in part.casefold() for part in parts):
        parts.append(f"Application Filing Date: {application}")
    return "; ".join(parts) or None


def _first_date(value: Any) -> str | None:
    for item in _string_values(value):
        parsed = _parse_date(item)
        if parsed:
            return parsed.isoformat()
    return None


def _date_from_text(text: str, labels: tuple[str, ...]) -> date | None:
    normalized = text.casefold()
    for label in labels:
        label_pattern = re.escape(label).replace("\\ ", r"\s+")
        match = re.search(rf"{label_pattern}\s*:?\s*((?:19|20)\d{{2}}(?:[-/]\d{{1,2}}(?:[-/]\d{{1,2}})?)?)", normalized)
        if match:
            parsed = _parse_date(match.group(1))
            if parsed:
                return parsed
    return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    text = value.strip()
    if match := re.search(r"(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?", text):
        text = match.group()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
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


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))
