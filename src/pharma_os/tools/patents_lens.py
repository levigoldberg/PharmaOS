"""Lens-only patent and LOE due-diligence tool."""

from __future__ import annotations

import os
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
                    "size": 5,
                    "from": 0,
                    "include": ["lens_id", "biblio.invention_title", "jurisdiction", "date_published", "legal_status"],
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
                            publication_date=first_text(raw.get("date_published")),
                            legal_status=first_text(raw.get("legal_status")),
                            source_id=source.source_id,
                        )
                    )
            if not candidates:
                flags.append(missing("patent-no-candidates", "patent_exclusivity", "candidates", "No Lens patent candidates were retrieved.", "medium"))
        except Exception as exc:
            flags.append(missing("patent-lens-error", "patent_exclusivity", "candidates", f"Lens request failed: {exc.__class__.__name__}.", "high"))

    if loe_year_override is None:
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
            estimated_loe_year=loe_year_override,
            source_ids=tuple(item.source_id for item in output_sources),
            missing_data_flags=tuple(flags),
            confidence=0.75 if candidates and loe_year_override else 0.2,
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
