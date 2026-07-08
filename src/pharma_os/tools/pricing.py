"""Pricing evidence from local WAC data and openFDA labels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from openpyxl import load_workbook

from pharma_os.schemas import AssetIdentityOutput, MissingDataFlag, PricingOutput, SourceMetadata
from pharma_os.tools._due_diligence_common import DEFAULT_WAC_DATA, OPENFDA_LABEL_URL, first_text, missing, norm, slug, to_float
from pharma_os.tools.rules import config_provenance, config_source, load_config


def lookup_pricing(
    asset: AssetIdentityOutput,
    *,
    wac_data_path: str | None = None,
    client: httpx.Client | None = None,
) -> tuple[PricingOutput, tuple[SourceMetadata, ...]]:
    """Match local WAC rows and openFDA dosing/label data."""

    wac_config = load_config("wac_sources.yaml", section="due_diligence")
    wac_config_source = config_source("wac_sources.yaml", section="due_diligence")
    configured_path = _configured_wac_path(wac_config)
    path = Path(wac_data_path or os.getenv("PHARMA_OS_WAC_DATA_PATH") or configured_path or DEFAULT_WAC_DATA)
    wac_source = SourceMetadata(
        source_id=f"wac:{slug(path.name)}",
        title="Local WAC workbook",
        provenance=f"Local WAC workbook combined_wac_increases lookup approved by {config_provenance('wac_sources.yaml', 'approved_sources[0]', section='due_diligence')}",
        source_type="wac_workbook",
        version=path.name,
    )
    label_source = SourceMetadata(
        source_id=f"openfda_label:{slug(asset.asset_name or asset.nct_id)}",
        title=f"openFDA label search for {asset.asset_name or asset.nct_id}",
        url=OPENFDA_LABEL_URL,
        provenance="openFDA drug label API",
        source_type="drug_label",
        version="openFDA",
    )
    flags: list[MissingDataFlag] = []
    wac_value = None
    matched_product = None
    if not path.exists():
        flags.append(missing("pricing-wac-file-missing", "pricing", "wac_value", f"WAC workbook not found: {path}", "high"))
    elif not asset.asset_name:
        flags.append(missing("pricing-asset-missing", "pricing", "matched_product", "No asset name available for WAC lookup.", "high"))
    else:
        try:
            terms = _pricing_terms(asset)
            wb = load_workbook(path, data_only=True, read_only=True)
            ws = wb["combined_wac_increases"]
            headers = [str(cell.value).strip() for cell in next(ws.iter_rows(min_row=1, max_row=1))]
            for row in ws.iter_rows(min_row=2, values_only=True):
                row_map = dict(zip(headers, row))
                description = str(row_map.get("Drug Product Description") or "")
                if any(norm(term) and norm(term) in norm(description) for term in terms):
                    wac_value = to_float(row_map.get("WAC After Increase"))
                    matched_product = description
                    break
            if wac_value is None:
                flags.append(missing("pricing-no-wac-match", "pricing", "wac_value", f"No WAC row matched {asset.asset_name}.", "high"))
        except Exception as exc:
            flags.append(missing("pricing-wac-error", "pricing", "wac_value", f"WAC lookup failed: {exc.__class__.__name__}.", "high"))

    dosing_summary = None
    terms = _pricing_terms(asset)
    if terms:
        try:
            label_payload = None
            for term in terms:
                response = (client or httpx.Client(timeout=20.0)).get(
                    OPENFDA_LABEL_URL,
                    params={"search": f'openfda.brand_name:"{term}"', "limit": "1"},
                    timeout=20.0,
                )
                if response.status_code == 200:
                    label_payload = response.json()
                    break
                response = (client or httpx.Client(timeout=20.0)).get(
                    OPENFDA_LABEL_URL,
                    params={"search": f'openfda.generic_name:"{term}"', "limit": "1"},
                    timeout=20.0,
                )
                if response.status_code == 200:
                    label_payload = response.json()
                    break
            if label_payload is not None:
                result = (label_payload.get("results") or [{}])[0]
                dosing_summary = first_text(result.get("dosage_and_administration"))
            else:
                flags.append(missing("pricing-no-openfda-label", "pricing", "dosing_summary", "openFDA returned no label match.", "medium"))
        except Exception as exc:
            flags.append(missing("pricing-openfda-error", "pricing", "dosing_summary", f"openFDA lookup failed: {exc.__class__.__name__}.", "medium"))
    annual_wac = wac_value
    if wac_value is not None and dosing_summary is None:
        flags.append(missing("pricing-dosing-review-required", "pricing", "annual_wac", "WAC was found but annualization lacks sourced dosing.", "high"))
    sources = tuple(source for source in (wac_config_source, wac_source if wac_value is not None else None, label_source if dosing_summary else None) if source)
    return (
        PricingOutput(
            annual_wac=annual_wac if dosing_summary else None,
            wac_value=wac_value,
            wac_unit_basis="package" if wac_value is not None else None,
            matched_product=matched_product,
            dosing_summary=dosing_summary,
            source_ids=tuple(source.source_id for source in sources),
            missing_data_flags=tuple(flags),
            confidence=0.8 if wac_value is not None and dosing_summary else 0.2,
        ),
        sources,
    )


def _pricing_terms(asset: AssetIdentityOutput) -> tuple[str, ...]:
    terms = [asset.asset_name, *asset.aliases]
    if asset.rxnorm_match:
        terms.extend([asset.rxnorm_match.matched_name, *asset.rxnorm_match.aliases])
    return tuple(dict.fromkeys(term for term in terms if term))


def _configured_wac_path(config: dict[str, Any]) -> str | None:
    sources = config.get("approved_sources")
    if not isinstance(sources, list):
        return None
    for source in sources:
        if isinstance(source, dict) and source.get("local_path"):
            return str(source["local_path"])
    return None
