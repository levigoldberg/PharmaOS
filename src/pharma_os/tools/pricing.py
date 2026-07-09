"""Pricing evidence from local WAC data and openFDA labels."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from openpyxl import load_workbook

from pharma_os.schemas import AssetIdentityOutput, MissingDataFlag, PricingOutput, SourceMetadata
from pharma_os.tools._due_diligence_common import DEFAULT_WAC_DATA, OPENFDA_LABEL_URL, first_text, missing, norm, slug, to_float
from pharma_os.tools.rules import config_provenance, config_source, load_config


@dataclass(frozen=True)
class PricingSearchTerm:
    """Normalized exact or source-constrained analog term for WAC lookup."""

    value: str
    is_analog: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class AnnualizedWAC:
    """Deterministic annual WAC calculation details."""

    annual_wac: float
    details: dict[str, str | int | float | bool | None]


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
    flags: list[MissingDataFlag] = []
    wac_value = None
    matched_product = None
    matched_term: PricingSearchTerm | None = None
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
            matches: list[tuple[dict[str, Any], PricingSearchTerm]] = []
            for row in ws.iter_rows(min_row=2, values_only=True):
                row_map = dict(zip(headers, row))
                match = _match_wac_row(row_map, terms)
                if match is not None:
                    matches.append((row_map, match))
            if matches:
                row_map, matched_term = _select_wac_match(matches)
                wac_value = to_float(row_map.get("WAC After Increase"))
                description = str(row_map.get("Drug Product Description") or "")
                matched_product = (
                    f"{description} (pricing analog: {matched_term.value}; {matched_term.reason})"
                    if matched_term.is_analog and matched_term.reason
                    else description
                )
            if wac_value is None:
                flags.append(missing("pricing-no-wac-match", "pricing", "wac_value", f"No WAC row matched {asset.asset_name}.", "high"))
        except Exception as exc:
            flags.append(missing("pricing-wac-error", "pricing", "wac_value", f"WAC lookup failed: {exc.__class__.__name__}.", "high"))

    dosing_summary = None
    selected_label_term = None
    terms = _label_terms(asset, matched_term)
    if terms:
        http_client = client or httpx.Client(timeout=20.0)
        try:
            label_payload: dict[str, Any] | None = None
            for term in terms:
                response = http_client.get(
                    OPENFDA_LABEL_URL,
                    params={"search": f'openfda.brand_name:"{term.value}"', "limit": "1"},
                    timeout=20.0,
                )
                if response.status_code == 200:
                    label_payload = response.json()
                    selected_label_term = term.value
                    break
                response = http_client.get(
                    OPENFDA_LABEL_URL,
                    params={"search": f'openfda.generic_name:"{term.value}"', "limit": "1"},
                    timeout=20.0,
                )
                if response.status_code == 200:
                    label_payload = response.json()
                    selected_label_term = term.value
                    break
            if label_payload is not None:
                result = (label_payload.get("results") or [{}])[0]
                dosing_summary = first_text(result.get("dosage_and_administration"))
            else:
                flags.append(missing("pricing-no-openfda-label", "pricing", "dosing_summary", "openFDA returned no label match.", "medium"))
        except Exception as exc:
            flags.append(missing("pricing-openfda-error", "pricing", "dosing_summary", f"openFDA lookup failed: {exc.__class__.__name__}.", "medium"))
    annualized = _annualize_wac(wac_value, matched_product, dosing_summary)
    annual_wac = annualized.annual_wac if annualized is not None else None
    annualization_details = annualized.details if annualized is not None else {}
    if wac_value is not None and dosing_summary is None:
        flags.append(missing("pricing-dosing-review-required", "pricing", "annual_wac", "WAC was found but annualization lacks sourced dosing.", "high"))
    elif wac_value is not None and annual_wac is None:
        flags.append(missing("pricing-annualization-review-required", "pricing", "annual_wac", "WAC and dosing were found, but package/frequency annualization requires review.", "high"))
    elif annualized is not None and not annualized.details.get("formula_check_passed"):
        flags.append(missing("pricing-annualization-formula-check", "pricing", "annual_wac", "Annualized WAC failed internal formula validation.", "critical"))
    label_source = (
        SourceMetadata(
            source_id=f"openfda_label:{slug(selected_label_term)}",
            title=f"openFDA label search for {selected_label_term}",
            url=OPENFDA_LABEL_URL,
            provenance="openFDA drug label API",
            source_type="drug_label",
            version="openFDA",
        )
        if selected_label_term and dosing_summary
        else None
    )
    sources = tuple(source for source in (wac_config_source, wac_source if wac_value is not None else None, label_source) if source)
    return (
        PricingOutput(
            annual_wac=annual_wac,
            wac_value=wac_value,
            wac_unit_basis="package" if wac_value is not None else None,
            matched_product=matched_product,
            dosing_summary=dosing_summary,
            annualization_details=annualization_details,
            source_ids=tuple(source.source_id for source in sources),
            missing_data_flags=tuple(flags),
            confidence=0.8 if annual_wac is not None else 0.45 if wac_value is not None else 0.2,
        ),
        sources,
    )


def _pricing_terms(asset: AssetIdentityOutput) -> tuple[PricingSearchTerm, ...]:
    terms = [PricingSearchTerm(term) for term in [asset.asset_name, *asset.aliases] if term]
    if asset.rxnorm_match:
        terms.extend(PricingSearchTerm(term) for term in [asset.rxnorm_match.matched_name, *asset.rxnorm_match.aliases] if term)
    terms.extend(_pricing_analog_terms(asset))
    deduped: dict[str, PricingSearchTerm] = {}
    for term in terms:
        key = norm(term.value)
        if key and key not in deduped:
            deduped[key] = term
    return tuple(deduped.values())


def _pricing_analog_terms(asset: AssetIdentityOutput) -> tuple[PricingSearchTerm, ...]:
    text = _word_text(asset.asset_name, asset.normalized_indication, asset.therapeutic_area, *asset.aliases)
    analogs: list[PricingSearchTerm] = []
    if any(term in text for term in ("sle", "lupus", "systemic lupus erythematosus")):
        if "nephritis" in text:
            analogs.extend(
                [
                    PricingSearchTerm("Lupkynis", is_analog=True, reason="lupus nephritis approved pricing analog"),
                    PricingSearchTerm("voclosporin", is_analog=True, reason="lupus nephritis approved pricing analog"),
                ]
            )
        analogs.extend(
            [
                PricingSearchTerm("Benlysta", is_analog=True, reason="systemic lupus erythematosus approved pricing analog"),
                PricingSearchTerm("belimumab", is_analog=True, reason="systemic lupus erythematosus approved pricing analog"),
            ]
        )
    if any(term in text for term in ("psoriasis", "psoriatic", "tyk2", "tyrosine kinase 2")):
        analogs.extend(
            [
                PricingSearchTerm("Sotyktu", is_analog=True, reason="TYK2/psoriasis approved pricing analog"),
                PricingSearchTerm("deucravacitinib", is_analog=True, reason="TYK2/psoriasis approved pricing analog"),
            ]
        )
    if "rheumatoid arthritis" in text or ("autoimmune" in text and "small molecule" in text):
        analogs.extend(
            [
                PricingSearchTerm("Rinvoq", is_analog=True, reason="autoimmune oral small-molecule approved pricing analog"),
                PricingSearchTerm("upadacitinib", is_analog=True, reason="autoimmune oral small-molecule approved pricing analog"),
            ]
        )
    return tuple(analogs)


def _label_terms(asset: AssetIdentityOutput, matched_term: PricingSearchTerm | None) -> tuple[PricingSearchTerm, ...]:
    terms: list[PricingSearchTerm] = []
    if matched_term is not None:
        terms.append(matched_term)
        if matched_term.is_analog:
            terms.extend(term for term in _pricing_analog_terms(asset) if term.reason == matched_term.reason)
    else:
        terms.extend(_pricing_terms(asset))
    deduped: dict[str, PricingSearchTerm] = {}
    for term in terms:
        key = norm(term.value)
        if key and key not in deduped:
            deduped[key] = term
    return tuple(deduped.values())


def _match_wac_row(row_map: dict[str, Any], terms: tuple[PricingSearchTerm, ...]) -> PricingSearchTerm | None:
    haystacks = [
        str(row_map.get("Drug Product Description") or ""),
        str(row_map.get("Manufacturer Name") or ""),
        str(row_map.get("NDC Number") or ""),
    ]
    normalized_haystacks = [norm(item) for item in haystacks]
    for term in terms:
        normalized_term = norm(term.value)
        if normalized_term and any(normalized_term in haystack for haystack in normalized_haystacks):
            return term
    return None


def _select_wac_match(matches: list[tuple[dict[str, Any], PricingSearchTerm]]) -> tuple[dict[str, Any], PricingSearchTerm]:
    return sorted(matches, key=_wac_selection_key, reverse=True)[0]


def _wac_selection_key(match: tuple[dict[str, Any], PricingSearchTerm]) -> tuple[datetime, int, int]:
    row, term = match
    return (
        _parse_date(str(row.get("WAC Effective Date") or "")),
        0 if term.is_analog else 1,
        -(_units_per_package(str(row.get("Drug Product Description") or "")) or 999999),
    )


def _annualize_wac(wac_value: float | None, matched_product: str | None, dosing_summary: str | None) -> AnnualizedWAC | None:
    if wac_value is None or not matched_product or not dosing_summary:
        return None
    administrations_per_year = _administrations_per_year(dosing_summary)
    if administrations_per_year is None:
        return None
    units_per_package = _units_per_package(matched_product)
    if not units_per_package:
        return None
    units_per_administration = _units_per_administration(matched_product, dosing_summary)
    if units_per_administration is None:
        return None
    package_units_per_year = administrations_per_year * units_per_administration
    packages_per_year = package_units_per_year / units_per_package
    annual_wac = round(float(wac_value) * packages_per_year, 2)
    expected = round(float(wac_value) * administrations_per_year * units_per_administration / units_per_package, 2)
    details: dict[str, str | int | float | bool | None] = {
        "formula": "annual_wac = wac_value * administrations_per_year * units_per_administration / units_per_package",
        "wac_value": float(wac_value),
        "wac_unit_basis": "package",
        "administrations_per_year": administrations_per_year,
        "units_per_administration": units_per_administration,
        "units_per_package": units_per_package,
        "package_units_per_year": round(package_units_per_year, 4),
        "packages_per_year": round(packages_per_year, 4),
        "annual_wac": annual_wac,
        "formula_check_expected_annual_wac": expected,
        "formula_check_passed": abs(annual_wac - expected) <= 0.01,
    }
    return AnnualizedWAC(annual_wac=annual_wac, details=details)


def _administrations_per_year(dosing_summary: str) -> float | None:
    text = _word_text(dosing_summary)
    if "twice weekly" in text or "two times weekly" in text:
        return 104.0
    if any(term in text for term in ("once weekly", "once a week", "every week", "weekly")):
        return 52.0
    if any(term in text for term in ("every 2 weeks", "every two weeks", "every other week", "once every 2 weeks")):
        return 26.0
    if any(term in text for term in ("every 4 weeks", "monthly", "once monthly", "once every month")):
        return 13.0
    if any(term in text for term in ("three times daily", "three times a day", "thrice daily")):
        return 1095.0
    if any(term in text for term in ("twice daily", "twice a day", "two times daily")):
        return 730.0
    if any(term in text for term in ("once daily", "once a day", "daily")):
        return 365.0
    return None


def _units_per_administration(product_description: str, dosing_summary: str) -> float | None:
    product = _word_text(product_description)
    if any(term in product for term in ("auto injector", "autoinjector", "prefilled syringe", "syringe", "vial")):
        return 1.0
    strength = _strength_mg(product_description)
    dose = _dose_mg(dosing_summary)
    if strength and dose:
        return max(1.0, dose / strength)
    if any(term in product for term in ("tablet", "capsule", "capsules", "tab", "bottle")):
        return 1.0
    return None


def _units_per_package(product_description: str) -> int | None:
    text = product_description.casefold()
    if match := re.search(r"\bx\s*(\d+)\b", text):
        return int(match.group(1))
    if match := re.search(r"\b(\d+)\s*(?:capsules?|tablets?|tabs?|count|ct|ea|pack)\b", text):
        return int(match.group(1))
    if match := re.search(r"\b(\d+)\s*day\s*bottle\b", text):
        return int(match.group(1))
    if any(term in text for term in ("auto-injector", "autoinjector", "prefilled syringe", "syringe", "vial")):
        return 1
    return None


def _strength_mg(product_description: str) -> float | None:
    if match := re.search(r"\b(\d+(?:\.\d+)?)\s*mg\b", product_description.casefold()):
        return float(match.group(1))
    return None


def _dose_mg(dosing_summary: str) -> float | None:
    if match := re.search(r"\b(\d+(?:\.\d+)?)\s*mg\b", dosing_summary.casefold()):
        return float(match.group(1))
    return None


def _parse_date(value: str) -> datetime:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return datetime.min


def _word_text(*values: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", " ".join(value or "" for value in values).casefold())).strip()


def _configured_wac_path(config: dict[str, Any]) -> str | None:
    sources = config.get("approved_sources")
    if not isinstance(sources, list):
        return None
    for source in sources:
        if isinstance(source, dict) and source.get("local_path"):
            return str(source["local_path"])
    return None
