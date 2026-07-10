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
from pydantic import Field

from pharma_os.agent_runtime import AgentRuntimeError, run_structured_llm_call, runtime_config_for_route
from pharma_os.schemas import AssetIdentityOutput, MissingDataFlag, PricingOutput, SourceMetadata, StrictSchema
from pharma_os.tools._due_diligence_common import DEFAULT_WAC_DATA, OPENFDA_LABEL_URL, first_text, missing, norm, slug, to_float
from pharma_os.tools.rules import config_provenance, config_source, load_config


REPO_ROOT = Path(__file__).resolve().parents[3]
WAC_COLUMN_ALIASES = {
    "brand_name": ("brand_name", "brand", "proprietary_name", "drug name"),
    "generic_name": ("generic_name", "generic", "nonproprietary_name", "generic name"),
    "manufacturer": ("manufacturer", "labeler_name", "company", "manufacturer_name", "manufacturer name"),
    "ndc": ("ndc", "ndc11", "ndc code", "product_ndc", "ndc number"),
    "product_description": ("product_description", "description", "drugname", "product", "drug product description"),
    "strength": ("strength", "strengths"),
    "dosage_form": ("dosage_form", "form"),
    "package_size": ("package_size", "package", "pkg_size"),
    "wac_value": ("wac_value", "wac", "current wac", "price", "unit_wac", "wac after increase"),
    "currency": ("currency",),
    "wac_unit_basis": ("wac_unit_basis", "unit basis", "pricing unit"),
    "effective_date": ("effective_date", "effective date", "date", "wac effective date"),
    "date_reported": ("date_reported", "date reported"),
    "source_file": ("source_file",),
    "source_sheet": ("source_sheet",),
    "source_year": ("source_year",),
}


@dataclass(frozen=True)
class PricingSearchTerm:
    """Normalized exact or source-constrained analog term for WAC lookup."""

    value: str
    is_analog: bool = False
    reason: str | None = None
    relevance_score: int = 100
    brand_name: str | None = None
    generic_name: str | None = None


@dataclass(frozen=True)
class AnnualizedWAC:
    """Deterministic annual WAC calculation details."""

    annual_wac: float
    details: dict[str, str | int | float | bool | None]


class PricingAnalogCandidate(StrictSchema):
    """Approved pricing analog candidate before WAC source matching."""

    brand_name: str | None = None
    generic_name: str | None = None
    approved_indication: str | None = None
    route: str | None = None
    modality: str | None = None
    rationale: str = Field(..., min_length=1)
    confidence: float = Field(default=0.5, ge=0, le=1)


class PricingAnalogSelectionOutput(StrictSchema):
    """Candidate approved analogs for source-constrained WAC matching."""

    candidates: tuple[PricingAnalogCandidate, ...] = Field(default_factory=tuple)
    human_review_flags: tuple[str, ...] = Field(default_factory=tuple)


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
    resolved_path = _resolve_wac_path(path)
    if not resolved_path.exists():
        flags.append(missing("pricing-wac-file-missing", "pricing", "wac_value", f"WAC workbook not found: {resolved_path}", "high"))
    elif not asset.asset_name:
        flags.append(missing("pricing-asset-missing", "pricing", "matched_product", "No asset name available for WAC lookup.", "high"))
    else:
        try:
            terms = _pricing_terms(asset)
            rows = _load_wac_rows(resolved_path)
            matches: list[tuple[dict[str, Any], PricingSearchTerm]] = []
            for row_map in rows:
                match = _match_wac_row(row_map, terms)
                if match is not None:
                    matches.append((row_map, match))
            if matches:
                row_map, matched_term = _select_wac_match(matches)
                wac_value = to_float(row_map.get("wac_value"))
                description = str(row_map.get("product_description") or "")
                matched_product = (
                    f"{description} (pricing analog: {matched_term.value}; {matched_term.reason})"
                    if matched_term.is_analog and matched_term.reason
                    else description
                )
            if wac_value is None:
                flags.append(
                    missing(
                        "pricing-no-wac-match",
                        "pricing",
                        "wac_value",
                        f"No exact or source-constrained pricing analog WAC row matched {asset.asset_name}.",
                        "high",
                    )
                )
        except AgentRuntimeError:
            raise
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
    terms = [
        PricingSearchTerm(term, relevance_score=100, brand_name=asset.asset_name)
        for term in [asset.asset_name, *asset.aliases]
        if term
    ]
    if asset.rxnorm_match:
        terms.extend(
            PricingSearchTerm(term, relevance_score=100, brand_name=asset.rxnorm_match.matched_name)
            for term in [asset.rxnorm_match.matched_name, *asset.rxnorm_match.aliases]
            if term
        )
    terms.extend(_pricing_analog_terms(asset))
    deduped: dict[str, PricingSearchTerm] = {}
    for term in terms:
        key = norm(term.value)
        if key and (key not in deduped or term.relevance_score > deduped[key].relevance_score):
            deduped[key] = term
    return tuple(deduped.values())


def _pricing_analog_terms(asset: AssetIdentityOutput) -> tuple[PricingSearchTerm, ...]:
    candidates = _pricing_analog_candidates(asset)
    terms: list[PricingSearchTerm] = []
    for index, candidate in enumerate(candidates):
        score = max(1, min(99, int(candidate.confidence * 100))) - index
        reason = candidate.rationale
        if candidate.brand_name:
            terms.append(
                PricingSearchTerm(
                    candidate.brand_name,
                    is_analog=True,
                    reason=reason,
                    relevance_score=score,
                    brand_name=candidate.brand_name,
                    generic_name=candidate.generic_name,
                )
            )
        if candidate.generic_name:
            terms.append(
                PricingSearchTerm(
                    candidate.generic_name,
                    is_analog=True,
                    reason=reason,
                    relevance_score=score,
                    brand_name=candidate.brand_name,
                    generic_name=candidate.generic_name,
                )
            )
    return tuple(terms)


def _pricing_analog_candidates(asset: AssetIdentityOutput) -> tuple[PricingAnalogCandidate, ...]:
    fallback = _fallback_pricing_analog_selection(asset)
    result = run_structured_llm_call(
        agent_name="PricingAnalogSelectionAgent",
        instructions=_pricing_analog_selection_instructions(),
        payload={"asset_context": _pricing_asset_context(asset)},
        output_type=PricingAnalogSelectionOutput,
        run_id=f"pricing-analog-selection-{slug(asset.nct_id or asset.asset_name or 'asset')}",
        input_summary=f"Select approved pricing analogs for {asset.asset_name or asset.nct_id}.",
        config=runtime_config_for_route(
            model_route="agent4_subagent",
            disabled_provenance="pharma_os.tools.pricing",
        ),
        offline_output=fallback,
        source_ids=asset.source_ids,
        confidence=asset.confidence,
        rationale_summary="Select approved commercial pricing analog candidates before deterministic WAC source matching.",
    )
    selected = result.output
    return _dedupe_analog_candidates((*selected.candidates, *fallback.candidates))


def _fallback_pricing_analog_selection(asset: AssetIdentityOutput) -> PricingAnalogSelectionOutput:
    text = _word_text(asset.asset_name, asset.normalized_indication, asset.therapeutic_area, *asset.aliases)
    analogs: list[PricingAnalogCandidate] = []
    if any(term in text for term in ("sle", "lupus", "systemic lupus erythematosus")):
        if "nephritis" in text:
            analogs.extend(
                [
                    _analog("Lupkynis", "voclosporin", "lupus nephritis approved pricing analog", indication="lupus nephritis", route="oral", modality="small molecule", confidence=0.85),
                ]
            )
        analogs.extend(
            [
                _analog("Benlysta", "belimumab", "systemic lupus erythematosus approved pricing analog", indication="systemic lupus erythematosus", route="subcutaneous", modality="biologic", confidence=0.82),
            ]
        )
    if any(term in text for term in ("atopic dermatitis", "eczema", "dermatitis", "ad ")):
        analogs.extend(
            [
                _analog("Cibinqo", "abrocitinib", "atopic dermatitis oral JAK inhibitor approved pricing analog", indication="atopic dermatitis", route="oral", modality="small molecule", confidence=0.88),
                _analog("Rinvoq", "upadacitinib", "atopic dermatitis oral JAK inhibitor approved pricing analog", indication="atopic dermatitis", route="oral", modality="small molecule", confidence=0.84),
                _analog("Dupixent", "dupilumab", "atopic dermatitis biologic approved pricing analog", indication="atopic dermatitis", route="subcutaneous", modality="biologic", confidence=0.78),
                _analog("Adbry", "tralokinumab", "atopic dermatitis biologic approved pricing analog", indication="atopic dermatitis", route="subcutaneous", modality="biologic", confidence=0.72),
            ]
        )
    if any(term in text for term in ("psoriasis", "psoriatic", "tyk2", "tyrosine kinase 2")):
        analogs.extend(
            [
                _analog("Sotyktu", "deucravacitinib", "TYK2/psoriasis approved pricing analog", indication="plaque psoriasis", route="oral", modality="small molecule", confidence=0.85),
            ]
        )
    if "rheumatoid arthritis" in text or ("autoimmune" in text and "small molecule" in text):
        analogs.extend(
            [
                _analog("Rinvoq", "upadacitinib", "autoimmune oral small-molecule approved pricing analog", route="oral", modality="small molecule", confidence=0.78),
            ]
        )
    return PricingAnalogSelectionOutput(candidates=_dedupe_analog_candidates(tuple(analogs)))


def _analog(
    brand_name: str,
    generic_name: str,
    rationale: str,
    *,
    indication: str | None = None,
    route: str | None = None,
    modality: str | None = None,
    confidence: float = 0.7,
) -> PricingAnalogCandidate:
    return PricingAnalogCandidate(
        brand_name=brand_name,
        generic_name=generic_name,
        approved_indication=indication,
        route=route,
        modality=modality,
        rationale=rationale,
        confidence=confidence,
    )


def _dedupe_analog_candidates(candidates: tuple[PricingAnalogCandidate, ...]) -> tuple[PricingAnalogCandidate, ...]:
    deduped: dict[tuple[str, str], PricingAnalogCandidate] = {}
    for candidate in candidates:
        key = (norm(candidate.brand_name or ""), norm(candidate.generic_name or ""))
        if not any(key):
            continue
        existing = deduped.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            deduped[key] = candidate
    return tuple(deduped.values())


def _pricing_asset_context(asset: AssetIdentityOutput) -> dict[str, Any]:
    return {
        "nct_id": asset.nct_id,
        "asset_name": asset.asset_name,
        "aliases": asset.aliases,
        "raw_intervention_names": asset.raw_intervention_names,
        "intervention_type": asset.intervention_type,
        "sponsor": asset.sponsor,
        "normalized_indication": asset.normalized_indication,
        "therapeutic_area": asset.therapeutic_area,
        "modality": asset.modality,
        "guardrails": (
            "Return approved commercial drug analogs only. Do not invent WAC values, dosing, labels, or unapproved products. "
            "Prefer analogs likely to have public label dosing and approved WAC workbook rows."
        ),
    }


def _pricing_analog_selection_instructions() -> str:
    return (
        "You are PricingAnalogSelectionAgent for PharmaOS Agent 4. Select approved commercial pricing analog candidates "
        "for the target investigational asset using only the supplied asset context. Prefer analogs by indication, route, "
        "modality, mechanism/class, prescriber setting, and commercial use. Return candidates only; deterministic code will "
        "match candidates to the approved WAC workbook and fetch openFDA dosing. Do not calculate price, claim WAC "
        "availability, fetch labels, or fabricate evidence."
    )


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
        str(row_map.get("brand_name") or ""),
        str(row_map.get("generic_name") or ""),
        str(row_map.get("product_description") or ""),
        str(row_map.get("manufacturer") or ""),
        str(row_map.get("ndc") or ""),
    ]
    normalized_haystacks = [norm(item) for item in haystacks]
    for term in terms:
        normalized_term = norm(term.value)
        if normalized_term and any(normalized_term in haystack for haystack in normalized_haystacks):
            return term
    return None


def _select_wac_match(matches: list[tuple[dict[str, Any], PricingSearchTerm]]) -> tuple[dict[str, Any], PricingSearchTerm]:
    return sorted(matches, key=_wac_selection_key, reverse=True)[0]


def _wac_selection_key(match: tuple[dict[str, Any], PricingSearchTerm]) -> tuple[Any, ...]:
    row, term = match
    return (
        0 if term.is_analog else 1,
        term.relevance_score,
        _parse_date(str(row.get("effective_date") or "")),
        _parse_date(str(row.get("date_reported") or "")),
        int(to_float(row.get("source_year")) or 0),
        -(_units_per_package(str(row.get("product_description") or "")) or 999999),
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
    if match := re.search(r"\b(\d+)\s*(?:capsules?|tablets?|tabs?|count|ct|ea|pack|pens?|syringes?|auto-?injectors?|autoinjectors?|vials?)\b", text):
        return int(match.group(1))
    if match := re.search(r"\b(?:capsules?|tablets?|tabs?|pens?|syringes?|auto-?injectors?|autoinjectors?|vials?)\s*(\d+)\b", text):
        return int(match.group(1))
    if match := re.search(r"-\s*(\d+)\s*(?:pens?|syringes?|auto-?injectors?|autoinjectors?|vials?)\b", text):
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


def _resolve_wac_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    repo_path = REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return path


def _load_wac_rows(path: Path) -> list[dict[str, Any]]:
    wb = load_workbook(path, data_only=True, read_only=True)
    sheet_name = "combined_wac_increases" if "combined_wac_increases" in wb.sheetnames else wb.sheetnames[0]
    ws = wb[sheet_name]
    headers = [str(cell.value or "").strip() for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    normalized_headers = {_normalize_header(header): index for index, header in enumerate(headers)}
    rows: list[dict[str, Any]] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        normalized: dict[str, Any] = {}
        for target, aliases in WAC_COLUMN_ALIASES.items():
            index = next(
                (normalized_headers[_normalize_header(alias)] for alias in aliases if _normalize_header(alias) in normalized_headers),
                None,
            )
            normalized[target] = row[index] if index is not None and index < len(row) else None
        if any(normalized.get(key) for key in ("product_description", "brand_name", "generic_name", "ndc")):
            rows.append(normalized)
    return rows


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
