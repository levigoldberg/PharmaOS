"""Shared helpers for deterministic due-diligence tools."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from pharma_os.schemas import AssumptionRecord, MissingDataFlag
from pharma_os.tools.rules import config_provenance, config_source_id


DEFAULT_POS_WORKBOOK = Path("data/Source_Based_PoS_Workbook.xlsx")
DEFAULT_WAC_DATA = Path("data/california_wac_data.xlsx")
OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
LENS_PATENT_URL = "https://api.lens.org/patent/search"


def missing(flag_id: str, section: str, field: str, reason: str, severity: str) -> MissingDataFlag:
    """Create a typed missing-data flag."""

    return MissingDataFlag(flag_id=flag_id, section=section, field=field, reason=reason, severity=severity)  # type: ignore[arg-type]


def select_assumption_value(
    assumption_id: str,
    name: str,
    user_value: Any,
    unit: str,
    *,
    config_value: Any,
    config_filename: str,
    config_field_path: str,
) -> AssumptionRecord:
    """Select user-reviewed value, config fallback, or typed missing assumption."""

    if user_value is not None:
        return assumption(
            assumption_id,
            name,
            user_value,
            unit,
            "cli.due_diligence",
            assumption_type="user_reviewed",
        )
    if config_value is not None:
        return assumption(
            assumption_id,
            name,
            config_value,
            unit,
            config_provenance(config_filename, config_field_path, section="due_diligence"),
            assumption_type="fallback_assumption",
            source_ids=(config_source_id(config_filename, section="due_diligence"),),
        )
    return assumption(
        assumption_id,
        name,
        None,
        unit,
        f"missing:{config_filename}:{config_field_path}",
        assumption_type="missing",
        requires_human_review=True,
    )


def assumption(
    assumption_id: str,
    name: str,
    value: Any,
    unit: str,
    provenance: str,
    *,
    assumption_type: str,
    source_ids: tuple[str, ...] = (),
    requires_human_review: bool | None = None,
) -> AssumptionRecord:
    """Create a typed assumption record."""

    return AssumptionRecord(
        assumption_id=assumption_id,
        name=name,
        value=value,
        unit=unit,
        assumption_type=assumption_type,  # type: ignore[arg-type]
        source_ids=source_ids,
        provenance=provenance,
        requires_human_review=value is None if requires_human_review is None else requires_human_review,
    )


def triplet_base(value: Any) -> float | None:
    """Extract a base value from a low/base/high config object."""

    return to_float(value.get("base")) if isinstance(value, dict) else None


def launch_year_from_config(config: dict[str, Any], phase: str | None) -> int | None:
    """Infer launch year from rNPV timing config."""

    valuation_year = to_float(config.get("valuation_year"))
    timing = ((config.get("launch_timing") or {}).get("default_years_to_launch_by_phase") or {})
    years = to_float(timing.get(phase or "") or timing.get("default"))
    if valuation_year is None or years is None:
        return None
    return int(valuation_year + years)


def development_cost_from_config(config: dict[str, Any], phase: str | None) -> float | None:
    """Infer development cost from rNPV phase config."""

    by_phase = ((config.get("development_costs") or {}).get("by_phase") or {})
    selected = by_phase.get(phase or "") or by_phase.get("default")
    if not isinstance(selected, dict):
        return None
    return to_float(selected.get("total_cost"))


def first_text(value: Any) -> str | None:
    """Return first non-empty text value from nested API fields."""

    if value is None:
        return None
    if isinstance(value, list):
        return first_text(value[0]) if value else None
    if isinstance(value, dict):
        for item in value.values():
            text = first_text(item)
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def to_float(value: Any) -> float | None:
    """Convert a scalar into float, preserving percentage-string behavior."""

    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number / 100 if number > 1 and number <= 100 and "%" in str(value) else number


def json_scalar(value: Any) -> str | int | float | bool | None:
    """Normalize workbook cell values into JSON-safe scalars."""

    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    return str(value)


def norm(value: str) -> str:
    """Normalize text for loose matching."""

    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def slug(value: str) -> str:
    """Create a stable source-id slug."""

    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "unknown"
