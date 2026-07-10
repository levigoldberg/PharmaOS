"""Panoptic-style due-diligence report payload assembly."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


NOT_AVAILABLE = "Not available"


def build_due_diligence_report_payload(output: Any) -> dict[str, Any]:
    """Build a compact final-report layer from a due-diligence output payload."""

    raw = _jsonable(output)
    commercial = _dict(raw.get("commercial_model"))
    rnpv = _dict(raw.get("rnpv"))
    pricing = _dict(raw.get("pricing"))
    forecast = _commercial_forecast(commercial, rnpv)
    sensitivity = _sensitivity_summary(commercial, rnpv)
    chart_specs = _chart_specs(commercial, forecast, sensitivity)
    flags = _top_confidence_flags(raw)
    return {
        "investment_snapshot": _investment_snapshot(raw),
        "market_conversion_assumptions": _market_conversion_assumptions(commercial),
        "pricing_source_logic": _pricing_source_logic(pricing, commercial),
        "commercial_forecast": forecast,
        "rnpv_summary": _rnpv_summary(rnpv, commercial),
        "sensitivity_summary": sensitivity,
        "top_confidence_flags": flags,
        "chart_specs": chart_specs,
    }


def _investment_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    asset = _dict(raw.get("asset_identity"))
    trial = _dict(raw.get("target_trial") or raw.get("trial"))
    commercial = _dict(raw.get("commercial_model"))
    rnpv = _dict(raw.get("rnpv"))
    pricing = _dict(raw.get("pricing"))
    pos = _dict(raw.get("pos"))
    funnel = _dict(commercial.get("patient_funnel"))
    return {
        "nct_id": _available(trial.get("nct_id")),
        "drug": _available(asset.get("asset_name") or trial.get("nct_id")),
        "sponsor": _available(asset.get("sponsor") or _value(trial, "lead_sponsor", "name")),
        "indication": _available(asset.get("normalized_indication") or ", ".join(_list(trial.get("conditions")))),
        "phase": _available(", ".join(_list(trial.get("phases")))),
        "trial_status": _available(trial.get("overall_status")),
        "disease_population": _available(_value(commercial, "selected_population_measure", "value")),
        "eligible_patients": _available(funnel.get("eligible_patients")),
        "commercially_addressable_patients": _available(funnel.get("commercially_addressable_patients")),
        "estimated_annual_net_price": _available(commercial.get("net_price")),
        "peak_sales": _available(commercial.get("peak_net_sales")),
        "launch_year": _available(rnpv.get("launch_year")),
        "estimated_loe_year": _available(rnpv.get("loe_year")),
        "phase_to_approval_pos": _available(pos.get("probability_of_success") or rnpv.get("probability_of_success")),
        "base_case_rnpv": _available(rnpv.get("rnpv")),
        "overall_confidence": _confidence_label(raw.get("confidence")),
        "primary_pricing_analog": _available(pricing.get("matched_product")),
    }


def _pricing_source_logic(pricing: dict[str, Any], commercial: dict[str, Any]) -> dict[str, Any]:
    details = _dict(pricing.get("annualization_details"))
    return {
        "pricing_basis": _available(pricing.get("matched_product")),
        "selected_wac_row": _available(pricing.get("matched_product")),
        "wac_per_package": _available(pricing.get("wac_value")),
        "wac_unit_basis": _available(pricing.get("wac_unit_basis")),
        "annual_gross_wac": _available(pricing.get("annual_wac")),
        "annualization_formula": _available(details.get("formula")),
        "gross_to_net": _available(commercial.get("gross_to_net")),
        "annual_net_price": _available(commercial.get("net_price")),
        "human_review_required": True,
    }


def _market_conversion_assumptions(commercial: dict[str, Any]) -> list[dict[str, Any]]:
    funnel = _dict(commercial.get("patient_funnel"))
    rows = [
        ("diagnosed fraction", "disease population -> diagnosed patients", funnel.get("diagnosed_fraction"), funnel.get("diagnosed_patients")),
        ("treated/managed fraction", "diagnosed patients -> treated/managed patients", funnel.get("treated_fraction"), funnel.get("treated_or_managed_patients")),
        ("eligibility fraction", "treated/managed patients -> eligible patients", funnel.get("eligibility_fraction"), funnel.get("eligible_patients")),
        ("commercially addressable fraction", "eligible patients -> commercially addressable patients", funnel.get("commercially_addressable_fraction"), funnel.get("commercially_addressable_patients")),
        ("peak treated fraction", "commercially addressable patients -> peak treated patients", commercial.get("peak_penetration"), _peak_treated(commercial)),
    ]
    return [
        {
            "conversion": label,
            "step": step,
            "base_fraction": _available(_round(fraction)),
            "resulting_patients": _available(_round(patients)),
            "source": "commercial_model.patient_funnel",
            "human_review_required": True,
        }
        for label, step, fraction, patients in rows
        if fraction not in (None, NOT_AVAILABLE) or patients not in (None, NOT_AVAILABLE)
    ]


def _commercial_forecast(commercial: dict[str, Any], rnpv: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    peak_penetration = _number(commercial.get("peak_penetration"))
    annual_patients = _number(commercial.get("annual_patients"))
    pos = _number(rnpv.get("probability_of_success"))
    launch_year = _number(rnpv.get("launch_year"))
    discount = _number(rnpv.get("discount_rate")) or 0.0
    valuation_year = _assumption_value(rnpv, "valuation_year")
    valuation = int(_number(valuation_year) or 0)
    for row in _list(commercial.get("revenue_forecast")):
        if not isinstance(row, dict):
            continue
        year = int(_number(row.get("year")) or 0)
        treated = _number(row.get("treated_patients"))
        revenue = _number(row.get("net_revenue"))
        actual_penetration = treated / annual_patients if treated is not None and annual_patients else None
        ramp = actual_penetration / peak_penetration if actual_penetration is not None and peak_penetration else None
        risk_adjusted = revenue * pos if revenue is not None and pos is not None else None
        calendar_year = int(launch_year + year - 1) if launch_year is not None and year else None
        years = max(0, calendar_year - valuation) if calendar_year and valuation else 0
        discounted = risk_adjusted / ((1 + discount) ** years) if risk_adjusted is not None and discount > -1 else risk_adjusted
        rows.append(
            {
                "year": year,
                "calendar_year": _available(calendar_year),
                "commercial_patient_base": _available(annual_patients),
                "peak_penetration": _available(peak_penetration),
                "ramp_to_peak": _available(_round(ramp)),
                "actual_penetration": _available(_round(actual_penetration)),
                "treated_patients": _available(treated),
                "net_price": _available(row.get("net_price")),
                "revenue": _available(revenue),
                "risk_adjusted_revenue": _available(_round_money(risk_adjusted)),
                "discounted_risk_adjusted_revenue": _available(_round_money(discounted)),
            }
        )
    return rows


def _rnpv_summary(rnpv: dict[str, Any], commercial: dict[str, Any]) -> dict[str, Any]:
    return {
        "formula_text": "Treated patients = commercial patient base x actual penetration. Net price = annual gross WAC x (1 - gross-to-net). Revenue = treated patients x net price. Risk-adjusted revenue = revenue x PoS. rNPV = discounted risk-adjusted revenue through LOE less development cost.",
        "base_case_rnpv": _available(rnpv.get("rnpv")),
        "launch_year": _available(rnpv.get("launch_year")),
        "loe_year": _available(rnpv.get("loe_year")),
        "discount_rate": _available(rnpv.get("discount_rate")),
        "pos": _available(rnpv.get("probability_of_success")),
        "operating_margin": _available(rnpv.get("operating_margin")),
        "development_cost": _available(rnpv.get("development_cost")),
        "commercial_calculable": _available(commercial.get("calculable")),
        "rnpv_calculable": _available(rnpv.get("calculable")),
        "interpretation": _available(_rnpv_interpretation(rnpv)),
    }


def _sensitivity_summary(commercial: dict[str, Any], rnpv: dict[str, Any]) -> list[dict[str, Any]]:
    if not _list(commercial.get("revenue_forecast")) or _number(rnpv.get("probability_of_success")) is None:
        return []
    base = _calculate_rnpv(_list(commercial.get("revenue_forecast")), rnpv)
    if base is None:
        return []
    rows: list[dict[str, Any]] = []
    cases = _dict(commercial.get("cases"))
    downside = _dict(cases.get("downside"))
    upside = _dict(cases.get("upside"))
    if downside.get("revenue_forecast") and upside.get("revenue_forecast"):
        rows.append(_sensitivity_row("net price / peak sales", "downside case", "upside case", _calculate_rnpv(_list(downside.get("revenue_forecast")), rnpv), base, _calculate_rnpv(_list(upside.get("revenue_forecast")), rnpv)))
    pos = _number(rnpv.get("probability_of_success"))
    if pos is not None:
        rows.append(_sensitivity_row("PoS", max(0.0, pos * 0.75), min(1.0, pos * 1.25), _calculate_rnpv(_list(commercial.get("revenue_forecast")), {**rnpv, "probability_of_success": max(0.0, pos * 0.75)}), base, _calculate_rnpv(_list(commercial.get("revenue_forecast")), {**rnpv, "probability_of_success": min(1.0, pos * 1.25)})))
    discount = _number(rnpv.get("discount_rate"))
    if discount is not None:
        rows.append(_sensitivity_row("discount rate", discount + 0.02, max(0.0, discount - 0.02), _calculate_rnpv(_list(commercial.get("revenue_forecast")), {**rnpv, "discount_rate": discount + 0.02}), base, _calculate_rnpv(_list(commercial.get("revenue_forecast")), {**rnpv, "discount_rate": max(0.0, discount - 0.02)})))
    margin = _number(rnpv.get("operating_margin"))
    if margin is not None:
        rows.append(_sensitivity_row("operating margin", max(0.0, margin - 0.10), min(1.0, margin + 0.10), _calculate_rnpv(_list(commercial.get("revenue_forecast")), {**rnpv, "operating_margin": max(0.0, margin - 0.10)}), base, _calculate_rnpv(_list(commercial.get("revenue_forecast")), {**rnpv, "operating_margin": min(1.0, margin + 0.10)})))
    cost = _number(rnpv.get("development_cost"))
    if cost is not None:
        rows.append(_sensitivity_row("development cost", cost * 1.25, cost * 0.75, _calculate_rnpv(_list(commercial.get("revenue_forecast")), {**rnpv, "development_cost": cost * 1.25}), base, _calculate_rnpv(_list(commercial.get("revenue_forecast")), {**rnpv, "development_cost": cost * 0.75})))
    return [row for row in rows if row]


def _sensitivity_row(variable: str, low_input: Any, high_input: Any, low: float | None, base: float, high: float | None) -> dict[str, Any]:
    if low is None or high is None:
        return {}
    return {
        "variable": variable,
        "low_input": _available(_round_for_display(low_input)),
        "base_input": NOT_AVAILABLE,
        "high_input": _available(_round_for_display(high_input)),
        "low_case_rnpv": _available(_round_money(low)),
        "base_case_rnpv": _available(_round_money(base)),
        "high_case_rnpv": _available(_round_money(high)),
    }


def _chart_specs(commercial: dict[str, Any], forecast: list[dict[str, Any]], sensitivity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    charts: list[dict[str, Any]] = []
    if forecast:
        charts.append(
            {
                "chart_id": "revenue_forecast",
                "title": "Revenue Forecast",
                "type": "line",
                "data": [
                    {
                        "year": row["year"],
                        "calendar_year": row.get("calendar_year"),
                        "revenue": row["revenue"],
                        "risk_adjusted_revenue": row["risk_adjusted_revenue"],
                    }
                    for row in forecast
                ],
                "x_key": "year",
                "y_keys": ["revenue", "risk_adjusted_revenue"],
                "notes": "Chart data is sourced from commercial_model and rnpv outputs.",
            }
        )
    funnel = _patient_funnel_chart_data(commercial)
    if funnel:
        charts.append(
            {
                "chart_id": "patient_funnel",
                "title": "Patient Funnel",
                "type": "funnel",
                "data": funnel,
                "x_key": "population_step",
                "y_keys": ["patients"],
                "notes": "Chart data is sourced from commercial_model.patient_funnel.",
            }
        )
    if sensitivity:
        charts.append(
            {
                "chart_id": "rnpv_sensitivity",
                "title": "rNPV Sensitivity",
                "type": "tornado",
                "data": sensitivity,
                "x_key": "variable",
                "y_keys": ["low_case_rnpv", "base_case_rnpv", "high_case_rnpv"],
                "notes": "Chart data is deterministically recalculated from the commercial forecast and rNPV assumptions.",
            }
        )
    return charts


def _patient_funnel_chart_data(commercial: dict[str, Any]) -> list[dict[str, Any]]:
    funnel = _dict(commercial.get("patient_funnel"))
    data = [
        ("disease population", funnel.get("starting_population") or _value(commercial, "selected_population_measure", "value")),
        ("diagnosed patients", funnel.get("diagnosed_patients")),
        ("treated/managed patients", funnel.get("treated_or_managed_patients")),
        ("eligible patients", funnel.get("eligible_patients")),
        ("commercially addressable patients", funnel.get("commercially_addressable_patients")),
        ("peak treated patients", _peak_treated(commercial)),
    ]
    rows = []
    previous = None
    for label, value in data:
        numeric = _number(value)
        if numeric is None:
            continue
        conversion = numeric / previous if previous else None
        rows.append(
            {
                "population_step": label,
                "patients": numeric,
                "conversion_from_prior": _available(_round(conversion)),
            }
        )
        previous = numeric
    return rows


def _calculate_rnpv(forecast_rows: list[Any], rnpv: dict[str, Any]) -> float | None:
    pos = _number(rnpv.get("probability_of_success"))
    launch_year = _number(rnpv.get("launch_year"))
    loe_year = _number(rnpv.get("loe_year"))
    discount = _number(rnpv.get("discount_rate"))
    margin = _number(rnpv.get("operating_margin"))
    development_cost = _number(rnpv.get("development_cost"))
    tax_rate = _number(_assumption_value(rnpv, "tax_rate")) or 0.21
    valuation_year = _number(_assumption_value(rnpv, "valuation_year")) or 2026
    if None in (pos, launch_year, loe_year, discount, margin, development_cost):
        return None
    value = -float(development_cost)
    for row in forecast_rows:
        if not isinstance(row, dict):
            continue
        year = _number(row.get("year"))
        revenue = _number(row.get("net_revenue") if row.get("net_revenue") is not None else row.get("revenue"))
        if year is None or revenue is None:
            continue
        calendar_year = int(launch_year) + int(year) - 1
        if calendar_year > int(loe_year):
            continue
        years = max(0, calendar_year - int(valuation_year))
        cash = revenue * float(margin) * (1 - tax_rate) * float(pos)
        value += cash / ((1 + float(discount)) ** years)
    return round(value, 2)


def _top_confidence_flags(raw: dict[str, Any]) -> list[dict[str, Any]]:
    flags = []
    for flag in _list(raw.get("red_flags")):
        if isinstance(flag, dict):
            flags.append({"module": flag.get("category"), "severity": flag.get("severity"), "message": flag.get("reason"), "requires_human_review": True})
    for flag in _list(raw.get("missing_data_flags")):
        if isinstance(flag, dict):
            flags.append({"module": flag.get("section"), "severity": flag.get("severity"), "message": flag.get("reason"), "requires_human_review": True})
    rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    deduped = []
    seen = set()
    for flag in sorted(flags, key=lambda item: (rank.get(str(item.get("severity")), 5), str(item.get("message")))):
        key = str(flag.get("message"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(flag)
    return deduped[:5]


def _rnpv_interpretation(rnpv: dict[str, Any]) -> str | None:
    value = _number(rnpv.get("rnpv"))
    development_cost = _number(rnpv.get("development_cost"))
    if value is None or development_cost is None:
        return None
    if value < 0:
        return "Base rNPV is negative after probability of success, discounting, and remaining development spend."
    return "Base rNPV is positive under the current source-backed and default assumptions."


def _peak_treated(commercial: dict[str, Any]) -> float | None:
    rows = _list(commercial.get("revenue_forecast"))
    values = [_number(row.get("treated_patients")) for row in rows if isinstance(row, dict)]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _assumption_value(section: dict[str, Any], name: str) -> Any:
    for item in _list(section.get("assumptions")):
        if isinstance(item, dict) and item.get("name") == name:
            return item.get("value")
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _value(raw: dict[str, Any], *path: str) -> Any:
    current: Any = raw
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            return None
    return current


def _available(value: Any) -> Any:
    if value is None or value == "":
        return NOT_AVAILABLE
    return value


def _number(value: Any) -> float | None:
    if value in (None, "", NOT_AVAILABLE):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any) -> float | None:
    number = _number(value)
    return round(number, 4) if number is not None else None


def _round_money(value: Any) -> float | None:
    number = _number(value)
    return round(number, 2) if number is not None else None


def _round_for_display(value: Any) -> Any:
    number = _number(value)
    if number is None:
        return value
    return round(number, 4) if abs(number) < 1 else round(number, 2)


def _confidence_label(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "low"
    if number >= 0.75:
        return "high"
    if number >= 0.45:
        return "medium"
    return "low"
