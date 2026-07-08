"""Deterministic rNPV calculator for Agent 4 due diligence."""

from __future__ import annotations

from pharma_os.schemas import CommercialModelOutput, MissingDataFlag, PatentExclusivityOutput, PoSOutput, RNPVOutput
from pharma_os.tools._due_diligence_common import (
    assumption,
    development_cost_from_config,
    launch_year_from_config,
    missing,
    select_assumption_value,
    to_float,
)
from pharma_os.tools.rules import config_source_id, load_config


def build_rnpv(
    *,
    commercial: CommercialModelOutput,
    pos: PoSOutput,
    patent: PatentExclusivityOutput,
    launch_year: int | None,
    loe_year: int | None,
    discount_rate: float | None,
    operating_margin: float | None,
    development_cost: float | None,
    phase: str | None = None,
) -> RNPVOutput:
    """Calculate rNPV when all upstream sourced/reviewed inputs exist."""

    flags: list[MissingDataFlag] = []
    config = load_config("rnpv_assumptions_config.yaml", section="due_diligence")
    config_id = config_source_id("rnpv_assumptions_config.yaml", section="due_diligence")
    selected_launch_year = select_assumption_value(
        "rnpv-launch-year",
        "launch_year",
        launch_year,
        "year",
        config_value=launch_year_from_config(config, phase),
        config_filename="rnpv_assumptions_config.yaml",
        config_field_path=f"launch_timing.default_years_to_launch_by_phase.{phase or 'default'}",
    )
    selected_discount_rate = select_assumption_value(
        "rnpv-discount-rate",
        "discount_rate",
        discount_rate,
        "fraction",
        config_value=to_float(config.get("discount_rate")),
        config_filename="rnpv_assumptions_config.yaml",
        config_field_path="discount_rate",
    )
    selected_operating_margin = select_assumption_value(
        "rnpv-operating-margin",
        "operating_margin",
        operating_margin,
        "fraction",
        config_value=to_float(config.get("operating_margin")),
        config_filename="rnpv_assumptions_config.yaml",
        config_field_path="operating_margin",
    )
    selected_development_cost = select_assumption_value(
        "rnpv-development-cost",
        "development_cost",
        development_cost,
        "USD",
        config_value=development_cost_from_config(config, phase),
        config_filename="rnpv_assumptions_config.yaml",
        config_field_path=f"development_costs.by_phase.{phase or 'default'}.total_cost",
    )
    selected_tax_rate = select_assumption_value(
        "rnpv-tax-rate",
        "tax_rate",
        None,
        "fraction",
        config_value=to_float(config.get("tax_rate")),
        config_filename="rnpv_assumptions_config.yaml",
        config_field_path="tax_rate",
    )
    selected_valuation_year = select_assumption_value(
        "rnpv-valuation-year",
        "valuation_year",
        None,
        "year",
        config_value=to_float(config.get("valuation_year")),
        config_filename="rnpv_assumptions_config.yaml",
        config_field_path="valuation_year",
    )
    selected_loe_year = loe_year or patent.estimated_loe_year
    assumptions = [
        selected_launch_year,
        assumption(
            "rnpv-loe-year",
            "loe_year",
            selected_loe_year,
            "year",
            "cli.due_diligence" if loe_year is not None else "Lens/regulatory source",
            assumption_type="user_reviewed" if loe_year is not None else "source_derived",
            source_ids=() if loe_year is not None else patent.source_ids,
        ),
        selected_discount_rate,
        selected_operating_margin,
        selected_development_cost,
        selected_tax_rate,
        selected_valuation_year,
    ]
    for selected_assumption in assumptions:
        if selected_assumption.value is None:
            flags.append(missing(f"{selected_assumption.assumption_id}-missing", "rnpv", selected_assumption.name, "Reviewed rNPV assumption is required.", "high"))
    if not commercial.calculable:
        flags.append(missing("rnpv-commercial-not-calculable", "rnpv", "commercial_model", "Commercial model is not calculable.", "high"))
    if pos.probability_of_success is None:
        flags.append(missing("rnpv-pos-missing", "rnpv", "probability_of_success", "PoS must come from workbook.", "high"))
    if selected_loe_year is None:
        flags.append(missing("rnpv-loe-missing", "rnpv", "loe_year", "LOE must cite Lens/regulatory source or human review.", "high"))
    if selected_launch_year.value and selected_loe_year and selected_loe_year < int(float(selected_launch_year.value)):
        flags.append(missing("rnpv-loe-before-launch", "rnpv", "loe_year", "LOE year is before launch year.", "critical"))
    calculable = not flags
    value = None
    if calculable:
        value = -float(selected_development_cost.value)
        for row in commercial.revenue_forecast:
            calendar_year = int(float(selected_launch_year.value)) + row.year - 1
            if calendar_year > int(selected_loe_year):
                continue
            years = max(0, calendar_year - int(float(selected_valuation_year.value)))
            cash_flow = row.net_revenue * float(selected_operating_margin.value) * (1 - float(selected_tax_rate.value)) * float(pos.probability_of_success)
            value += cash_flow / ((1 + float(selected_discount_rate.value)) ** years)
        value = round(value, 2)
    return RNPVOutput(
        calculable=calculable,
        rnpv=value,
        probability_of_success=pos.probability_of_success,
        loe_year=selected_loe_year,
        launch_year=int(float(selected_launch_year.value)) if selected_launch_year.value is not None else None,
        discount_rate=float(selected_discount_rate.value) if selected_discount_rate.value is not None else None,
        operating_margin=float(selected_operating_margin.value) if selected_operating_margin.value is not None else None,
        development_cost=float(selected_development_cost.value) if selected_development_cost.value is not None else None,
        assumptions=tuple(assumptions),
        source_ids=tuple(dict.fromkeys([*commercial.source_ids, *pos.source_ids, *patent.source_ids, *([config_id] if any(config_id in item.source_ids for item in assumptions) else [])])),
        missing_data_flags=tuple(flags),
        confidence=0.75 if calculable else 0.1,
    )
