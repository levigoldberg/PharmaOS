"""Deterministic commercial model for Agent 4 due diligence."""

from __future__ import annotations

from pharma_os.schemas import CommercialModelOutput, MissingDataFlag, PricingOutput, RevenueForecastYear
from pharma_os.tools._due_diligence_common import assumption, missing, select_assumption_value, to_float, triplet_base
from pharma_os.tools.rules import config_provenance, config_source_id, load_config


def build_commercial_model(
    *,
    annual_patients: float | None,
    peak_penetration: float | None,
    gross_to_net: float | None,
    pricing: PricingOutput,
) -> CommercialModelOutput:
    """Run a compact deterministic commercial model when reviewed inputs exist."""

    flags: list[MissingDataFlag] = []
    archetype_name = "chronic_specialty_prevalence"
    config = load_config("default_archetypes.yaml", section="due_diligence")
    config_id = config_source_id("default_archetypes.yaml", section="due_diligence")
    archetype = (config.get("archetypes") or {}).get(archetype_name, {})
    selected_peak_penetration = select_assumption_value(
        "commercial-peak-penetration",
        "peak_penetration",
        peak_penetration,
        "fraction",
        config_value=triplet_base(archetype.get("peak_penetration")),
        config_filename="default_archetypes.yaml",
        config_field_path=f"archetypes.{archetype_name}.peak_penetration.base",
    )
    selected_gross_to_net = select_assumption_value(
        "commercial-gross-to-net",
        "gross_to_net",
        gross_to_net,
        "fraction",
        config_value=triplet_base(archetype.get("gross_to_net")),
        config_filename="default_archetypes.yaml",
        config_field_path=f"archetypes.{archetype_name}.gross_to_net.base",
    )
    launch_ramp = [float(value) for value in archetype.get("launch_ramp") or [] if to_float(value) is not None]
    assumptions = [
        assumption("commercial-annual-patients", "annual_patients", annual_patients, "patients", "cli.due_diligence", assumption_type="user_reviewed"),
        selected_peak_penetration,
        selected_gross_to_net,
    ]
    if launch_ramp:
        assumptions.append(
            assumption(
                "commercial-launch-ramp",
                "launch_ramp",
                launch_ramp,
                "fraction_by_year",
                config_provenance("default_archetypes.yaml", f"archetypes.{archetype_name}.launch_ramp", section="due_diligence"),
                assumption_type="config_default",
                source_ids=(config_id,),
            )
        )
    else:
        flags.append(missing("commercial-launch-ramp-missing", "commercial_model", "launch_ramp", "No launch ramp was available from default_archetypes.yaml.", "high"))
    for selected_assumption in assumptions:
        if selected_assumption.value is None:
            flags.append(missing(f"{selected_assumption.assumption_id}-missing", "commercial_model", selected_assumption.name, "No source-backed, user-reviewed, or config fallback value is available.", "high"))
    if pricing.annual_wac is None:
        flags.append(missing("commercial-annual-wac-missing", "commercial_model", "annual_wac", "Annual WAC must come from pricing evidence.", "high"))
    calculable = not flags
    rows: list[RevenueForecastYear] = []
    net_price = None
    peak_sales = None
    if calculable:
        net_price = float(pricing.annual_wac) * (1 - float(selected_gross_to_net.value))
        for year, ramp in enumerate(launch_ramp, start=1):
            treated = float(annual_patients) * float(selected_peak_penetration.value) * ramp
            rows.append(RevenueForecastYear(year=year, treated_patients=round(treated, 2), net_price=round(net_price, 2), net_revenue=round(treated * net_price, 2)))
        peak_sales = rows[-1].net_revenue
    return CommercialModelOutput(
        calculable=calculable,
        annual_patients=annual_patients,
        peak_penetration=float(selected_peak_penetration.value) if selected_peak_penetration.value is not None else None,
        gross_to_net=float(selected_gross_to_net.value) if selected_gross_to_net.value is not None else None,
        net_price=net_price,
        peak_net_sales=peak_sales,
        revenue_forecast=tuple(rows),
        assumptions=tuple(assumptions),
        source_ids=tuple(dict.fromkeys([*pricing.source_ids, *([config_id] if any(item.source_ids and config_id in item.source_ids for item in assumptions) else [])])),
        missing_data_flags=tuple(flags),
        confidence=0.75 if calculable else 0.15,
    )
