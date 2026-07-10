"""Commercial market sizing and deterministic revenue model for Agent 4."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from pharma_os.agent_runtime import AgentRuntimeConfig, run_structured_llm_call, runtime_config_for_route
from pharma_os.schemas import (
    AssetIdentityOutput,
    ClinicalEvidenceSummary,
    ClinicalTrialRecord,
    CommercialAssumptionLedgerRecord,
    CommercialAssumptionTriplet,
    CommercialCaseOutput,
    CommercialInputBundle,
    CommercialModelOutput,
    CommercialPenetrationCase,
    CommercialPricingCase,
    MarketSizingInterpretation,
    MissingDataFlag,
    PatientFunnel,
    PricingOutput,
    RevenueForecastYear,
    SelectedPopulationMeasure,
    ValueTriplet,
)
from pharma_os.tools._due_diligence_common import (
    assumption,
    missing,
    select_assumption_value,
    to_float,
    triplet_base,
)
from pharma_os.tools.rules import config_provenance, config_source_id, load_config


class CommercialModelRunResult(BaseModel):
    """Commercial model output plus the optional market-sizing agent trace."""

    output: CommercialModelOutput
    agent_trace: object | None = None
    trace_metadata: dict[str, str | int | float | bool | None] = {}


MARKET_SIZING_INSTRUCTIONS = """You are a market-sizing interpretation subagent for PharmaOS Agent 4 due diligence.
Use only the supplied commercial input bundle. Do not invent source facts.
Do not calculate revenue, treated patients, launch ramp, peak sales, rNPV, discounted revenue, probability-adjusted revenue, or LOE-adjusted revenue.
Your job is only to select or infer market-sizing assumptions for a deterministic Python calculator.
If no usable population measure exists, set calculable false, set the selected population measure value to null, and explain the missing evidence.
You may infer missing diagnosed, treated, eligibility, or commercially addressable fractions only when the source_type is model_inferred, default_assumption, or fallback.
Return structured output matching the schema only."""


def build_commercial_model(
    *,
    annual_patients: float | None,
    peak_penetration: float | None,
    gross_to_net: float | None,
    pricing: PricingOutput,
    trial: ClinicalTrialRecord | None = None,
    asset: AssetIdentityOutput | None = None,
    clinical_evidence: ClinicalEvidenceSummary | None = None,
    run_id: str | None = None,
    config: AgentRuntimeConfig | None = None,
) -> CommercialModelOutput:
    """Build a commercial model while preserving the legacy output contract."""

    return build_commercial_model_with_trace(
        annual_patients=annual_patients,
        peak_penetration=peak_penetration,
        gross_to_net=gross_to_net,
        pricing=pricing,
        trial=trial,
        asset=asset,
        clinical_evidence=clinical_evidence,
        run_id=run_id,
        config=config,
    ).output


def build_commercial_model_with_trace(
    *,
    annual_patients: float | None,
    peak_penetration: float | None,
    gross_to_net: float | None,
    pricing: PricingOutput,
    trial: ClinicalTrialRecord | None = None,
    asset: AssetIdentityOutput | None = None,
    clinical_evidence: ClinicalEvidenceSummary | None = None,
    run_id: str | None = None,
    config: AgentRuntimeConfig | None = None,
) -> CommercialModelRunResult:
    """Build a Panoptic-style commercial model and retain market-sizing trace metadata."""

    default_config = load_config("default_archetypes.yaml", section="due_diligence")
    config_id = config_source_id("default_archetypes.yaml", section="due_diligence")
    bundle = assemble_commercial_input_bundle(
        trial=trial,
        asset=asset,
        clinical_evidence=clinical_evidence,
        pricing=pricing,
        annual_patients=annual_patients,
        peak_penetration=peak_penetration,
        gross_to_net=gross_to_net,
        default_archetypes=default_config,
    )
    if annual_patients is not None:
        interpretation = _interpretation_from_reviewed_population(
            annual_patients=annual_patients,
            default_config=default_config,
        )
        agent_trace = None
        trace_metadata: dict[str, str | int | float | bool | None] = {}
    else:
        fallback = _fallback_interpretation(default_config, reason="No source-backed population measure was available.")
        result = run_structured_llm_call(
            agent_name="CommercialMarketSizingAgent",
            instructions=MARKET_SIZING_INSTRUCTIONS,
            payload=bundle,
            output_type=MarketSizingInterpretation,
            run_id=run_id or f"commercial-market-sizing-{datetime.now(timezone.utc).isoformat()}",
            input_summary="Select market-sizing assumptions for Agent 4 commercial forecast.",
            config=runtime_config_for_route(
                model_route="agent4_subagent",
                disabled_provenance="pharma_os.tools.commercial_model.market_sizing",
                config=config,
            ),
            offline_output=fallback,
            source_ids=tuple(pricing.source_ids),
            confidence=None,
            rationale_summary="Market-sizing interpretation feeds deterministic commercial calculations.",
        )
        interpretation = result.output
        agent_trace = result.trace
        trace_metadata = result.trace_metadata

    output = calculate_commercial_model(
        bundle=bundle,
        interpretation=interpretation,
        default_config=default_config,
        pricing=pricing,
        config_id=config_id,
    )
    return CommercialModelRunResult(output=output, agent_trace=agent_trace, trace_metadata=trace_metadata)


def assemble_commercial_input_bundle(
    *,
    trial: ClinicalTrialRecord | None,
    asset: AssetIdentityOutput | None,
    clinical_evidence: ClinicalEvidenceSummary | None,
    pricing: PricingOutput,
    annual_patients: float | None,
    peak_penetration: float | None,
    gross_to_net: float | None,
    default_archetypes: dict[str, Any],
) -> CommercialInputBundle:
    """Assemble a compact commercial evidence bundle from existing Agent 4 context."""

    user_overrides = {
        key: value
        for key, value in {
            "annual_patients": annual_patients,
            "peak_penetration": peak_penetration,
            "gross_to_net": gross_to_net,
        }.items()
        if value is not None
    }
    population_evidence: list[dict[str, Any]] = []
    if annual_patients is not None:
        population_evidence.append(
            {
                "value": annual_patients,
                "unit": "patients",
                "measure_type": "reviewed_annual_eligible_patients",
                "condition": asset.normalized_indication if asset else None,
                "geography": "United States",
                "source_type": "user_override",
                "evidence_reference": "DueDiligenceInput.annual_patients",
                "rationale": "User-reviewed annual eligible patient assumption supplied to PharmaOS.",
            }
        )
    pubmed_titles = tuple(clinical_evidence.pubmed_titles if clinical_evidence else ())
    segmentation = [
        {
            "value": None,
            "unit": None,
            "measure_type": "literature_metadata",
            "condition": asset.normalized_indication if asset else None,
            "geography": "United States",
            "source_type": "source_derived",
            "evidence_reference": "clinical_evidence.pubmed_titles",
            "rationale": title,
        }
        for title in pubmed_titles
    ]
    missing_inputs = list(pricing.missing_data_flags)
    if annual_patients is None:
        missing_inputs.append(missing("commercial-population-measure-missing", "commercial_model", "selected_population_measure", "No source-backed or reviewed population measure is available.", "high"))
    missing_labels = tuple(dict.fromkeys(flag.flag_id for flag in missing_inputs))
    return CommercialInputBundle(
        asset_summary={
            "asset_name": asset.asset_name if asset else None,
            "sponsor": asset.sponsor if asset else None,
            "nct_id": trial.nct_id if trial else asset.nct_id if asset else None,
            "indication": asset.normalized_indication if asset else None,
            "condition_terms": list(trial.conditions) if trial else [],
            "phase": (trial.phases[0] if trial and trial.phases else None),
            "status": trial.overall_status if trial else None,
            "modality": asset.modality if asset else None,
        },
        disease_population_evidence=tuple(population_evidence),
        prevalence_evidence=tuple(item for item in population_evidence if "preval" in str(item.get("measure_type", "")).casefold() or item.get("measure_type") == "reviewed_annual_eligible_patients"),
        incidence_evidence=tuple(item for item in population_evidence if "incidence" in str(item.get("measure_type", "")).casefold()),
        segmentation_evidence=tuple(segmentation),
        trial_eligibility_criteria={
            "eligibility_criteria": trial.eligibility_criteria if trial else None,
            "enrollment_count": trial.enrollment_count if trial else None,
            "primary_endpoints": list(trial.primary_endpoints) if trial else [],
        },
        pricing_benchmark=_pricing_benchmark(pricing, default_archetypes),
        missing_inputs=missing_labels,
        user_overrides=user_overrides,
        predefined_archetype_assumptions=default_archetypes,
    )


def calculate_commercial_model(
    *,
    bundle: CommercialInputBundle,
    interpretation: MarketSizingInterpretation,
    default_config: dict[str, Any],
    pricing: PricingOutput,
    config_id: str,
) -> CommercialModelOutput:
    """Calculate deterministic revenue cases from selected market-sizing assumptions."""

    archetype_name = interpretation.selected_market_archetype or "chronic_specialty_prevalence"
    annual_wac = _annual_wac_triplet(bundle.pricing_benchmark, default_config)
    gross_to_net, gross_source = _triplet_from_archetype(default_config, archetype_name, "gross_to_net", bundle.user_overrides)
    peak_penetration, peak_source = _triplet_from_archetype(default_config, archetype_name, "peak_penetration", bundle.user_overrides)
    launch_ramp, ramp_source = _launch_ramp_from_archetype(default_config, archetype_name, bundle.user_overrides)
    ledger = _ledger_records(
        interpretation=interpretation,
        annual_wac=annual_wac,
        gross_to_net=gross_to_net,
        gross_source=gross_source,
        peak_penetration=peak_penetration,
        peak_source=peak_source,
        launch_ramp=launch_ramp,
        ramp_source=ramp_source,
    )
    missing_values = _missing_required_values(interpretation, annual_wac, gross_to_net, peak_penetration, launch_ramp)
    flags = _commercial_missing_flags(bundle, interpretation, missing_values)
    calculable = interpretation.calculable and not missing_values
    cases = (
        _build_cases(
            interpretation=interpretation,
            annual_wac=annual_wac,
            gross_to_net=gross_to_net,
            peak_penetration=peak_penetration,
            launch_ramp=launch_ramp,
            forecast_years=int(default_config.get("forecast_years", len(launch_ramp) or 1)),
        )
        if calculable
        else {}
    )
    base = cases.get("base")
    legacy_assumptions = _legacy_assumptions(
        interpretation=interpretation,
        peak_penetration=peak_penetration,
        gross_to_net=gross_to_net,
        launch_ramp=launch_ramp,
        config_id=config_id,
        archetype_name=archetype_name,
        user_overrides=bundle.user_overrides,
    )
    source_ids = tuple(
        dict.fromkeys(
            (
                *pricing.source_ids,
                config_id,
            )
        )
    )
    confidence_flags = tuple(
        dict.fromkeys(
            (
                *bundle.missing_inputs,
                *interpretation.assumption_flags,
                *interpretation.human_review_flags,
                *(f"missing_required_commercial_input:{item}" for item in missing_values),
            )
        )
    )
    return CommercialModelOutput(
        calculable=calculable,
        annual_patients=base.patient_funnel.commercially_addressable_patients if base else _float_or_none(bundle.user_overrides.get("annual_patients")),
        peak_penetration=base.penetration.peak_penetration if base else peak_penetration.base,
        gross_to_net=base.pricing.gross_to_net if base else gross_to_net.base,
        net_price=base.pricing.net_price if base else None,
        peak_net_sales=base.peak_net_sales if base else None,
        revenue_forecast=base.revenue_forecast if base else (),
        selected_market_archetype=archetype_name,
        market_basis=interpretation.market_basis,
        selected_population_measure=interpretation.selected_population_measure,
        patient_funnel=base.patient_funnel if base else None,
        cases=cases,
        assumption_ledger=tuple(ledger),
        commercial_input_bundle_summary=_bundle_summary(bundle),
        confidence_flags=confidence_flags,
        human_review_questions=tuple(_human_review_questions(interpretation, missing_values)),
        assumptions=tuple(legacy_assumptions),
        source_ids=source_ids,
        missing_data_flags=tuple(flags),
        confidence=_confidence(calculable, interpretation.confidence_score, flags),
    )


def _interpretation_from_reviewed_population(*, annual_patients: float, default_config: dict[str, Any]) -> MarketSizingInterpretation:
    archetype_name = "chronic_specialty_prevalence"
    return MarketSizingInterpretation(
        calculable=True,
        selected_market_archetype=archetype_name,
        market_basis="prevalence_stock",
        selected_population_measure=SelectedPopulationMeasure(
            value=annual_patients,
            unit="patients",
            measure_type="reviewed_annual_eligible_patients",
            condition=None,
            geography="United States",
            source_type="user_override",
            rationale="User-reviewed annual eligible patient assumption supplied to PharmaOS.",
            evidence_reference="DueDiligenceInput.annual_patients",
            confidence_score=8,
            human_review_required=False,
        ),
        yearly_eligible_patient_logic="Use reviewed annual eligible patients directly; funnel fractions are set to 1.0 to preserve the reviewed input.",
        diagnosed_fraction=_unit_fraction("diagnosed_fraction", source_type="user_override"),
        treated_fraction=_unit_fraction("treated_fraction", source_type="user_override"),
        eligibility_fraction=_unit_fraction("eligibility_fraction", source_type="user_override"),
        commercially_addressable_fraction=_unit_fraction("commercially_addressable_fraction", source_type="user_override"),
        rationale="Reviewed annual eligible patients are authoritative for this run.",
        confidence_score=8,
        key_evidence_used=("DueDiligenceInput.annual_patients",),
        assumption_flags=(),
        human_review_flags=(),
    )


def _fallback_interpretation(default_config: dict[str, Any], *, reason: str) -> MarketSizingInterpretation:
    archetype_name = "chronic_specialty_prevalence"
    archetype = (default_config.get("archetypes") or {}).get(archetype_name, {})
    return MarketSizingInterpretation(
        calculable=False,
        selected_market_archetype=archetype_name,
        market_basis=archetype.get("market_basis") or "prevalence_stock",
        selected_population_measure=SelectedPopulationMeasure(
            value=None,
            unit="patients",
            measure_type=None,
            condition=None,
            geography="United States",
            source_type="missing",
            rationale=reason,
            evidence_reference=None,
            confidence_score=0,
            human_review_required=True,
        ),
        yearly_eligible_patient_logic="Commercial model cannot calculate until a source-backed or reviewed population measure is available.",
        diagnosed_fraction=_fraction_from_archetype(archetype, "diagnosed_fraction"),
        treated_fraction=_fraction_from_archetype(archetype, "treated_fraction"),
        eligibility_fraction=_fraction_from_archetype(archetype, "eligibility_fraction"),
        commercially_addressable_fraction=_fraction_from_archetype(archetype, "commercially_addressable_fraction"),
        rationale=reason,
        confidence_score=0,
        key_evidence_used=(),
        assumption_flags=("selected_population_measure_missing",),
        human_review_flags=("Confirm the source-backed disease population or provide reviewed annual eligible patients.",),
    )


def _unit_fraction(name: str, *, source_type: str) -> CommercialAssumptionTriplet:
    return CommercialAssumptionTriplet(
        low=1.0,
        base=1.0,
        high=1.0,
        source_type=source_type,  # type: ignore[arg-type]
        rationale=f"{name} set to 1.0 because annual_patients already represents the reviewed eligible/addressable population.",
        evidence_reference="DueDiligenceInput.annual_patients",
        confidence_score=8,
        human_review_required=False,
    )


def _fraction_from_archetype(archetype: dict[str, Any], key: str) -> CommercialAssumptionTriplet:
    value = archetype.get(key) or {}
    return CommercialAssumptionTriplet(
        low=to_float(value.get("low")),
        base=to_float(value.get("base")),
        high=to_float(value.get("high")),
        source_type="default_assumption",
        rationale=f"{key} fallback from default_archetypes.yaml.",
        evidence_reference=f"default_archetypes.yaml:{key}",
        confidence_score=5,
        human_review_required=True,
    )


def _pricing_benchmark(pricing: PricingOutput, default_config: dict[str, Any]) -> dict[str, Any]:
    base = pricing.annual_wac
    multipliers = default_config.get("price_sensitivity_multipliers") or {}
    low_multiplier = float(multipliers.get("low", 0.8))
    high_multiplier = float(multipliers.get("high", 1.2))
    return {
        "annual_gross_wac": {
            "low": base * low_multiplier if base is not None else None,
            "base": base,
            "high": base * high_multiplier if base is not None else None,
        },
        "source": "pricing_output.annual_wac" if base is not None else None,
        "primary_analog": pricing.matched_product,
        "annualization_details": pricing.annualization_details,
    }


def _annual_wac_triplet(pricing_benchmark: dict[str, Any], default_config: dict[str, Any]) -> ValueTriplet:
    del default_config
    value = pricing_benchmark.get("annual_gross_wac") or {}
    return ValueTriplet(low=_float_or_none(value.get("low")), base=_float_or_none(value.get("base")), high=_float_or_none(value.get("high")))


def _triplet_from_archetype(
    default_config: dict[str, Any],
    archetype_name: str | None,
    key: str,
    user_overrides: dict[str, Any],
) -> tuple[ValueTriplet, str]:
    override = user_overrides.get(key)
    if override is not None:
        value = _triplet(override) if isinstance(override, dict) else ValueTriplet(low=float(override), base=float(override), high=float(override))
        return value, "user_override"
    archetype = _archetype(default_config, archetype_name)
    return _triplet(archetype.get(key) or {}), "default_assumption"


def _launch_ramp_from_archetype(default_config: dict[str, Any], archetype_name: str | None, user_overrides: dict[str, Any]) -> tuple[tuple[float, ...], str]:
    override = user_overrides.get("launch_ramp")
    if override is not None:
        return tuple(float(item) for item in override), "user_override"
    value = _archetype(default_config, archetype_name).get("launch_ramp") or ()
    return tuple(float(item) for item in value if to_float(item) is not None), "default_assumption"


def _archetype(default_config: dict[str, Any], archetype_name: str | None) -> dict[str, Any]:
    archetypes = default_config.get("archetypes") or {}
    if archetype_name and archetype_name in archetypes:
        return archetypes[archetype_name]
    if "chronic_specialty_prevalence" in archetypes:
        return archetypes["chronic_specialty_prevalence"]
    return next(iter(archetypes.values()), {})


def _triplet(value: dict[str, Any]) -> ValueTriplet:
    return ValueTriplet(low=_float_or_none(value.get("low")), base=_float_or_none(value.get("base")), high=_float_or_none(value.get("high")))


def _build_cases(
    *,
    interpretation: MarketSizingInterpretation,
    annual_wac: ValueTriplet,
    gross_to_net: ValueTriplet,
    peak_penetration: ValueTriplet,
    launch_ramp: tuple[float, ...],
    forecast_years: int,
) -> dict[str, CommercialCaseOutput]:
    cases: dict[str, CommercialCaseOutput] = {}
    for case in ("downside", "base", "upside"):
        starting_population = _case_value(ValueTriplet(low=interpretation.selected_population_measure.value, base=interpretation.selected_population_measure.value, high=interpretation.selected_population_measure.value), case)
        diagnosed_fraction = _case_value(_triplet_from_ai(interpretation.diagnosed_fraction), case)
        treated_fraction = _case_value(_triplet_from_ai(interpretation.treated_fraction), case)
        eligibility_fraction = _case_value(_triplet_from_ai(interpretation.eligibility_fraction), case)
        addressable_fraction = _case_value(_triplet_from_ai(interpretation.commercially_addressable_fraction), case)
        annual_price = _case_value(annual_wac, case)
        gtn = _case_value(gross_to_net, case, inverse=True)
        peak = _case_value(peak_penetration, case)
        diagnosed = starting_population * diagnosed_fraction
        treated_or_managed = diagnosed * treated_fraction
        eligible = treated_or_managed * eligibility_fraction
        commercially_addressable = eligible * addressable_fraction
        net_price = annual_price * (1 - gtn)
        rows: list[RevenueForecastYear] = []
        for year_index in range(forecast_years):
            ramp = launch_ramp[year_index] if year_index < len(launch_ramp) else launch_ramp[-1]
            treated = commercially_addressable * peak * ramp
            rows.append(
                RevenueForecastYear(
                    year=year_index + 1,
                    treated_patients=round(treated, 2),
                    net_price=round(net_price, 2),
                    net_revenue=round(treated * net_price, 2),
                )
            )
        peak_gross = commercially_addressable * peak * annual_price
        peak_net = peak_gross * (1 - gtn)
        cases[case] = CommercialCaseOutput(
            patient_funnel=PatientFunnel(
                starting_population=round(starting_population, 4),
                diagnosed_patients=round(diagnosed, 4),
                treated_or_managed_patients=round(treated_or_managed, 4),
                eligible_patients=round(eligible, 4),
                commercially_addressable_patients=round(commercially_addressable, 4),
                diagnosed_fraction=round(diagnosed_fraction, 4),
                treated_fraction=round(treated_fraction, 4),
                eligibility_fraction=round(eligibility_fraction, 4),
                commercially_addressable_fraction=round(addressable_fraction, 4),
            ),
            pricing=CommercialPricingCase(annual_gross_wac=round(annual_price, 2), gross_to_net=round(gtn, 4), net_price=round(net_price, 2)),
            penetration=CommercialPenetrationCase(peak_penetration=round(peak, 4), launch_ramp=tuple(round(item, 4) for item in launch_ramp)),
            revenue_forecast=tuple(rows),
            peak_gross_sales=round(peak_gross, 2),
            peak_net_sales=round(peak_net, 2),
        )
    return cases


def _case_value(value: ValueTriplet, case: str, *, inverse: bool = False) -> float:
    selected = {"downside": value.high if inverse else value.low, "base": value.base, "upside": value.low if inverse else value.high}[case]
    if selected is None:
        raise ValueError(f"Missing {case} value")
    return float(selected)


def _triplet_from_ai(value: CommercialAssumptionTriplet) -> ValueTriplet:
    return ValueTriplet(low=value.low, base=value.base, high=value.high)


def _missing_required_values(
    interpretation: MarketSizingInterpretation,
    annual_wac: ValueTriplet,
    gross_to_net: ValueTriplet,
    peak_penetration: ValueTriplet,
    launch_ramp: tuple[float, ...],
) -> tuple[str, ...]:
    missing_values: list[str] = []
    if interpretation.selected_population_measure.value is None:
        missing_values.append("selected_population_measure.value")
    for name in ("diagnosed_fraction", "treated_fraction", "eligibility_fraction", "commercially_addressable_fraction"):
        value = getattr(interpretation, name)
        if value.low is None or value.base is None or value.high is None:
            missing_values.append(name)
    for name, triplet in (("annual_gross_wac", annual_wac), ("gross_to_net", gross_to_net), ("peak_penetration", peak_penetration)):
        if triplet.low is None or triplet.base is None or triplet.high is None:
            missing_values.append(name)
    if not launch_ramp:
        missing_values.append("launch_ramp")
    return tuple(missing_values)


def _commercial_missing_flags(bundle: CommercialInputBundle, interpretation: MarketSizingInterpretation, missing_values: tuple[str, ...]) -> list[MissingDataFlag]:
    flags: list[MissingDataFlag] = []
    for item in bundle.missing_inputs:
        flags.append(missing(f"commercial-{_slug(item)}", "commercial_model", item, f"Commercial sizing input gap: {item}.", "high" if "missing" in item else "medium"))
    for item in interpretation.assumption_flags:
        flags.append(missing(f"commercial-assumption-{_slug(item)}", "commercial_model", item, f"Market sizing assumption flag: {item}.", "medium"))
    for item in missing_values:
        flags.append(missing(f"commercial-required-{_slug(item)}", "commercial_model", item, f"Required commercial model input is unresolved: {item}.", "high"))
    return _dedupe_flags(flags)


def _legacy_assumptions(
    *,
    interpretation: MarketSizingInterpretation,
    peak_penetration: ValueTriplet,
    gross_to_net: ValueTriplet,
    launch_ramp: tuple[float, ...],
    config_id: str,
    archetype_name: str,
    user_overrides: dict[str, Any],
) -> list[Any]:
    selected_annual_patients = interpretation.selected_population_measure.value
    assumptions = [
        assumption(
            "commercial-annual-patients",
            "annual_patients",
            selected_annual_patients,
            "patients",
            interpretation.selected_population_measure.evidence_reference or "commercial_market_sizing",
            assumption_type="user_reviewed" if interpretation.selected_population_measure.source_type == "user_override" else "source_derived" if selected_annual_patients is not None else "missing",
            source_ids=() if interpretation.selected_population_measure.source_type == "user_override" else (config_id,) if selected_annual_patients is not None else (),
            requires_human_review=interpretation.selected_population_measure.human_review_required,
        ),
        select_assumption_value(
            "commercial-peak-penetration",
            "peak_penetration",
            user_overrides.get("peak_penetration"),
            "fraction",
            config_value=peak_penetration.base,
            config_filename="default_archetypes.yaml",
            config_field_path=f"archetypes.{archetype_name}.peak_penetration.base",
        ),
        select_assumption_value(
            "commercial-gross-to-net",
            "gross_to_net",
            user_overrides.get("gross_to_net"),
            "fraction",
            config_value=gross_to_net.base,
            config_filename="default_archetypes.yaml",
            config_field_path=f"archetypes.{archetype_name}.gross_to_net.base",
        ),
    ]
    if launch_ramp:
        assumptions.append(
            assumption(
                "commercial-launch-ramp",
                "launch_ramp",
                list(launch_ramp),
                "fraction_by_year",
                config_provenance("default_archetypes.yaml", f"archetypes.{archetype_name}.launch_ramp", section="due_diligence"),
                assumption_type="config_default",
                source_ids=(config_id,),
                requires_human_review=True,
            )
        )
    return assumptions


def _ledger_records(
    *,
    interpretation: MarketSizingInterpretation,
    annual_wac: ValueTriplet,
    gross_to_net: ValueTriplet,
    gross_source: str,
    peak_penetration: ValueTriplet,
    peak_source: str,
    launch_ramp: tuple[float, ...],
    ramp_source: str,
) -> list[CommercialAssumptionLedgerRecord]:
    population = interpretation.selected_population_measure
    records = [
        CommercialAssumptionLedgerRecord(
            assumption_name="starting_population",
            value=population.value,
            unit=population.unit,
            source_type=population.source_type,
            rationale=population.rationale,
            evidence_reference=population.evidence_reference,
            confidence_score=population.confidence_score,
            human_review_required=population.human_review_required,
        ),
        _fraction_record("diagnosed_fraction", interpretation.diagnosed_fraction),
        _fraction_record("treated_fraction", interpretation.treated_fraction),
        _fraction_record("eligibility_fraction", interpretation.eligibility_fraction),
        _fraction_record("commercially_addressable_fraction", interpretation.commercially_addressable_fraction),
        CommercialAssumptionLedgerRecord(assumption_name="annual_gross_wac", low=annual_wac.low, base=annual_wac.base, high=annual_wac.high, unit="USD/year", source_type="source_derived" if annual_wac.base is not None else "missing", rationale="Annual gross WAC from PharmaOS pricing module.", evidence_reference="pricing.annual_wac", confidence_score=8 if annual_wac.base is not None else 0, human_review_required=annual_wac.base is None),
        CommercialAssumptionLedgerRecord(assumption_name="gross_to_net", low=gross_to_net.low, base=gross_to_net.base, high=gross_to_net.high, unit="fraction", source_type=gross_source, rationale="Gross-to-net assumption used for deterministic revenue forecast.", evidence_reference="default_archetypes.yaml or user override", confidence_score=5, human_review_required=True),  # type: ignore[arg-type]
        CommercialAssumptionLedgerRecord(assumption_name="peak_penetration", low=peak_penetration.low, base=peak_penetration.base, high=peak_penetration.high, unit="fraction", source_type=peak_source, rationale="Peak penetration assumption used for deterministic forecast.", evidence_reference="default_archetypes.yaml or user override", confidence_score=5, human_review_required=True),  # type: ignore[arg-type]
        CommercialAssumptionLedgerRecord(assumption_name="launch_ramp", value=list(launch_ramp), unit="fraction_by_year", source_type=ramp_source, rationale="Launch ramp assumption used for deterministic forecast.", evidence_reference="default_archetypes.yaml or user override", confidence_score=5, human_review_required=True),  # type: ignore[arg-type]
    ]
    return records


def _fraction_record(name: str, value: CommercialAssumptionTriplet) -> CommercialAssumptionLedgerRecord:
    return CommercialAssumptionLedgerRecord(
        assumption_name=name,
        low=value.low,
        base=value.base,
        high=value.high,
        unit="fraction",
        source_type=value.source_type,
        rationale=value.rationale,
        evidence_reference=value.evidence_reference,
        confidence_score=value.confidence_score,
        human_review_required=value.human_review_required,
    )


def _bundle_summary(bundle: CommercialInputBundle) -> dict[str, Any]:
    return {
        "disease_population_evidence_count": len(bundle.disease_population_evidence),
        "prevalence_evidence_count": len(bundle.prevalence_evidence),
        "incidence_evidence_count": len(bundle.incidence_evidence),
        "segmentation_evidence_count": len(bundle.segmentation_evidence),
        "has_trial_eligibility": bool(bundle.trial_eligibility_criteria),
        "pricing_source": bundle.pricing_benchmark.get("source"),
        "missing_inputs": list(bundle.missing_inputs),
    }


def _human_review_questions(interpretation: MarketSizingInterpretation, missing_values: tuple[str, ...]) -> list[str]:
    questions = [
        "Confirm the selected population measure and market basis before using revenue forecast.",
        "Confirm diagnosed, treated, eligibility, and commercially addressable fractions.",
        "Confirm gross-to-net, peak penetration, and launch ramp assumptions.",
    ]
    if missing_values:
        questions.append(f"Resolve missing required commercial inputs: {', '.join(missing_values)}.")
    questions.extend(interpretation.human_review_flags)
    return list(dict.fromkeys(questions))


def _confidence(calculable: bool, confidence_score: int, flags: list[MissingDataFlag]) -> float:
    if not calculable:
        return 0.1 if any(flag.severity == "high" for flag in flags) else 0.25
    return max(0.2, min(0.85, confidence_score / 10 if confidence_score else 0.65))


def _dedupe_flags(flags: list[MissingDataFlag]) -> list[MissingDataFlag]:
    deduped: dict[str, MissingDataFlag] = {}
    for flag in flags:
        deduped[flag.flag_id] = flag
    return list(deduped.values())


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.casefold()).strip("-") or "unknown"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
