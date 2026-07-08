from __future__ import annotations

from pharma_os.schemas import PatentExclusivityOutput, PoSOutput, PricingOutput
from pharma_os.tools.due_diligence import build_commercial_model, build_rnpv
from pharma_os.tools.rules import config_source_id, load_config


def test_config_loader_reads_shared_and_due_diligence_layout() -> None:
    shared = load_config("modality_rules.yaml", section="shared")
    diligence = load_config("default_archetypes.yaml", section="due_diligence")

    assert any(rule["id"] == "modality_antibody" for rule in shared["rules"])
    assert "chronic_specialty_prevalence" in diligence["archetypes"]


def test_commercial_assumption_precedence_and_config_provenance() -> None:
    output = build_commercial_model(
        annual_patients=1000,
        peak_penetration=0.2,
        gross_to_net=None,
        pricing=PricingOutput(
            annual_wac=1000.0,
            wac_value=1000.0,
            dosing_summary="source-backed dosing",
            source_ids=("wac:fixture", "openfda_label:fixture"),
            confidence=0.8,
        ),
    )

    assumptions = {item.name: item for item in output.assumptions}
    assert assumptions["peak_penetration"].value == 0.2
    assert assumptions["peak_penetration"].assumption_type == "user_reviewed"
    assert assumptions["gross_to_net"].value == 0.18
    assert assumptions["gross_to_net"].assumption_type == "fallback_assumption"
    assert assumptions["gross_to_net"].source_ids == (
        config_source_id("default_archetypes.yaml", section="due_diligence"),
    )
    assert "default_archetypes.yaml:archetypes.chronic_specialty_prevalence.gross_to_net.base" in assumptions["gross_to_net"].provenance
    assert output.calculable


def test_rnpv_uses_config_fallbacks_with_provenance() -> None:
    commercial = build_commercial_model(
        annual_patients=1000,
        peak_penetration=0.2,
        gross_to_net=0.15,
        pricing=PricingOutput(
            annual_wac=1000.0,
            wac_value=1000.0,
            dosing_summary="source-backed dosing",
            source_ids=("wac:fixture", "openfda_label:fixture"),
            confidence=0.8,
        ),
    )
    output = build_rnpv(
        commercial=commercial,
        pos=PoSOutput(
            probability_of_success=0.46,
            current_phase="Phase III",
            disease_area="Neurology",
            source_ids=("pos_workbook:fixture",),
            confidence=0.9,
        ),
        patent=PatentExclusivityOutput(
            asset_name="Example",
            estimated_loe_year=2037,
            source_ids=("human_override:loe:example",),
            confidence=0.7,
        ),
        launch_year=None,
        loe_year=None,
        discount_rate=None,
        operating_margin=None,
        development_cost=None,
        phase="Phase III",
    )

    assumptions = {item.name: item for item in output.assumptions}
    assert output.calculable
    assert assumptions["launch_year"].value == 2028
    assert assumptions["launch_year"].assumption_type == "fallback_assumption"
    assert assumptions["development_cost"].value == 125000000.0
    assert assumptions["development_cost"].source_ids == (
        config_source_id("rnpv_assumptions_config.yaml", section="due_diligence"),
    )
    assert "rnpv_assumptions_config.yaml:development_costs.by_phase.Phase III.total_cost" in assumptions["development_cost"].provenance
