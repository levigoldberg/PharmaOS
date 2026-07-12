from __future__ import annotations

from types import SimpleNamespace

from pharma_os.schemas import (
    AssetIdentityOutput,
    ClinicalTrialRecord,
    CommercialAssumptionTriplet,
    CommercialInputBundle,
    MarketSizingInterpretation,
    PatentExclusivityOutput,
    PoSOutput,
    PricingOutput,
    SelectedPopulationMeasure,
)
from pharma_os.tools import commercial_model as commercial_model_module
from pharma_os.tools.commercial_model import build_commercial_model, build_commercial_model_with_trace
from pharma_os.tools.rnpv import build_rnpv
from pharma_os.tools.rules import config_source_id, load_config


def test_config_loader_reads_shared_and_due_diligence_layout() -> None:
    shared = load_config("modality_rules.yaml", section="shared")
    diligence = load_config("default_archetypes.yaml", section="due_diligence")
    market_queries = load_config("market_query_templates.yaml", section="due_diligence")
    market_buckets = load_config("market_bucket_templates.yaml", section="due_diligence")

    assert any(rule["id"] == "modality_antibody" for rule in shared["rules"])
    assert "chronic_specialty_prevalence" in diligence["archetypes"]
    assert "disease_prevalence" in market_queries["templates"]
    assert "disease_population" in market_buckets


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


def test_commercial_model_ai_unavailable_returns_reviewable_noncalculable_output(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    output = build_commercial_model_with_trace(
        annual_patients=None,
        peak_penetration=None,
        gross_to_net=None,
        pricing=PricingOutput(
            annual_wac=1000.0,
            wac_value=1000.0,
            dosing_summary="source-backed dosing",
            source_ids=("wac:fixture", "openfda_label:fixture"),
            confidence=0.8,
        ),
        run_id="commercial-fixture",
    )

    assert output.output.calculable is False
    assert output.output.selected_market_archetype == "chronic_specialty_prevalence"
    assert output.output.assumption_ledger
    assert output.output.human_review_questions
    assert any(flag.field == "selected_population_measure.value" for flag in output.output.missing_data_flags)
    assert output.agent_trace is not None


def test_commercial_model_uses_ai_extracted_market_population(monkeypatch) -> None:
    class FakePubMedClient:
        def search(self, query, *, max_results=5):
            return (
                commercial_model_module.PubMedArticle(
                    pmid="123",
                    title="United States atopic dermatitis prevalence",
                    abstract_snippet="An estimated 1000000 patients in the United States have the target condition.",
                ),
            )

    def fake_llm_call(**kwargs):
        if kwargs["output_type"].__name__ == "EpidemiologyEvidenceExtraction":
            output = commercial_model_module.EpidemiologyEvidenceExtraction(
                evidence_status="population_measure_selected",
                selected_population_measure=SelectedPopulationMeasure(
                    value=1_000_000.0,
                    unit="patients",
                    measure_type="patient_count",
                    condition="atopic dermatitis",
                    geography="United States",
                    source_type="source_derived",
                    rationale="PubMed abstract states a US patient count.",
                    evidence_reference="pubmed:123",
                    confidence_score=8,
                    human_review_required=True,
                ),
                selected_source_id="pubmed:123",
                confidence_score=8,
            )
        else:
            triplet = CommercialAssumptionTriplet(
                low=1.0,
                base=1.0,
                high=1.0,
                source_type="source_derived",
                rationale="Use selected source-backed denominator directly.",
                evidence_reference="pubmed:123",
                confidence_score=8,
                human_review_required=True,
            )
            output = MarketSizingInterpretation(
                calculable=True,
                selected_market_archetype="chronic_specialty_prevalence",
                market_basis="prevalence_stock",
                selected_population_measure=SelectedPopulationMeasure(
                    value=1_000_000.0,
                    unit="patients",
                    measure_type="patient_count",
                    condition="atopic dermatitis",
                    geography="United States",
                    source_type="source_derived",
                    rationale="Selected from PubMed evidence.",
                    evidence_reference="pubmed:123",
                    confidence_score=8,
                    human_review_required=True,
                ),
                yearly_eligible_patient_logic="Use the selected disease-population denominator directly.",
                diagnosed_fraction=triplet,
                treated_fraction=triplet,
                eligibility_fraction=triplet,
                commercially_addressable_fraction=triplet,
                rationale="Source-backed market denominator was available.",
                confidence_score=8,
                key_evidence_used=("pubmed:123",),
            )
        return SimpleNamespace(output=output, trace=SimpleNamespace(agent_name=kwargs["agent_name"]), trace_metadata={"agent_name": kwargs["agent_name"]})

    monkeypatch.setattr(commercial_model_module, "PubMedClient", FakePubMedClient)
    monkeypatch.setattr(commercial_model_module, "run_structured_llm_call", fake_llm_call)

    output = build_commercial_model_with_trace(
        annual_patients=None,
        peak_penetration=None,
        gross_to_net=None,
        pricing=PricingOutput(
            annual_wac=1000.0,
            wac_value=1000.0,
            dosing_summary="source-backed dosing",
            source_ids=("wac:fixture",),
            confidence=0.8,
        ),
        trial=ClinicalTrialRecord(
            nct_id="NCT12345678",
            brief_title="Trial in patients with atopic dermatitis",
            conditions=("Atopic Dermatitis",),
            source_id="ctgov:NCT12345678",
        ),
        asset=AssetIdentityOutput(
            nct_id="NCT12345678",
            asset_name="Example",
            normalized_indication="atopic dermatitis",
            source_ids=("ctgov:NCT12345678",),
            confidence=0.8,
        ),
        run_id="commercial-market-fixture",
    )

    assert output.output.calculable
    assert output.output.annual_patients == 1_000_000.0
    assert output.output.revenue_forecast
    assert "pubmed:123" in output.output.source_ids
    assert len(output.agent_traces) == 2


def test_commercial_model_passes_us_population_denominator_for_prevalence_conversion(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePubMedClient:
        def search(self, query, *, max_results=5):
            return (
                commercial_model_module.PubMedArticle(
                    pmid="321",
                    title="United States atopic dermatitis prevalence",
                    abstract_snippet="Prevalence of atopic dermatitis was 7.3% among adults in the United States.",
                ),
            )

    def fake_census(self):
        raise commercial_model_module.CensusPopulationError("no key")

    def fake_llm_call(**kwargs):
        if kwargs["output_type"].__name__ == "EpidemiologyEvidenceExtraction":
            payload = kwargs["payload"]
            captured["us_population_denominator"] = payload["us_population_denominator"]
            output = commercial_model_module.EpidemiologyEvidenceExtraction(
                evidence_status="converted_prevalence_measure_selected",
                selected_population_measure=SelectedPopulationMeasure(
                    value=19_504_262.0,
                    unit="patients",
                    measure_type="converted_prevalent_patients",
                    condition="atopic dermatitis",
                    geography="United States",
                    source_type="source_derived",
                    rationale="Converted 7.3% adult prevalence using adult denominator 267181678.",
                    evidence_reference="pubmed:321; census:acs1_subject:2024:us_population_config_fallback",
                    confidence_score=7,
                    human_review_required=True,
                ),
                selected_source_id="pubmed:321",
                candidate_population_measures=(
                    commercial_model_module.EpidemiologyPopulationCandidate(
                        value=7.3,
                        unit="percent",
                        measure_type="prevalence_percent",
                        condition="atopic dermatitis",
                        geography="United States",
                        source_type="source_derived",
                        evidence_reference="pubmed:321",
                        rationale="Raw prevalence percent should stay segmentation evidence, not patient-count evidence.",
                        confidence_score=7,
                        human_review_required=True,
                    ),
                ),
                confidence_score=7,
            )
        else:
            assert isinstance(kwargs["payload"], CommercialInputBundle)
            assert kwargs["payload"].us_population_denominator["total_us_population"] == 340110990
            assert kwargs["payload"].disease_population_evidence[0]["value"] == 19_504_262.0
            assert all(item.get("unit") != "percent" for item in kwargs["payload"].disease_population_evidence)
            triplet = CommercialAssumptionTriplet(
                low=1.0,
                base=1.0,
                high=1.0,
                source_type="source_derived",
                rationale="Use converted denominator directly.",
                evidence_reference="pubmed:321",
                confidence_score=7,
                human_review_required=True,
            )
            output = MarketSizingInterpretation(
                calculable=True,
                selected_market_archetype="chronic_specialty_prevalence",
                market_basis="prevalence_stock",
                selected_population_measure=SelectedPopulationMeasure(
                    value=19_504_262.0,
                    unit="patients",
                    measure_type="converted_prevalent_patients",
                    condition="atopic dermatitis",
                    geography="United States",
                    source_type="source_derived",
                    rationale="Selected converted prevalence denominator.",
                    evidence_reference="pubmed:321; census:acs1_subject:2024:us_population_config_fallback",
                    confidence_score=7,
                    human_review_required=True,
                ),
                yearly_eligible_patient_logic="Use converted disease-population denominator directly.",
                diagnosed_fraction=triplet,
                treated_fraction=triplet,
                eligibility_fraction=triplet,
                commercially_addressable_fraction=triplet,
                rationale="Source-backed prevalence conversion was available.",
                confidence_score=7,
                key_evidence_used=("pubmed:321", "census:acs1_subject:2024:us_population_config_fallback"),
            )
        return SimpleNamespace(output=output, trace=SimpleNamespace(agent_name=kwargs["agent_name"]), trace_metadata={"agent_name": kwargs["agent_name"]})

    monkeypatch.setattr(commercial_model_module, "PubMedClient", FakePubMedClient)
    monkeypatch.setattr(commercial_model_module.CensusPopulationClient, "get_latest_us_population", fake_census)
    monkeypatch.setattr(commercial_model_module, "run_structured_llm_call", fake_llm_call)

    output = build_commercial_model_with_trace(
        annual_patients=None,
        peak_penetration=None,
        gross_to_net=None,
        pricing=PricingOutput(
            annual_wac=1000.0,
            wac_value=1000.0,
            dosing_summary="source-backed dosing",
            source_ids=("wac:fixture",),
            confidence=0.8,
        ),
        trial=ClinicalTrialRecord(
            nct_id="NCT12345678",
            brief_title="Trial in patients with atopic dermatitis",
            conditions=("Atopic Dermatitis",),
            source_id="ctgov:NCT12345678",
        ),
        asset=AssetIdentityOutput(
            nct_id="NCT12345678",
            asset_name="Example",
            normalized_indication="atopic dermatitis",
            source_ids=("ctgov:NCT12345678",),
            confidence=0.8,
        ),
        run_id="commercial-market-prevalence-fixture",
    )

    assert output.output.calculable
    assert output.output.annual_patients == 19_504_262.0
    assert "census:acs1_subject:2024:us_population_config_fallback" in output.output.source_ids
    assert output.output.commercial_input_bundle_summary["us_population_denominator"]["human_review_required"] is True
    assert captured["us_population_denominator"]["adult_population"] == 267181678


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
