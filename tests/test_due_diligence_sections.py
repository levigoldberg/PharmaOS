from __future__ import annotations

import httpx

from pharma_os.components.due_diligence_sections import (
    build_asset_memo,
    build_clinical_evidence_summary,
    build_competitive_landscape_summary,
    build_patent_loe_review,
    build_red_flags,
    build_safety_label_summary,
)
from pharma_os.schemas import (
    ApprovalLikelihoodProxy,
    AssetIdentityOutput,
    ClinicalOutcomePredictionInput,
    ClinicalOutcomePredictionOutput,
    ClinicalRiskSummary,
    ClinicalTrialRecord,
    CommercialModelOutput,
    ComparatorBenchmarkBundle,
    EndpointRiskAssessment,
    EnrollmentDurationRisk,
    EvidenceClaim,
    FailureModeClassification,
    HistoricalPoSEstimate,
    LabelExpansionClinicalRationale,
    PatentExclusivityOutput,
    PatentLOEReview,
    PoSOutput,
    PricingOutput,
    RNPVOutput,
    SafetyContext,
    SafetyLabelSummary,
    SourceAvailabilityReport,
    SourceMetadata,
    TrialDesignFeatures,
    TrialIdentity,
    TrialIntervention,
    TrialSponsor,
)
from pharma_os.tools.pubmed import PubMedClient
from pharma_os.tools.patents_lens import search_patent_exclusivity


def _source(source_id: str) -> SourceMetadata:
    return SourceMetadata(source_id=source_id, title=source_id, provenance="test", source_type="fixture")


def _trial() -> ClinicalTrialRecord:
    return ClinicalTrialRecord(
        nct_id="NCT12345678",
        brief_title="Example trial",
        overall_status="RECRUITING",
        phases=("PHASE2",),
        conditions=("Glioblastoma",),
        interventions=(TrialIntervention(name="Examplemab", type="DRUG"),),
        lead_sponsor=TrialSponsor(name="Example Bio"),
        enrollment_count=42,
        source_id="ctgov:NCT12345678",
    )


def _agent3() -> ClinicalOutcomePredictionOutput:
    return ClinicalOutcomePredictionOutput(
        output_id="cop-output",
        run_id="cop-run",
        input=ClinicalOutcomePredictionInput(nct_id="NCT12345678"),
        trial_identity=TrialIdentity(nct_id="NCT12345678", phases=("PHASE2",), conditions=("Glioblastoma",), sponsor="Example Bio", source_ids=("ctgov:NCT12345678",)),
        asset_identity=AssetIdentityOutput(nct_id="NCT12345678", asset_name="Examplemab", normalized_indication="glioblastoma", sponsor="Example Bio", source_ids=("ctgov:NCT12345678",), confidence=0.9),
        trial_design_features=TrialDesignFeatures(primary_endpoint_count=1, source_ids=("ctgov:NCT12345678",)),
        endpoint_risk_assessment=EndpointRiskAssessment(risk_level="low", rationale="fixture", source_ids=("ctgov:NCT12345678",), confidence=0.8),
        enrollment_duration_risk=EnrollmentDurationRisk(risk_level="low", rationale="fixture", source_ids=("ctgov:NCT12345678",), confidence=0.8),
        comparator_benchmarking=ComparatorBenchmarkBundle(
            matched_public_trials_count=2,
            comparator_trial_ids=("NCT11111111", "NCT22222222"),
            benchmark_summary="Agent 3 comparator summary.",
            status_summary="COMPLETED: 2",
            source_ids=("ctgov:NCT11111111", "ctgov:NCT22222222"),
            confidence=0.7,
        ),
        historical_pos_estimate=HistoricalPoSEstimate(probability_of_success=0.344, current_phase="Phase II", lookup_key="Disease Area|Oncology|Phase II", assumption_type="source_derived", source_ids=("pos_workbook:fixture",), confidence=0.9),
        approval_likelihood_proxy=ApprovalLikelihoodProxy(probability=0.344, basis="fixture", assumption_type="source_derived", source_ids=("pos_workbook:fixture",), confidence=0.9),
        failure_mode_classification=FailureModeClassification(overall_risk_level="low", source_ids=("ctgov:NCT12345678",), confidence=0.7),
        safety_context=SafetyContext(label_available=False, confidence=0.0),
        label_expansion_clinical_rationale=LabelExpansionClinicalRationale(rationale="fixture", source_ids=("ctgov:NCT12345678",), confidence=0.5),
        source_availability=SourceAvailabilityReport(),
        sources=(_source("ctgov:NCT12345678"), _source("pos_workbook:fixture")),
        confidence=0.8,
    )


def test_clinical_evidence_extracts_ctgov_and_pubmed_with_sources() -> None:
    esearch = {"esearchresult": {"idlist": ["123"]}}
    efetch = """
    <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
    <Article><ArticleTitle>Examplemab glioblastoma evidence</ArticleTitle>
    <Journal><Title>Example Journal</Title><JournalIssue><PubDate><Year>2025</Year></PubDate></JournalIssue></Journal>
    <Abstract><AbstractText>Fixture abstract.</AbstractText></Abstract></Article>
    </MedlineCitation></PubmedArticle></PubmedArticleSet>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, json=esearch)
        return httpx.Response(200, text=efetch)

    summary, sources, claims = build_clinical_evidence_summary(
        run_id="run",
        trial=_trial(),
        asset=AssetIdentityOutput(nct_id="NCT12345678", asset_name="Examplemab", confidence=0.9),
        pubmed_client=PubMedClient(client=httpx.Client(transport=httpx.MockTransport(handler))),
    )

    assert summary.pubmed_article_count == 1
    assert "pubmed:123" in summary.source_ids
    assert sources[0].source_id == "pubmed:123"
    assert any(claim.source_ids == ("pubmed:123",) for claim in claims)


def test_competitive_landscape_reuses_agent3_comparators() -> None:
    summary = build_competitive_landscape_summary(_agent3())

    assert summary.comparator_trial_ids == ("NCT11111111", "NCT22222222")
    assert summary.benchmark_summary == "Agent 3 comparator summary."


def test_openfda_missing_label_creates_flag() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404, json={})))

    summary, sources = build_safety_label_summary(
        AssetIdentityOutput(nct_id="NCT12345678", asset_name="Examplemab", confidence=0.9),
        client=client,
    )

    assert summary.label_available is False
    assert summary.missing_data_flags
    assert sources == ()


def test_patent_loe_review_does_not_fake_loe() -> None:
    review = build_patent_loe_review(
        PatentExclusivityOutput(asset_name="Examplemab", searched_terms=("Examplemab",), confidence=0.2)
    )

    assert review.estimated_loe_year is None
    assert "does not invent or default LOE" in review.review_summary


def test_lens_patent_search_with_mocked_api(monkeypatch) -> None:
    monkeypatch.setenv("LENS_API_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "lens_id": "123-456",
                        "biblio": {"invention_title": ["Examplemab composition"]},
                        "jurisdiction": "US",
                        "date_published": "2024-01-01",
                        "legal_status": "ACTIVE",
                    }
                ]
            },
        )

    patent, sources = search_patent_exclusivity(
        AssetIdentityOutput(nct_id="NCT12345678", asset_name="Examplemab", sponsor="Example Bio", confidence=0.9),
        loe_year_override=2040,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    review = build_patent_loe_review(patent)

    assert patent.candidates[0].candidate_id == "123-456"
    assert review.candidate_count == 1
    assert review.estimated_loe_year == 2040
    assert sources[0].source_id == "lens:examplemab"


def test_red_flags_and_memo_assembly_cover_noncalculable_sections() -> None:
    clinical = ClinicalRiskSummary(nct_id="NCT12345678", endpoint_risk_level="high", enrollment_duration_risk_level="low", confidence=0.5)
    safety = build_safety_label_summary(AssetIdentityOutput(nct_id="NCT12345678"))[0]
    patent = build_patent_loe_review(PatentExclusivityOutput(asset_name="Examplemab", confidence=0.2))
    pricing = PricingOutput(confidence=0.0)
    commercial = CommercialModelOutput(calculable=False, confidence=0.0)
    rnpv = RNPVOutput(calculable=False, confidence=0.0)

    flags = build_red_flags(
        clinical_risk=clinical,
        safety=safety,
        patent=patent,
        pricing=pricing,
        commercial=commercial,
        rnpv=rnpv,
        missing_data_flags=(),
    )
    memo = build_asset_memo(
        run_id="run",
        asset=AssetIdentityOutput(nct_id="NCT12345678", asset_name="Examplemab", confidence=0.9),
        clinical_risk=clinical,
        evidence=build_clinical_evidence_summary(run_id="run", trial=_trial(), asset=AssetIdentityOutput(nct_id="NCT12345678", asset_name="Examplemab", confidence=0.9), pubmed_client=PubMedClient(client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"esearchresult": {"idlist": []}})))))[0],
        landscape=build_competitive_landscape_summary(_agent3()),
        safety=safety,
        patent=patent,
        pricing=pricing,
        commercial=commercial,
        rnpv=rnpv,
        red_flags=flags,
        claims=(
            EvidenceClaim(
                claim_id="claim-1",
                claim_text="Source-backed fixture claim.",
                source_ids=("ctgov:NCT12345678",),
                provenance="test",
                confidence=0.9,
                confidence_level="high",
            ),
        ),
        missing_data_flags=safety.missing_data_flags,
    )

    assert any(flag.category == "rnpv" for flag in flags)
    assert memo.requires_human_review is True
    assert "no final recommendations" in memo.summary
    assert memo.source_backed_claims == ("Source-backed fixture claim.",)
    assert memo.missing_evidence


def test_serious_safety_label_language_creates_high_red_flag() -> None:
    flags = build_red_flags(
        clinical_risk=ClinicalRiskSummary(nct_id="NCT12345678", confidence=0.5),
        safety=SafetyLabelSummary(asset_name="Examplemab", label_available=True, warnings_summary="Boxed warning: fatal reactions.", source_ids=("openfda_label:examplemab",), confidence=0.7),
        patent=PatentLOEReview(asset_name="Examplemab", review_summary="Reviewed.", estimated_loe_year=2040, candidate_count=1, confidence=0.7),
        pricing=PricingOutput(annual_wac=100.0, wac_value=100.0, dosing_summary="Dose.", source_ids=("wac:fixture",), confidence=0.8),
        commercial=CommercialModelOutput(calculable=True, confidence=0.8),
        rnpv=RNPVOutput(calculable=True, rnpv=1.0, confidence=0.8),
        missing_data_flags=(),
    )

    assert any(flag.flag_id == "safety-serious-label-warning" and flag.severity == "high" for flag in flags)
