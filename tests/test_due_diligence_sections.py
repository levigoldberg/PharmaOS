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
from pharma_os.tools.pos import lookup_pos
from pharma_os.tools.pricing import lookup_pricing


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


def test_pubmed_client_retries_transient_failures_and_normalizes_articles() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            calls["count"] += 1
            if calls["count"] == 1:
                return httpx.Response(429, headers={"retry-after": "0"})
            return httpx.Response(200, json={"esearchresult": {"idlist": ["123"]}})
        return httpx.Response(
            200,
            text="""
            <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
            <Article><ArticleTitle>Atopic dermatitis prevalence in the United States</ArticleTitle>
            <Journal><Title>Example Journal</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
            <Abstract><AbstractText>Prevalence evidence.</AbstractText></Abstract></Article>
            </MedlineCitation></PubmedArticle></PubmedArticleSet>
            """,
        )

    client = PubMedClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        min_interval=0,
        max_retries=2,
    )

    articles = client.search("atopic dermatitis prevalence United States", max_results=1)

    assert calls["count"] == 2
    assert articles[0].pmid == "123"
    assert articles[0].title == "Atopic dermatitis prevalence in the United States"


def test_pubmed_client_falls_back_from_rejected_quoted_query() -> None:
    terms: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            term = str(request.url.params.get("term"))
            terms.append(term)
            if '"' in term:
                return httpx.Response(400, text="quoted query rejected")
            return httpx.Response(200, json={"esearchresult": {"idlist": ["456"]}})
        return httpx.Response(
            200,
            text="""
            <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>456</PMID>
            <Article><ArticleTitle>Psoriasis epidemiology in the United States</ArticleTitle>
            <AuthorList><Author><ForeName>Alex</ForeName><LastName>Smith</LastName></Author></AuthorList>
            <Journal><Title>Example Journal</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal>
            <Abstract><AbstractText>US prevalence evidence.</AbstractText></Abstract></Article>
            </MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1000/example</ArticleId></ArticleIdList></PubmedData>
            </PubmedArticle></PubmedArticleSet>
            """,
        )

    client = PubMedClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        min_interval=0,
        max_retries=1,
    )

    articles = client.search('"psoriasis" prevalence United States', max_results=1)

    assert terms == ['"psoriasis" prevalence United States', "psoriasis prevalence United States"]
    assert articles[0].pmid == "456"
    assert articles[0].authors == ("Alex Smith",)
    assert articles[0].doi == "10.1000/example"


def test_pubmed_client_skips_single_malformed_article() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": {"idlist": ["bad", "123"]}})
        return httpx.Response(
            200,
            text="""
            <PubmedArticleSet>
              <PubmedArticle><MedlineCitation><PMID>bad</PMID><Article /></MedlineCitation></PubmedArticle>
              <PubmedArticle><MedlineCitation><PMID>123</PMID>
              <Article><ArticleTitle>Usable epidemiology article</ArticleTitle></Article>
              </MedlineCitation></PubmedArticle>
            </PubmedArticleSet>
            """,
        )

    client = PubMedClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        min_interval=0,
        max_retries=1,
    )

    articles = client.search("epidemiology", max_results=2)

    assert [article.pmid for article in articles] == ["123"]


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


def test_lens_patent_search_estimates_loe_from_candidate_dates(monkeypatch) -> None:
    monkeypatch.setenv("LENS_API_TOKEN", "test-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "lens_id": "123-456",
                        "biblio": {
                            "invention_title": ["Example TYK2 inhibitor"],
                            "application_reference": {"date": "2024-09-20"},
                        },
                        "jurisdiction": "US",
                        "date_published": "2026-04-21",
                        "legal_status": "ACTIVE",
                    }
                ]
            },
        )

    patent, _sources = search_patent_exclusivity(
        AssetIdentityOutput(nct_id="NCT12345678", asset_name="Examplemab", sponsor="Example Bio", aliases=("EX-001",), confidence=0.9),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    review = build_patent_loe_review(patent)

    assert patent.estimated_loe_year == 2044
    assert not any(flag.flag_id == "patent-loe-review-required" for flag in patent.missing_data_flags)
    assert "Source-derived LOE year is 2044" in review.review_summary


def test_pos_maps_sle_to_autoimmune_workbook_row() -> None:
    pos, source = lookup_pos(
        ClinicalTrialRecord(
            nct_id="NCT05966480",
            phases=("PHASE2",),
            conditions=("SLE",),
            interventions=(TrialIntervention(name="Envudeucitinib", type="DRUG"),),
            lead_sponsor=TrialSponsor(name="Alumis Inc"),
            source_id="ctgov:NCT05966480",
        ),
        AssetIdentityOutput(
            nct_id="NCT05966480",
            asset_name="Envudeucitinib",
            aliases=("ESK-001",),
            normalized_indication="SLE",
            modality="small molecule",
            confidence=0.7,
        ),
    )

    assert source.source_id.startswith("pos_workbook:")
    assert pos.current_phase == "Phase II"
    assert pos.disease_area == "Autoimmune"
    assert pos.lookup_key == "Disease Area|Autoimmune|Phase II"
    assert pos.probability_of_success is not None
    assert not pos.missing_data_flags


def test_pricing_uses_sle_analog_wac_and_openfda_dosing(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OS_AGENTS_DISABLED", "true")

    def handler(request: httpx.Request) -> httpx.Response:
        query = str(request.url.params.get("search", ""))
        if "Benlysta" not in query and "belimumab" not in query:
            return httpx.Response(404, json={})
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "dosage_and_administration": [
                            "For adult patients with active systemic lupus erythematosus, administer 200 mg subcutaneously once weekly."
                        ]
                    }
                ]
            },
        )

    pricing, sources = lookup_pricing(
        AssetIdentityOutput(
            nct_id="NCT05966480",
            asset_name="Envudeucitinib",
            aliases=("ESK-001",),
            normalized_indication="SLE",
            modality="small molecule",
            confidence=0.7,
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert pricing.wac_value == 1210.63
    assert pricing.annual_wac == 62952.76
    assert pricing.matched_product and "Benlysta" in pricing.matched_product
    assert pricing.dosing_summary and "once weekly" in pricing.dosing_summary
    assert not any(flag.flag_id == "pricing-no-wac-match" for flag in pricing.missing_data_flags)
    assert "wac:california-wac-data-xlsx" in {source.source_id for source in sources}
    assert "openfda_label:benlysta" in {source.source_id for source in sources}


def test_pricing_uses_atopic_dermatitis_source_constrained_analog(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OS_AGENTS_DISABLED", "true")

    def handler(request: httpx.Request) -> httpx.Response:
        query = str(request.url.params.get("search", ""))
        if "Cibinqo" not in query and "abrocitinib" not in query:
            return httpx.Response(404, json={})
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "dosage_and_administration": [
                            "Recommended dosage for adults with atopic dermatitis is CIBINQO 100 mg orally once daily."
                        ]
                    }
                ]
            },
        )

    pricing, sources = lookup_pricing(
        AssetIdentityOutput(
            nct_id="NCT07011706",
            asset_name="ATI-045",
            aliases=("ATI-045",),
            normalized_indication="Atopic Dermatitis",
            modality="unknown",
            confidence=0.4,
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert pricing.wac_value == 6167.2
    assert pricing.annual_wac == 75034.27
    assert pricing.matched_product and "Cibinqo" in pricing.matched_product
    assert pricing.matched_product and "pricing analog" in pricing.matched_product
    assert pricing.annualization_details["units_per_package"] == 30
    assert not pricing.missing_data_flags
    assert "wac:california-wac-data-xlsx" in {source.source_id for source in sources}
    assert "openfda_label:cibinqo" in {source.source_id for source in sources}


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
