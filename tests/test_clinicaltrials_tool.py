from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from pharma_os.schemas import ClinicalTrialIntelligenceInput
from pharma_os.tools.clinicaltrials import ClinicalTrialsGovClient, ClinicalTrialsGovError


FIXTURE = Path(__file__).parent / "fixtures" / "clinicaltrials_search_glioblastoma.json"


def test_search_trials_normalizes_fixture() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query.cond"] == "glioblastoma"
        return httpx.Response(200, json=payload)

    client = ClinicalTrialsGovClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = client.search_trials(
        ClinicalTrialIntelligenceInput(disease="glioblastoma", target="EGFR", limit=10)
    )

    assert len(result.trials) == 2
    assert result.trials[0].nct_id == "NCT01234567"
    assert result.trials[0].lead_sponsor is not None
    assert result.trials[0].lead_sponsor.name == "Example Bio"
    assert result.trials[0].enrollment_count == 42
    assert result.trials[0].primary_endpoints[0].measure == "Progression-free survival"
    assert result.sources[0].source_id == "ctgov:NCT01234567"


def test_fetch_trial_rejects_missing_protocol_section() -> None:
    client = ClinicalTrialsGovClient(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    )
    with pytest.raises(ClinicalTrialsGovError, match="protocolSection"):
        client.fetch_trial("NCT01234567")


def test_fetch_trial_handles_not_found() -> None:
    client = ClinicalTrialsGovClient(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(404, json={})))
    )
    with pytest.raises(ClinicalTrialsGovError, match="No ClinicalTrials.gov study"):
        client.fetch_trial("NCT01234567")
