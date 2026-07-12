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


def test_search_trials_normalizes_structured_design_and_arm_fields() -> None:
    payload = {
        "studies": [
            {
                "hasResults": False,
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT01234567", "briefTitle": "Structured design trial"},
                    "statusModule": {},
                    "designModule": {
                        "studyType": "INTERVENTIONAL",
                        "allocation": "RANDOMIZED",
                        "interventionModel": "PARALLEL",
                        "maskingInfo": {"masking": "DOUBLE"},
                        "observationalModel": "COHORT",
                        "numberOfArms": 2,
                    },
                    "conditionsModule": {"conditions": ["Glioblastoma"]},
                    "armsInterventionsModule": {
                        "armGroups": [
                            {
                                "label": "Experimental Arm",
                                "type": "EXPERIMENTAL",
                                "description": "Examplemab arm",
                                "interventionNames": ["Drug: Examplemab"],
                            },
                            {
                                "label": "Placebo Control",
                                "type": "PLACEBO_COMPARATOR",
                                "description": "Placebo arm",
                                "interventionNames": ["Drug: Placebo"],
                            },
                        ],
                        "interventions": [
                            {
                                "name": "Examplemab",
                                "type": "DRUG",
                                "armGroupLabels": ["Experimental Arm"],
                            },
                            {
                                "name": "Placebo",
                                "type": "DRUG",
                                "armGroupLabels": ["Placebo Control"],
                            },
                        ],
                    },
                },
            }
        ]
    }

    client = ClinicalTrialsGovClient(
        client=httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    )
    result = client.search_trials(ClinicalTrialIntelligenceInput(disease="glioblastoma"))
    trial = result.trials[0]

    assert trial.allocation == "RANDOMIZED"
    assert trial.intervention_model == "PARALLEL"
    assert trial.masking == "DOUBLE"
    assert trial.observational_model == "COHORT"
    assert trial.number_of_arms == 2
    assert trial.arm_groups[1].label == "Placebo Control"
    assert trial.arm_groups[1].intervention_names == ("Drug: Placebo",)
    assert trial.interventions[0].arm_group_labels == ("Experimental Arm",)


def test_search_trials_prefers_nested_design_info_and_sponsor_query() -> None:
    payload = {
        "studies": [
            {
                "hasResults": False,
                "derivedSection": {
                    "interventionBrowseModule": {
                        "meshes": [{"term": "Secukinumab"}],
                        "ancestors": [{"term": "Antibodies, Monoclonal"}],
                    }
                },
                "protocolSection": {
                    "identificationModule": {"nctId": "NCT01234568", "briefTitle": "Nested design trial"},
                    "statusModule": {},
                    "designModule": {
                        "studyType": "INTERVENTIONAL",
                        "designInfo": {
                            "allocation": "RANDOMIZED",
                            "interventionModel": "PARALLEL",
                            "maskingInfo": {"masking": "QUADRUPLE"},
                            "primaryPurpose": "TREATMENT",
                        },
                    },
                    "conditionsModule": {"conditions": ["Plaque Psoriasis"]},
                    "armsInterventionsModule": {
                        "armGroups": [
                            {"label": "Dose 1", "type": "EXPERIMENTAL", "interventionNames": ["Drug: AIN457"]},
                            {"label": "Placebo", "type": "PLACEBO_COMPARATOR", "interventionNames": ["Drug: Placebo"]},
                        ],
                        "interventions": [{"name": "AIN457", "type": "BIOLOGICAL"}],
                    },
                },
            }
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query.spons"] == "Novartis"
        return httpx.Response(200, json=payload)

    client = ClinicalTrialsGovClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = client.search_trials(ClinicalTrialIntelligenceInput(disease="Psoriasis", sponsor="Novartis"))
    trial = result.trials[0]

    assert trial.allocation == "RANDOMIZED"
    assert trial.intervention_model == "PARALLEL"
    assert trial.masking == "QUADRUPLE"
    assert trial.primary_purpose == "TREATMENT"
    assert trial.number_of_arms == 2
    assert trial.intervention_browse_terms == ("Secukinumab", "Antibodies, Monoclonal")


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
