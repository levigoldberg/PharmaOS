"""ClinicalTrials.gov API v2 deterministic tools."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import httpx

from pharma_os.schemas import (
    ClinicalTrialIntelligenceInput,
    ClinicalTrialRecord,
    ClinicalTrialsSearchResult,
    SourceMetadata,
    TrialArmGroup,
    TrialEndpoint,
    TrialIntervention,
    TrialLocation,
    TrialSponsor,
)


NCT_PATTERN = re.compile(r"^NCT\d{8}$", re.IGNORECASE)
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


class ClinicalTrialsGovError(RuntimeError):
    """Raised when ClinicalTrials.gov cannot return usable data."""


class ClinicalTrialsGovClient:
    """Thin ClinicalTrials.gov API v2 client."""

    def __init__(self, timeout: float = 20.0, client: httpx.Client | None = None) -> None:
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)

    @staticmethod
    def normalize_nct_id(nct_id: str) -> str:
        """Normalize and validate an NCT identifier."""

        normalized = nct_id.strip().upper()
        if not NCT_PATTERN.fullmatch(normalized):
            raise ValueError("NCT ID must match NCT followed by exactly 8 digits")
        return normalized

    def search_trials(self, input_data: ClinicalTrialIntelligenceInput) -> ClinicalTrialsSearchResult:
        """Search ClinicalTrials.gov and normalize study records."""

        params = _search_params(input_data)
        api_url = f"{BASE_URL}?{urlencode(params)}"
        try:
            response = self.client.get(BASE_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise ClinicalTrialsGovError(f"ClinicalTrials.gov returned HTTP {exc.response.status_code}") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise ClinicalTrialsGovError(f"ClinicalTrials.gov search failed: {exc.__class__.__name__}") from exc
        studies = payload.get("studies") if isinstance(payload, dict) else None
        if studies is None:
            raise ClinicalTrialsGovError("ClinicalTrials.gov response is missing studies")
        if not isinstance(studies, list):
            raise ClinicalTrialsGovError("ClinicalTrials.gov studies field is malformed")
        records = tuple(_normalize_study(study) for study in studies[: input_data.limit])
        sources = tuple(_source_for_record(record) for record in records)
        return ClinicalTrialsSearchResult(
            query=input_data,
            trials=records,
            sources=sources,
            retrieved_at=datetime.now(timezone.utc),
            api_url=api_url,
            errors=(),
        )

    def fetch_trial(self, nct_id: str) -> ClinicalTrialRecord:
        """Fetch and normalize one ClinicalTrials.gov study by NCT ID."""

        normalized = self.normalize_nct_id(nct_id)
        url = f"{BASE_URL}/{normalized}"
        try:
            response = self.client.get(url, timeout=self.timeout)
            if response.status_code == 404:
                raise ClinicalTrialsGovError(f"No ClinicalTrials.gov study found for {normalized}")
            response.raise_for_status()
            payload = response.json()
        except ClinicalTrialsGovError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ClinicalTrialsGovError(f"ClinicalTrials.gov returned HTTP {exc.response.status_code}") from exc
        except (httpx.RequestError, ValueError) as exc:
            raise ClinicalTrialsGovError(f"ClinicalTrials.gov fetch failed: {exc.__class__.__name__}") from exc
        return _normalize_study(payload)


def search_trials(input_data: ClinicalTrialIntelligenceInput) -> ClinicalTrialsSearchResult:
    """Module-level convenience wrapper for agent tools."""

    return ClinicalTrialsGovClient().search_trials(input_data)


def fetch_trial(nct_id: str) -> ClinicalTrialRecord:
    """Module-level convenience wrapper for agent tools."""

    return ClinicalTrialsGovClient().fetch_trial(nct_id)


def _search_params(input_data: ClinicalTrialIntelligenceInput) -> dict[str, str]:
    terms = [input_data.target, input_data.phase]
    query_term = " ".join(term for term in terms if term)
    params = {
        "query.cond": input_data.disease,
        "pageSize": str(input_data.limit),
        "format": "json",
    }
    if input_data.drug:
        params["query.intr"] = input_data.drug
    if query_term:
        params["query.term"] = query_term
    return params


def _normalize_study(payload: dict[str, Any]) -> ClinicalTrialRecord:
    protocol = payload.get("protocolSection")
    if not isinstance(protocol, dict):
        raise ClinicalTrialsGovError("ClinicalTrials.gov response is missing protocolSection")
    identification = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    design = protocol.get("designModule") or {}
    sponsor_module = protocol.get("sponsorCollaboratorsModule") or {}
    conditions_module = protocol.get("conditionsModule") or {}
    arms_module = protocol.get("armsInterventionsModule") or {}
    outcomes_module = protocol.get("outcomesModule") or {}
    eligibility_module = protocol.get("eligibilityModule") or {}
    contacts_locations_module = protocol.get("contactsLocationsModule") or {}
    nct_id = identification.get("nctId")
    if not isinstance(nct_id, str):
        raise ClinicalTrialsGovError("ClinicalTrials.gov response is missing nctId")
    normalized_nct = ClinicalTrialsGovClient.normalize_nct_id(nct_id)
    enrollment = design.get("enrollmentInfo") or {}
    return ClinicalTrialRecord(
        nct_id=normalized_nct,
        brief_title=identification.get("briefTitle"),
        official_title=identification.get("officialTitle"),
        overall_status=status.get("overallStatus"),
        phases=tuple(str(value) for value in design.get("phases") or ()),
        study_type=design.get("studyType"),
        allocation=design.get("allocation"),
        intervention_model=design.get("interventionModel"),
        masking=_masking_value(design),
        observational_model=design.get("observationalModel"),
        number_of_arms=_int(design.get("numberOfArms")),
        conditions=tuple(str(value) for value in conditions_module.get("conditions") or ()),
        interventions=tuple(_intervention(item) for item in arms_module.get("interventions") or ()),
        arm_groups=tuple(_arm_group(item) for item in arms_module.get("armGroups") or ()),
        lead_sponsor=_sponsor(sponsor_module.get("leadSponsor")),
        collaborators=tuple(
            sponsor
            for item in sponsor_module.get("collaborators") or ()
            if (sponsor := _sponsor(item)) is not None
        ),
        enrollment_count=_int(enrollment.get("count")),
        enrollment_type=enrollment.get("type"),
        start_date=_date_value(status, "startDateStruct"),
        primary_completion_date=_date_value(status, "primaryCompletionDateStruct"),
        completion_date=_date_value(status, "completionDateStruct"),
        results_available=bool(payload.get("hasResults", False)),
        primary_endpoints=tuple(
            _endpoint(item, "primary") for item in outcomes_module.get("primaryOutcomes") or ()
        ),
        secondary_endpoints=tuple(
            _endpoint(item, "secondary") for item in outcomes_module.get("secondaryOutcomes") or ()
        ),
        locations=tuple(_location(item) for item in contacts_locations_module.get("locations") or ()),
        eligibility_criteria=eligibility_module.get("eligibilityCriteria"),
        minimum_age=eligibility_module.get("minimumAge"),
        maximum_age=eligibility_module.get("maximumAge"),
        sex=eligibility_module.get("sex"),
        source_id=f"ctgov:{normalized_nct}",
    )


def _source_for_record(record: ClinicalTrialRecord) -> SourceMetadata:
    return SourceMetadata(
        source_id=record.source_id,
        title=record.brief_title or record.official_title or record.nct_id,
        url=f"https://clinicaltrials.gov/study/{record.nct_id}",
        authors=tuple(
            sponsor.name
            for sponsor in (record.lead_sponsor, *record.collaborators)
            if sponsor is not None
        ),
        retrieved_at=datetime.now(timezone.utc),
        provenance="ClinicalTrials.gov API v2 protocolSection",
        source_type="clinical_trial_registry",
        version="v2",
    )


def _intervention(item: dict[str, Any]) -> TrialIntervention:
    return TrialIntervention(
        name=str(item.get("name") or "Unknown intervention"),
        type=item.get("type"),
        description=item.get("description"),
        other_names=tuple(str(value) for value in item.get("otherNames") or ()),
        arm_group_labels=tuple(str(value) for value in item.get("armGroupLabels") or ()),
    )


def _arm_group(item: dict[str, Any]) -> TrialArmGroup:
    return TrialArmGroup(
        label=str(item.get("label") or "Unknown arm"),
        type=item.get("type"),
        description=item.get("description"),
        intervention_names=tuple(str(value) for value in item.get("interventionNames") or ()),
    )


def _masking_value(design: dict[str, Any]) -> str | None:
    masking_info = design.get("maskingInfo")
    if isinstance(masking_info, dict):
        masking = masking_info.get("masking")
        if masking:
            return str(masking)
    masking = design.get("masking")
    return str(masking) if masking else None


def _sponsor(item: dict[str, Any] | None) -> TrialSponsor | None:
    if not isinstance(item, dict) or not item.get("name"):
        return None
    return TrialSponsor(name=str(item["name"]), sponsor_class=item.get("class"))


def _endpoint(item: dict[str, Any], endpoint_type: str) -> TrialEndpoint:
    return TrialEndpoint(
        measure=str(item.get("measure") or "Unspecified endpoint"),
        time_frame=item.get("timeFrame"),
        description=item.get("description"),
        endpoint_type=endpoint_type,  # type: ignore[arg-type]
    )


def _location(item: dict[str, Any]) -> TrialLocation:
    return TrialLocation(
        facility=item.get("facility"),
        city=item.get("city"),
        state=item.get("state"),
        country=item.get("country"),
        status=item.get("status"),
    )


def _date_value(module: dict[str, Any], key: str) -> str | None:
    value = module.get(key)
    return value.get("date") if isinstance(value, dict) else None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
