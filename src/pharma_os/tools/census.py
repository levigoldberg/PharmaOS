"""US Census population denominator helper for commercial market sizing."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from urllib.parse import urlencode

import httpx

from pharma_os.schemas import SourceMetadata


class CensusPopulationError(RuntimeError):
    """Raised when Census cannot return a normalized US population denominator."""


@dataclass(frozen=True)
class CensusPopulationDenominator:
    total_us_population: int
    adult_population: int | None
    pediatric_population: int | None
    source_year: int
    dataset: str
    source_url: str
    source_type: str = "structured_api"
    human_review_required: bool = False

    @property
    def source_id(self) -> str:
        return f"census:acs1_subject:{self.source_year}:us_population"

    def source(self) -> SourceMetadata:
        return SourceMetadata(
            source_id=self.source_id,
            title=f"US Census ACS 1-Year Subject Tables {self.source_year}",
            url=self.source_url,
            provenance="US Census API national ACS subject table lookup",
            source_type="structured_api",
            version=str(self.source_year),
        )

    def model_payload(self) -> dict[str, Any]:
        return {
            "total_us_population": self.total_us_population,
            "adult_population": self.adult_population,
            "pediatric_population": self.pediatric_population,
            "geography": "United States",
            "source_year": self.source_year,
            "dataset": self.dataset,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "human_review_required": self.human_review_required,
        }


class CensusPopulationClient:
    """Small Census API client for national population denominators."""

    discovery_url = "https://api.census.gov/data.json"
    api_base = "https://api.census.gov/data"
    dataset_name = "ACS 1-Year Subject Tables"
    variables = {
        "total_us_population": "S0101_C01_001E",
        "pediatric_population": "S0101_C01_022E",
        "adult_population": "S0101_C01_026E",
    }

    def __init__(self, *, api_key: str | None = None, client: httpx.Client | None = None, timeout: float = 15.0) -> None:
        self.api_key = (api_key if api_key is not None else os.getenv("CENSUS_API_KEY", "")).strip()
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self.timeout = timeout

    def get_latest_us_population(self, year: int | None = None) -> CensusPopulationDenominator:
        if not self.api_key:
            raise CensusPopulationError("CENSUS_API_KEY is required for live Census population lookup")
        source_year = year or self._configured_or_latest_year()
        endpoint = f"{self.api_base}/{source_year}/acs/acs1/subject"
        variable_names = ",".join(("NAME", *self.variables.values()))
        public_params = {"get": variable_names, "for": "us:1"}
        params = {**public_params, "key": self.api_key}
        try:
            response = self.client.get(endpoint, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise CensusPopulationError(f"Census population request failed: {exc.__class__.__name__}") from exc
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[0], list) or not isinstance(payload[1], list):
            raise CensusPopulationError("Census population response is malformed")
        row = dict(zip(payload[0], payload[1]))
        try:
            values = {name: int(row[variable]) for name, variable in self.variables.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise CensusPopulationError("Census population response is missing required variables") from exc
        return CensusPopulationDenominator(
            total_us_population=values["total_us_population"],
            adult_population=values.get("adult_population"),
            pediatric_population=values.get("pediatric_population"),
            source_year=source_year,
            dataset=self.dataset_name,
            source_url=f"{endpoint}?{urlencode(public_params)}",
        )

    def _configured_or_latest_year(self) -> int:
        configured = os.getenv("PHARMA_OS_CENSUS_YEAR")
        if configured:
            try:
                return int(configured)
            except ValueError:
                pass
        try:
            response = self.client.get(self.discovery_url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise CensusPopulationError(f"Census dataset discovery failed: {exc.__class__.__name__}") from exc
        datasets = payload.get("dataset") if isinstance(payload, dict) else None
        if not isinstance(datasets, list):
            raise CensusPopulationError("Census dataset discovery response is malformed")
        years = [
            item.get("c_vintage")
            for item in datasets
            if isinstance(item, dict)
            and item.get("c_dataset") == ["acs", "acs1", "subject"]
            and isinstance(item.get("c_vintage"), int)
        ]
        if not years:
            raise CensusPopulationError("No ACS 1-Year Subject Tables dataset was discovered")
        return max(years)
