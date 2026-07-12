"""Thin PubMed E-utilities client for Agent 4 evidence retrieval."""

from __future__ import annotations

import os
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx

from pharma_os.schemas import SourceMetadata


class PubMedError(RuntimeError):
    """Raised when PubMed cannot return normalized article metadata."""


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    journal: str | None = None
    year: int | None = None
    authors: tuple[str, ...] = ()
    doi: str | None = None
    abstract_snippet: str | None = None

    @property
    def source_id(self) -> str:
        return f"pubmed:{self.pmid}"

    def source(self) -> SourceMetadata:
        return SourceMetadata(
            source_id=self.source_id,
            title=self.title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/",
            authors=self.authors,
            provenance="PubMed E-utilities esearch/efetch",
            source_type="literature",
            version=str(self.year) if self.year else None,
        )


class PubMedClient:
    """Small PubMed E-utilities client."""

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        email: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
        min_interval: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("NCBI_API_KEY") or "").strip() or None
        self.email = (email or os.getenv("NCBI_EMAIL") or "").strip() or None
        headers = {"User-Agent": "pharma_os/0.1"}
        if self.email:
            headers["User-Agent"] = f"pharma_os/0.1 ({self.email})"
        self.client = client or httpx.Client(timeout=timeout, headers=headers)
        self.timeout = timeout
        self.min_interval = min_interval if min_interval is not None else (0.1 if self.api_key else 0.34)
        self.max_retries = max_retries if max_retries is not None else _env_int("PHARMA_OS_PUBMED_MAX_RETRIES", 3, minimum=1, maximum=8)
        self.retry_initial_delay = _env_float("PHARMA_OS_PUBMED_RETRY_INITIAL_DELAY_SECONDS", 0.5, minimum=0.0, maximum=30.0)
        self.retry_max_delay = _env_float("PHARMA_OS_PUBMED_RETRY_MAX_DELAY_SECONDS", 8.0, minimum=0.0, maximum=120.0)
        self._last_request_at = 0.0

    def search(self, query: str, *, max_results: int = 5) -> tuple[PubMedArticle, ...]:
        """Search PubMed and return normalized article metadata."""

        try:
            return self._search_once(query, max_results=max_results)
        except PubMedError:
            fallback_query = _fallback_query(query)
            if fallback_query == query:
                raise
            return self._search_once(fallback_query, max_results=max_results)

    def _search_once(self, query: str, *, max_results: int) -> tuple[PubMedArticle, ...]:
        common = self._common_params()
        response = self._get(
            "esearch.fcgi",
            params={**common, "db": "pubmed", "term": query, "retmode": "json", "retmax": str(max_results), "sort": "relevance"},
        )
        try:
            payload = response.json()
            ids = (payload.get("esearchresult") or {}).get("idlist")
        except (ValueError, AttributeError) as exc:
            raise PubMedError(f"PubMed ESearch failed: {exc.__class__.__name__}") from exc
        if not isinstance(ids, list):
            raise PubMedError("PubMed ESearch response is missing idlist")
        ids = [str(value) for value in ids[:max_results] if value]
        if not ids:
            return ()
        fetch = self._get(
            "efetch.fcgi",
            params={**common, "db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        )
        try:
            root = ET.fromstring(fetch.text)
        except ET.ParseError as exc:
            raise PubMedError(f"PubMed EFetch failed: {exc.__class__.__name__}") from exc
        parsed: list[PubMedArticle] = []
        malformed_count = 0
        for article in root.findall(".//PubmedArticle"):
            try:
                parsed.append(self._parse_article(article))
            except PubMedError:
                malformed_count += 1
        if not parsed and malformed_count:
            raise PubMedError("PubMed EFetch returned articles but none could be normalized")
        return tuple(parsed)

    def _common_params(self) -> dict[str, str]:
        common = {"tool": "pharma_os"}
        if self.api_key:
            common["api_key"] = self.api_key
        if self.email:
            common["email"] = self.email
        return common

    def _get(self, path: str, *, params: dict[str, str]) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._respect_rate_limit()
            try:
                response = self.client.get(
                    f"{self.base_url}/{path}",
                    params=params,
                    timeout=self.timeout,
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt >= self.max_retries or not _is_transient_httpx_error(exc):
                    break
                time.sleep(_retry_delay(exc, attempt, initial=self.retry_initial_delay, max_delay=self.retry_max_delay))
        raise PubMedError(f"PubMed request failed: {last_exc.__class__.__name__ if last_exc else 'UnknownError'}") from last_exc

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _parse_article(self, element: ET.Element) -> PubMedArticle:
        pmid = _text(element.find("./MedlineCitation/PMID"))
        title = _text(element.find("./MedlineCitation/Article/ArticleTitle"))
        if not pmid or not title:
            raise PubMedError("PubMed article is missing PMID or title")
        journal = _text(element.find("./MedlineCitation/Article/Journal/Title"))
        year_text = (
            _text(element.find("./MedlineCitation/Article/ArticleDate/Year"))
            or _text(element.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year"))
            or _text(element.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/MedlineDate"))
        )
        match = re.search(r"(?:19|20)\d{2}", year_text or "")
        authors: list[str] = []
        for author in element.findall("./MedlineCitation/Article/AuthorList/Author"):
            collective = _text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue
            last = _text(author.find("LastName"))
            given = _text(author.find("ForeName")) or _text(author.find("Initials"))
            name = " ".join(part for part in (given, last) if part)
            if name:
                authors.append(name)
        doi = None
        for article_id in element.findall("./PubmedData/ArticleIdList/ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = _text(article_id)
                break
        if doi is None:
            for location_id in element.findall("./MedlineCitation/Article/ELocationID"):
                if location_id.attrib.get("EIdType") == "doi":
                    doi = _text(location_id)
                    break
        abstract = " ".join(
            part
            for item in element.findall("./MedlineCitation/Article/Abstract/AbstractText")
            if (part := _text(item))
        )
        return PubMedArticle(
            pmid=pmid,
            title=title,
            journal=journal,
            year=int(match.group()) if match else None,
            authors=tuple(authors),
            doi=doi,
            abstract_snippet=abstract or None,
        )


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None


def _fallback_query(query: str) -> str:
    """Return a simpler PubMed query when ESearch rejects a quoted/symbol-heavy term."""

    fallback = query.replace('"', " ")
    fallback = re.sub(r"\s+", " ", fallback).strip()
    return fallback or query


def _is_transient_httpx_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False


def _retry_delay(exc: httpx.HTTPError, attempt: int, *, initial: float, max_delay: float) -> float:
    if isinstance(exc, httpx.HTTPStatusError):
        retry_after = exc.response.headers.get("retry-after")
        if retry_after:
            try:
                return max(0.0, min(max_delay, float(retry_after)))
            except ValueError:
                pass
    base = min(max_delay, initial * (2 ** max(0, attempt - 1)))
    return base + random.uniform(0.0, min(1.0, base))


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))
