"""Thin PubMed E-utilities client for Agent 4 evidence retrieval."""

from __future__ import annotations

import os
import re
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
    abstract_snippet: str | None = None

    @property
    def source_id(self) -> str:
        return f"pubmed:{self.pmid}"

    def source(self) -> SourceMetadata:
        return SourceMetadata(
            source_id=self.source_id,
            title=self.title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/",
            provenance="PubMed E-utilities esearch/efetch",
            source_type="literature",
            version=str(self.year) if self.year else None,
        )


class PubMedClient:
    """Small PubMed E-utilities client."""

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, *, client: httpx.Client | None = None, timeout: float = 15.0) -> None:
        self.client = client or httpx.Client(timeout=timeout)
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5) -> tuple[PubMedArticle, ...]:
        """Search PubMed and return normalized article metadata."""

        common = {"tool": "pharma_os"}
        if api_key := os.getenv("NCBI_API_KEY"):
            common["api_key"] = api_key
        response = self.client.get(
            f"{self.base_url}/esearch.fcgi",
            params={**common, "db": "pubmed", "term": query, "retmode": "json", "retmax": str(max_results), "sort": "relevance"},
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
            payload = response.json()
            ids = (payload.get("esearchresult") or {}).get("idlist")
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise PubMedError(f"PubMed ESearch failed: {exc.__class__.__name__}") from exc
        if not isinstance(ids, list):
            raise PubMedError("PubMed ESearch response is missing idlist")
        ids = [str(value) for value in ids[:max_results] if value]
        if not ids:
            return ()
        fetch = self.client.get(
            f"{self.base_url}/efetch.fcgi",
            params={**common, "db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
            timeout=self.timeout,
        )
        try:
            fetch.raise_for_status()
            root = ET.fromstring(fetch.text)
        except (httpx.HTTPError, ET.ParseError) as exc:
            raise PubMedError(f"PubMed EFetch failed: {exc.__class__.__name__}") from exc
        return tuple(self._parse_article(article) for article in root.findall(".//PubmedArticle"))

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
            abstract_snippet=abstract or None,
        )


def _text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    value = "".join(element.itertext()).strip()
    return value or None
