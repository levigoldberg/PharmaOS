"""RxNorm normalization tool."""

from __future__ import annotations

from typing import Any

import httpx

from pharma_os.schemas import RxNormMatch, SourceMetadata


RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


class RxNormError(RuntimeError):
    """Raised when RxNorm retrieval fails."""


class RxNormClient:
    """Best-effort RxNorm client using httpx."""

    def __init__(self, timeout: float = 15.0, client: httpx.Client | None = None) -> None:
        self.timeout = timeout
        self.client = client or httpx.Client(timeout=timeout)

    def normalize(self, drug_name: str) -> tuple[RxNormMatch | None, SourceMetadata]:
        """Return a normalized RxNorm match and source metadata."""

        source = SourceMetadata(
            source_id=f"rxnorm:{_slug(drug_name)}",
            title=f"RxNorm normalization for {drug_name}",
            url=RXNORM_BASE_URL,
            authors=("National Library of Medicine",),
            provenance="RxNorm REST API rxcui/allrelated lookup",
            source_type="drug_normalization",
            version="REST",
        )
        payload = self._get_json("rxcui.json", name=drug_name, search="2")
        ids = payload.get("idGroup", {}).get("rxnormId") or []
        if not ids:
            return None, source
        rxcui = str(ids[0])
        aliases = self._aliases(rxcui)
        return (
            RxNormMatch(
                matched_name=aliases[0] if aliases else drug_name,
                rxcui=rxcui,
                aliases=tuple(aliases),
                source_id=source.source_id,
            ),
            source,
        )

    def _get_json(self, path: str, **params: str) -> dict[str, Any]:
        try:
            response = self.client.get(f"{RXNORM_BASE_URL}/{path}", params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
            raise RxNormError(f"RxNorm request failed: {exc.__class__.__name__}") from exc
        if not isinstance(payload, dict):
            raise RxNormError("RxNorm returned malformed JSON")
        return payload

    def _aliases(self, rxcui: str) -> list[str]:
        payload = self._get_json(f"rxcui/{rxcui}/allrelated.json")
        aliases: list[str] = []
        seen: set[str] = set()
        for group in payload.get("allRelatedGroup", {}).get("conceptGroup") or []:
            for concept in group.get("conceptProperties") or []:
                name = concept.get("name")
                if not name:
                    continue
                key = str(name).casefold()
                if key not in seen:
                    seen.add(key)
                    aliases.append(str(name))
                if len(aliases) >= 20:
                    return aliases
        return aliases


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "unknown"
