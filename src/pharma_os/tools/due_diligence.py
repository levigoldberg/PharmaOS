"""Compatibility facade for Agent 4 due-diligence tools.

The implementation lives in narrower modules aligned to the data layers that
exist today: CT.gov/RxNorm asset identity, Lens IP, local PoS, pricing/openFDA,
commercial model, and rNPV.
"""

from __future__ import annotations

from pharma_os.components.due_diligence_sections import (
    build_asset_memo,
    build_clinical_evidence_summary,
    build_competitive_landscape_summary,
    build_patent_loe_review,
    build_red_flags,
    build_safety_label_summary,
)
from pharma_os.tools._due_diligence_common import DEFAULT_POS_WORKBOOK, DEFAULT_WAC_DATA, LENS_PATENT_URL, OPENFDA_LABEL_URL
from pharma_os.tools.asset_identity import resolve_asset_identity
from pharma_os.tools.commercial_model import build_commercial_model
from pharma_os.tools.patents_lens import search_patent_exclusivity
from pharma_os.tools.pos import lookup_pos
from pharma_os.tools.pricing import lookup_pricing
from pharma_os.tools.rnpv import build_rnpv


__all__ = [
    "DEFAULT_POS_WORKBOOK",
    "DEFAULT_WAC_DATA",
    "LENS_PATENT_URL",
    "OPENFDA_LABEL_URL",
    "resolve_asset_identity",
    "search_patent_exclusivity",
    "lookup_pos",
    "lookup_pricing",
    "build_commercial_model",
    "build_rnpv",
    "build_clinical_evidence_summary",
    "build_competitive_landscape_summary",
    "build_safety_label_summary",
    "build_patent_loe_review",
    "build_red_flags",
    "build_asset_memo",
]
