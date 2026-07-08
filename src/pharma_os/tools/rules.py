"""Rule configuration loading for due-diligence tools."""

from __future__ import annotations

from importlib import resources
from typing import Any


DEFAULT_RULES: dict[str, Any] = {
    "modality_rules.yaml": {
        "default": "unknown",
        "rules": [
            {"id": "modality_cell_therapy", "modality": "cell therapy", "keywords": ["cell therapy", "CAR-T", "CART", "TCR-T"]},
            {"id": "modality_gene_therapy", "modality": "gene therapy", "keywords": ["gene therapy", "AAV", "adeno-associated"]},
            {"id": "modality_oligonucleotide", "modality": "oligonucleotide", "keywords": ["oligonucleotide", "siRNA", "ASO", "antisense"]},
            {"id": "modality_antibody", "modality": "antibody", "keywords": ["antibody", "monoclonal", "mAb", "bispecific"]},
            {"id": "modality_vaccine", "modality": "vaccine", "keywords": ["vaccine"]},
            {"id": "modality_small_molecule", "modality": "small molecule", "keywords": ["inhibitor", "agonist", "antagonist", "modulator", "capsule", "tablet", "oral"]},
        ],
    },
    "indication_rules.yaml": {
        "rules": [
            {"id": "indication_glioblastoma", "terms": ["Glioblastoma", "GBM"], "normalized_indication": "glioblastoma", "therapeutic_area": "oncology"},
            {"id": "indication_solid_tumor", "terms": ["Solid Tumor", "Solid Malignancy"], "normalized_indication": "advanced solid tumors", "therapeutic_area": "oncology"},
            {"id": "indication_alzheimers", "terms": ["Alzheimer Disease", "Alzheimer's Disease"], "normalized_indication": "Alzheimer's disease", "therapeutic_area": "neurology"},
            {"id": "indication_parkinsons", "terms": ["Parkinson Disease", "Parkinson's Disease"], "normalized_indication": "Parkinson's disease", "therapeutic_area": "neurology"},
            {"id": "indication_dementia_related_psychosis", "terms": ["Dementia-related Psychosis", "Dementia With Lewy Bodies"], "normalized_indication": "dementia-related psychosis", "therapeutic_area": "neurology"},
            {"id": "indication_multiple_sclerosis", "terms": ["Multiple Sclerosis"], "normalized_indication": "multiple sclerosis", "therapeutic_area": "neurology/immunology"},
        ],
    },
    "sponsor_aliases.yaml": {
        "aliases": {
            "Eli Lilly and Company": "Eli Lilly and Company",
            "Hoffmann-La Roche": "Roche Holding AG",
            "Genentech, Inc.": "Roche Holding AG",
            "Janssen Research & Development, LLC": "Johnson & Johnson",
            "Bristol-Myers Squibb": "Bristol Myers Squibb",
        }
    },
    "human_overrides.yaml": {},
}


def load_rule_config(filename: str) -> dict[str, Any]:
    """Load a YAML config file when PyYAML is installed, otherwise use defaults."""

    default = DEFAULT_RULES.get(filename, {})
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return default
    try:
        text = resources.files("pharma_os.data").joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return default
    loaded = yaml.safe_load(text) or {}
    return loaded if isinstance(loaded, dict) else default


def human_override(nct_id: str) -> dict[str, Any]:
    """Return reviewed human override values for one NCT ID."""

    overrides = load_rule_config("human_overrides.yaml").get(nct_id, {})
    return overrides if isinstance(overrides, dict) else {}
