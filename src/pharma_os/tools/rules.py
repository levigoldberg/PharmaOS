"""Configuration loading for PharmaOS due-diligence tools."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

from pharma_os.schemas import SourceMetadata


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
            {"id": "indication_psoriasis", "terms": ["Psoriasis", "Plaque Psoriasis", "Psoriatic"], "normalized_indication": "psoriasis", "therapeutic_area": "immunology/dermatology"},
            {"id": "indication_atopic_dermatitis", "terms": ["Atopic Dermatitis", "AD", "Eczema"], "normalized_indication": "atopic dermatitis", "therapeutic_area": "immunology/dermatology"},
            {"id": "indication_systemic_lupus_erythematosus", "terms": ["Systemic Lupus Erythematosus", "SLE", "Lupus"], "normalized_indication": "systemic lupus erythematosus", "therapeutic_area": "immunology"},
        ],
    },
    "asset_aliases.yaml": {
        "aliases": {
            "CP-690,550": ["CP 690,550", "CP-690550", "tofacitinib", "tasocitinib"],
            "AIN457": ["secukinumab"],
            "secukinumab": ["AIN457"],
            "JNJ-77242113": ["JNJ 77242113"],
            "NDI-034858": ["TAK-279", "TAK279"],
            "TAK-279": ["NDI-034858", "TAK279"],
        }
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


CONFIG_SOURCE_IDS = {
    "shared": {
        "human_overrides.yaml": "config:shared:human_overrides",
        "asset_aliases.yaml": "config:shared:asset_aliases",
        "indication_rules.yaml": "config:shared:indication_rules",
        "modality_rules.yaml": "config:shared:modality_rules",
        "sponsor_aliases.yaml": "config:shared:sponsor_aliases",
    },
    "due_diligence": {
        "default_archetypes.yaml": "config:due_diligence:default_archetypes",
        "market_bucket_templates.yaml": "config:due_diligence:market_bucket_templates",
        "market_query_templates.yaml": "config:due_diligence:market_query_templates",
        "rnpv_assumptions_config.yaml": "config:due_diligence:rnpv_assumptions_config",
        "wac_sources.yaml": "config:due_diligence:wac_sources",
    },
}


def load_config(filename: str, *, section: str = "shared") -> dict[str, Any]:
    """Load a PharmaOS YAML config from package data or repo data."""

    default = DEFAULT_RULES.get(filename, {})
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        return default
    text = _read_config_text(filename, section)
    if text is None:
        return default
    loaded = yaml.safe_load(text) or {}
    return loaded if isinstance(loaded, dict) else default


def load_rule_config(filename: str) -> dict[str, Any]:
    """Backward-compatible shared config loader."""

    return load_config(filename, section="shared")


def config_source_id(filename: str, *, section: str = "shared") -> str:
    """Return the stable source ID for a config file."""

    return CONFIG_SOURCE_IDS.get(section, {}).get(filename, f"config:{section}:{filename}")


def config_source(filename: str, *, section: str = "shared") -> SourceMetadata:
    """Return source metadata for a config file."""

    return SourceMetadata(
        source_id=config_source_id(filename, section=section),
        title=f"{section}/{filename}",
        provenance="PharmaOS packaged YAML configuration",
        source_type="configuration",
        version="local",
    )


def config_provenance(filename: str, field_path: str, *, section: str = "shared") -> str:
    """Return provenance text for a config-derived assumption."""

    return f"{section}/{filename}:{field_path}"


def _read_config_text(filename: str, section: str) -> str | None:
    package_candidates = [
        ("pharma_os.data", (section, filename)),
        ("pharma_os.data", (filename,)),
    ]
    for package, parts in package_candidates:
        try:
            return resources.files(package).joinpath(*parts).read_text(encoding="utf-8")
        except (FileNotFoundError, ModuleNotFoundError):
            pass
    repo_root = Path(__file__).resolve().parents[3]
    for path in (
        repo_root / "data" / section / filename,
        repo_root / "data" / filename,
    ):
        if path.exists():
            return path.read_text(encoding="utf-8")
    return None


def human_override(nct_id: str) -> dict[str, Any]:
    """Return reviewed human override values for one NCT ID."""

    overrides = load_config("human_overrides.yaml", section="shared").get(nct_id, {})
    return overrides if isinstance(overrides, dict) else {}
