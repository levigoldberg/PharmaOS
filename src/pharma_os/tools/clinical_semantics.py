"""Clinical normalization helpers shared by protocol-design tools."""

from __future__ import annotations

import re
from functools import lru_cache

from pharma_os.schemas import ClinicalTrialRecord, TrialIntervention
from pharma_os.tools._due_diligence_common import slug
from pharma_os.tools.rules import load_rule_config


PSORIASIS_EFFICACY_ENDPOINTS = frozenset(
    {
        "pasi_75",
        "pasi_90",
        "pasi_100",
        "pga_iga_0_1",
        "continuous_pasi_change",
    }
)

CONTROL_SLUGS = frozenset(
    {
        "placebo",
        "control",
        "vehicle",
        "standardofcare",
        "bestsupportivecare",
        "nointervention",
        "sham",
    }
)

GENERIC_INTERVENTION_SLUGS = frozenset(
    {
        "drug",
        "therapy",
        "treatment",
        "investigationaldrug",
        "investigationalproduct",
        "study drug",
        "studydrug",
    }
)


def clean_intervention_name(name: str | None) -> str | None:
    """Remove CT.gov intervention prefixes and empty control wrappers."""

    if not name:
        return None
    text = str(name).strip()
    for prefix in (
        "Drug:",
        "Biological:",
        "Biologic:",
        "Device:",
        "Procedure:",
        "Radiation:",
        "Combination Product:",
    ):
        if text.casefold().startswith(prefix.casefold()):
            text = text[len(prefix) :].strip()
    return text or None


def strip_dose_suffix(value: str) -> str | None:
    """Strip common dose/unit suffixes while preserving development codes."""

    text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|g|ml|%)\b.*$", "", value, flags=re.I)
    text = re.sub(r"\b(?:dose|tablet|capsule|solution|injection|infusion)s?\b.*$", "", text, flags=re.I)
    text = text.strip(" -/,:;")
    return text or None


def is_control_or_generic_intervention(value: str | None) -> bool:
    """Return true for placebo, vehicle, standard care, or empty generic labels."""

    normalized = slug(value or "")
    if not normalized:
        return True
    if normalized in CONTROL_SLUGS or normalized in GENERIC_INTERVENTION_SLUGS:
        return True
    return any(token in normalized for token in CONTROL_SLUGS)


def active_interventions(trial: ClinicalTrialRecord) -> tuple[TrialIntervention, ...]:
    """Return non-control active interventions, preferring drug-like CT.gov types."""

    drug_like_types = {"DRUG", "BIOLOGICAL", "GENETIC"}
    drug_like = tuple(
        item
        for item in trial.interventions
        if (item.type or "").upper() in drug_like_types and not is_control_or_generic_intervention(item.name)
    )
    if drug_like:
        return _dedupe_interventions(drug_like)
    non_control = tuple(item for item in trial.interventions if not is_control_or_generic_intervention(item.name))
    return _dedupe_interventions(non_control)


def active_intervention_text(trial: ClinicalTrialRecord) -> str:
    """Text used for route/modality inference from active interventions and arms."""

    values: list[str] = []
    for intervention in active_interventions(trial):
        values.extend(
            item
            for item in (intervention.name, intervention.description, *intervention.other_names, *intervention.arm_group_labels)
            if item
        )
    for arm in trial.arm_groups:
        values.extend(item for item in (arm.label, arm.description, *arm.intervention_names) if item)
    values.extend(item for item in (trial.brief_title, trial.official_title) if item)
    return " ".join(values)


def route_set(trial: ClinicalTrialRecord) -> set[str]:
    """Infer clinically meaningful active-intervention route categories."""

    text = active_intervention_text(trial).casefold()
    routes: set[str] = set()
    if re.search(r"\b(oral|orally|po|by mouth|tablet|tablets|capsule|capsules)\b", text):
        routes.add("oral")
    if re.search(r"\b(intravenous|iv|infusion)\b", text):
        routes.add("intravenous")
    if re.search(r"\b(subcutaneous|sc|s/c)\b", text):
        routes.add("subcutaneous")
    if re.search(r"\b(topical|cream|ointment|gel)\b", text):
        routes.add("topical")
    if re.search(r"\b(intramuscular|im)\b", text):
        routes.add("intramuscular")
    return routes


def modality_set(trial: ClinicalTrialRecord) -> set[str]:
    """Infer modality for active interventions without placebo/control pollution."""

    text = active_intervention_text(trial).casefold()
    types = {(item.type or "").upper() for item in active_interventions(trial)}
    modalities: set[str] = set()
    if "BIOLOGICAL" in types or any(term in text for term in ("antibody", "monoclonal", "mab", "biologic", "secukinumab")):
        modalities.add("biologic")
    if "GENETIC" in types or any(term in text for term in ("gene therapy", "sirna", "antisense", "oligonucleotide")):
        modalities.add("genetic_or_oligonucleotide")
    if "DRUG" in types or any(term in text for term in ("inhibitor", "agonist", "antagonist", "modulator", "tablet", "capsule", "oral")):
        modalities.add("small_molecule")
    return modalities


def comparable_modality(left: ClinicalTrialRecord, right: ClinicalTrialRecord) -> bool | None:
    """Compare active asset modality and route while ignoring controls."""

    left_modalities = modality_set(left)
    right_modalities = modality_set(right)
    left_routes = route_set(left)
    right_routes = route_set(right)
    if left_modalities and right_modalities and left_modalities & right_modalities:
        if left_routes and right_routes:
            return bool(left_routes & right_routes)
        return True
    if left_routes and right_routes and left_routes & right_routes:
        return True
    if (left_modalities and right_modalities) or (left_routes and right_routes):
        return False
    return None


def normalized_indication_terms(terms: tuple[str, ...]) -> set[str]:
    """Normalize condition strings into hierarchy-aware clinical indication buckets."""

    normalized: set[str] = set()
    for term in terms:
        text = " ".join(str(term).split()).strip(" .,:;")
        lowered = text.casefold()
        if not text:
            continue
        normalized.add(slug(text))
        if "plaque psoriasis" in lowered or "psoriasis" in lowered or "psoriatic" in lowered:
            normalized.update({"psoriasis", "plaquepsoriasis"})
        if "atopic dermatitis" in lowered or lowered in {"ad", "eczema"}:
            normalized.add("atopicdermatitis")
        if "systemic lupus erythematosus" in lowered or lowered in {"sle", "lupus"}:
            normalized.add("systemiclupuserythematosus")
        if "rheumatoid arthritis" in lowered:
            normalized.add("rheumatoidarthritis")
        if "glioblastoma" in lowered or lowered == "gbm":
            normalized.add("glioblastoma")
    return normalized


def same_indication(left_terms: tuple[str, ...], right_terms: tuple[str, ...]) -> bool:
    """Hierarchy-aware same-indication comparison."""

    left = normalized_indication_terms(left_terms)
    right = normalized_indication_terms(right_terms)
    if not left or not right:
        return False
    if left & right:
        return True
    return any(
        (len(left_term) >= 5 and left_term in right_term)
        or (len(right_term) >= 5 and right_term in left_term)
        for left_term in left
        for right_term in right
    )


def condition_variants(value: str | None) -> tuple[str, ...]:
    """Return CT.gov query variants for an indication."""

    if not value:
        return ()
    text = " ".join(str(value).split()).strip(" .,:;")
    variants = [text]
    lowered = text.casefold()
    for prefix in ("moderate to severe ", "moderate-to-severe ", "severe "):
        if lowered.startswith(prefix):
            variants.append(text[len(prefix) :].strip())
    if "plaque psoriasis" in lowered:
        variants.extend(("Psoriasis", "Plaque Psoriasis"))
    if "psoriasis" in lowered:
        variants.append("Psoriasis")
    if "atopic dermatitis" in lowered:
        variants.append("Atopic Dermatitis")
    if "systemic lupus erythematosus" in lowered or lowered in {"sle", "lupus"}:
        variants.extend(("Systemic Lupus Erythematosus", "SLE"))
    return tuple(dict.fromkeys(item for item in variants if item))


def endpoint_family(measure: str | None, description: str | None = None, time_frame: str | None = None) -> str:
    """Classify clinically meaningful endpoint families before generic buckets."""

    text = " ".join(item for item in (measure, description, time_frame) if item).casefold()
    if not text:
        return "other"
    if re.search(r"\bpasi[-\s]?100\b", text) or ("100%" in text and "pasi" in text):
        return "pasi_100"
    if re.search(r"\bpasi[-\s]?90\b", text) or ("90%" in text and "pasi" in text):
        return "pasi_90"
    if re.search(r"\bpasi[-\s]?75\b", text) or ("75%" in text and "pasi" in text):
        return "pasi_75"
    if (
        re.search(r"\b(?:pga|iga|spga|static physician global assessment|investigator global assessment)\b", text)
        and re.search(r"\b(?:0\s*/?\s*1|0\s+or\s+1|clear|almost clear|success)\b", text)
    ):
        return "pga_iga_0_1"
    if "dlqi" in text or "dermatology life quality index" in text:
        return "dlqi"
    if "pasi" in text and any(term in text for term in ("change", "reduction", "improvement", "score", "area and severity")):
        return "continuous_pasi_change"
    if any(term in text for term in ("pharmacokinetic", "cmax", "auc", "trough", "concentration", "half-life", "t1/2")):
        return "pk"
    if any(term in text for term in ("pharmacodynamic", "biomarker", "marker", "cytokine", "receptor occupancy")):
        return "pd_biomarker"
    if any(term in text for term in ("safety", "adverse", "toxicity", "serious adverse", "tolerability")):
        return "safety"
    if any(term in text for term in ("overall survival", "mortality", "death")):
        return "survival"
    if any(term in text for term in ("progression-free", "time to", "duration")):
        return "time_to_event"
    if any(term in text for term in ("response", "orr", "remission")):
        return "response"
    return "other"


def endpoint_families_for_trial(trial: ClinicalTrialRecord, *, primary_only: bool = True) -> tuple[str, ...]:
    """Return ordered endpoint families for primary endpoints, falling back to secondary endpoints."""

    endpoints = trial.primary_endpoints if primary_only else (*trial.primary_endpoints, *trial.secondary_endpoints)
    if not endpoints and primary_only:
        endpoints = trial.secondary_endpoints
    families = tuple(
        endpoint_family(endpoint.measure, endpoint.description, endpoint.time_frame)
        for endpoint in endpoints
    )
    return tuple(dict.fromkeys(families))


def endpoint_family_from_trial(trial: ClinicalTrialRecord, *, primary_only: bool = True) -> str | None:
    """Return the first informative endpoint family for a trial."""

    families = endpoint_families_for_trial(trial, primary_only=primary_only)
    for family in families:
        if family != "other":
            return family
    return families[0] if families else None


def same_endpoint_domain(left_family: str | None, right_family: str | None) -> bool | None:
    """Return true when endpoint families belong to the same clinical domain."""

    if not left_family or not right_family:
        return None
    if left_family == right_family:
        return True
    if left_family in PSORIASIS_EFFICACY_ENDPOINTS and right_family in PSORIASIS_EFFICACY_ENDPOINTS:
        return True
    return False


def title_code_aliases(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    """Extract asset-like development codes from trial titles."""

    text = " ".join(item for item in (trial.brief_title, trial.official_title) if item)
    aliases = []
    for match in re.finditer(r"\b[A-Z]{2,}[A-Z0-9]*[- ]?\d+[A-Z0-9]*\b", text):
        alias = match.group(0).strip()
        if not is_control_or_generic_intervention(alias):
            aliases.append(alias)
    return tuple(dict.fromkeys(aliases))


def asset_aliases(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    """Return active intervention aliases before config expansion."""

    aliases: list[str] = []
    for intervention in active_interventions(trial):
        for name in (intervention.name, *intervention.other_names):
            cleaned = clean_intervention_name(name)
            if cleaned and not is_control_or_generic_intervention(cleaned):
                aliases.append(cleaned)
    aliases.extend(alias for alias in getattr(trial, "intervention_browse_terms", ()) if not is_control_or_generic_intervention(alias))
    aliases.extend(title_code_aliases(trial))
    return tuple(dict.fromkeys(aliases))


def expanded_asset_aliases(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    """Expand active aliases through dose stripping and configured development-name maps."""

    aliases: list[str] = []
    for alias in asset_aliases(trial):
        for value in (alias, strip_dose_suffix(alias)):
            if value and not is_control_or_generic_intervention(value):
                aliases.append(value)
                aliases.extend(configured_asset_aliases(value))
    return tuple(dict.fromkeys(alias for alias in aliases if alias and not is_control_or_generic_intervention(alias)))


def configured_asset_aliases(value: str) -> tuple[str, ...]:
    """Look up configured and fallback asset aliases."""

    alias_sets = _asset_alias_sets()
    return tuple(alias_sets.get(slug(value), ()))


def equivalent_asset_slugs(value: str) -> set[str]:
    """Return slug-equivalent names for one asset alias."""

    values = {value, *configured_asset_aliases(value)}
    stripped = strip_dose_suffix(value)
    if stripped:
        values.add(stripped)
        values.update(configured_asset_aliases(stripped))
    return {slug(item) for item in values if item and not is_control_or_generic_intervention(item)}


def asset_matches(trial: ClinicalTrialRecord, alias: str) -> bool:
    """Match a candidate trial to an anchor asset alias, allowing code-name evolution."""

    alias_slugs = equivalent_asset_slugs(alias)
    for name in expanded_asset_aliases(trial):
        name_slugs = equivalent_asset_slugs(name)
        if alias_slugs & name_slugs:
            return True
        if any(len(left) >= 5 and len(right) >= 5 and (left in right or right in left) for left in alias_slugs for right in name_slugs):
            return True
    return False


def normalized_sponsor_names(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    """Return sponsor names normalized through sponsor alias configuration."""

    aliases = load_rule_config("sponsor_aliases.yaml").get("aliases", {})
    alias_by_slug = {slug(str(key)): str(value) for key, value in aliases.items()} if isinstance(aliases, dict) else {}
    names: list[str] = []
    for sponsor in (trial.lead_sponsor, *trial.collaborators):
        if not sponsor or not sponsor.name:
            continue
        names.append(sponsor.name)
        canonical = alias_by_slug.get(slug(sponsor.name))
        if canonical:
            names.append(canonical)
    return tuple(dict.fromkeys(names))


@lru_cache(maxsize=1)
def _asset_alias_sets() -> dict[str, tuple[str, ...]]:
    config = load_rule_config("asset_aliases.yaml").get("aliases", {})
    alias_sets: dict[str, set[str]] = {}
    if isinstance(config, dict):
        for key, values in config.items():
            members = [str(key)]
            if isinstance(values, (list, tuple)):
                members.extend(str(value) for value in values)
            elif values:
                members.append(str(values))
            for member in members:
                alias_sets.setdefault(slug(member), set()).update(members)
    fallback = {
        "cp-690-550": ("CP-690,550", "CP 690,550", "CP-690550", "tofacitinib", "tasocitinib"),
        "cp-690550": ("CP-690,550", "CP 690,550", "CP-690550", "tofacitinib", "tasocitinib"),
        "cp690550": ("CP-690,550", "CP 690,550", "CP-690550", "tofacitinib", "tasocitinib"),
        "tofacitinib": ("tofacitinib", "CP-690,550", "CP 690,550", "CP-690550", "tasocitinib"),
        "tasocitinib": ("tasocitinib", "tofacitinib", "CP-690,550", "CP 690,550", "CP-690550"),
        "ain457": ("AIN457", "secukinumab"),
        "secukinumab": ("secukinumab", "AIN457"),
    }
    for key, values in fallback.items():
        alias_sets.setdefault(slug(key), set()).update(values)
        for value in values:
            alias_sets.setdefault(slug(value), set()).update(values)
    return {key: tuple(dict.fromkeys(values)) for key, values in alias_sets.items()}


def _dedupe_interventions(interventions: tuple[TrialIntervention, ...]) -> tuple[TrialIntervention, ...]:
    seen: set[str] = set()
    result: list[TrialIntervention] = []
    for intervention in interventions:
        canonical = strip_dose_suffix(clean_intervention_name(intervention.name) or intervention.name) or intervention.name
        key = slug(canonical)
        if key in seen:
            continue
        seen.add(key)
        result.append(intervention)
    return tuple(result)
