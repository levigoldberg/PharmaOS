"""Asset identity resolution from CT.gov, RxNorm, and shared rules."""

from __future__ import annotations

import re
from typing import Any

from pharma_os.schemas import AssetIdentityOutput, ClinicalTrialRecord, MissingDataFlag, SourceMetadata
from pharma_os.tools._due_diligence_common import missing
from pharma_os.tools.clinical_semantics import active_interventions, title_code_aliases
from pharma_os.tools.rxnorm import RxNormClient, RxNormError
from pharma_os.tools.rules import human_override, load_rule_config


def resolve_asset_identity(
    trial: ClinicalTrialRecord,
    *,
    rxnorm_client: RxNormClient | None = None,
) -> tuple[AssetIdentityOutput, tuple[SourceMetadata, ...]]:
    """Resolve asset identity from a normalized CT.gov trial and RxNorm."""

    sources: list[SourceMetadata] = [
        SourceMetadata(
            source_id=trial.source_id,
            title=trial.brief_title or trial.official_title or trial.nct_id,
            url=f"https://clinicaltrials.gov/study/{trial.nct_id}",
            authors=tuple(
                sponsor.name
                for sponsor in (trial.lead_sponsor, *trial.collaborators)
                if sponsor is not None
            ),
            provenance="ClinicalTrials.gov API v2 protocolSection",
            source_type="clinical_trial_registry",
            version="v2",
        )
    ]
    flags: list[MissingDataFlag] = []
    overrides = human_override(trial.nct_id)
    candidates = list(active_interventions(trial))
    if not candidates:
        candidates = [item for item in trial.interventions if "placebo" not in item.name.casefold()]
    selected = candidates[0] if candidates else None
    if len(candidates) > 1:
        flags.append(missing("asset-multiple-candidates", "asset_identity", "asset_name", "Multiple non-placebo interventions need review.", "medium"))
    if selected is None:
        flags.append(missing("asset-missing-name", "asset_identity", "asset_name", "No non-placebo asset candidate was found.", "high"))

    rxnorm_match = None
    if selected is not None:
        try:
            rxnorm_match, rx_source = (rxnorm_client or RxNormClient()).normalize(selected.name)
            sources.append(rx_source)
            if rxnorm_match is None:
                flags.append(missing("asset-no-rxnorm", "asset_identity", "rxnorm_match", "RxNorm returned no match.", "medium"))
        except RxNormError as exc:
            flags.append(missing("asset-rxnorm-error", "asset_identity", "rxnorm_match", str(exc), "medium"))

    modality, modality_rule = _infer_modality(selected)
    if overrides.get("modality"):
        modality = str(overrides["modality"])
        modality_rule = "human_override"
    indication, therapeutic_area, indication_rule = _infer_indication(trial)
    if overrides.get("indication"):
        indication = str(overrides["indication"])
        indication_rule = "human_override"
    if overrides.get("therapeutic_area"):
        therapeutic_area = str(overrides["therapeutic_area"])
        indication_rule = "human_override"
    sponsor = trial.lead_sponsor.name if trial.lead_sponsor else None
    sponsor_rule = "lead_sponsor_fallback" if sponsor else None
    aliases_config = load_rule_config("sponsor_aliases.yaml").get("aliases", {})
    if isinstance(aliases_config, dict) and sponsor in aliases_config:
        sponsor = str(aliases_config[sponsor])
        sponsor_rule = "sponsor_alias_exact"
    if overrides.get("sponsor"):
        sponsor = str(overrides["sponsor"])
        sponsor_rule = "human_override"
    if sponsor is None:
        flags.append(missing("asset-missing-sponsor", "asset_identity", "sponsor", "ClinicalTrials.gov did not list a lead sponsor.", "medium"))
    if indication is None:
        flags.append(missing("asset-missing-indication", "asset_identity", "normalized_indication", "No deterministic indication rule matched.", "medium"))
    if modality == "unknown":
        flags.append(missing("asset-unknown-modality", "asset_identity", "modality", "No deterministic modality rule matched.", "medium"))

    aliases = tuple(
        dict.fromkeys(
            [
                *(selected.other_names if selected else ()),
                *title_code_aliases(trial),
                *(rxnorm_match.aliases if rxnorm_match else ()),
            ]
        )
    )
    secondary_indications = tuple(
        item
        for item in _secondary_indications(trial)
        if indication is None or item.casefold() != indication.casefold()
    )
    confidence = 0.85 - min(len(flags), 4) * 0.15
    return (
        AssetIdentityOutput(
            nct_id=trial.nct_id,
            asset_name=selected.name if selected else None,
            raw_intervention_names=tuple(item.name for item in trial.interventions),
            intervention_type=selected.type if selected else None,
            aliases=aliases,
            rxnorm_match=rxnorm_match,
            sponsor=sponsor,
            normalized_indication=indication,
            secondary_indications=secondary_indications,
            therapeutic_area=therapeutic_area,
            modality=modality,
            rule_ids=tuple(item for item in (modality_rule, indication_rule, sponsor_rule) if item),
            source_ids=tuple(source.source_id for source in sources),
            missing_data_flags=tuple(flags),
            confidence=max(0.1, confidence),
        ),
        tuple(sources),
    )


def _infer_modality(selected: Any) -> tuple[str, str | None]:
    text = " ".join([getattr(selected, "name", "") or "", getattr(selected, "description", "") or "", *getattr(selected, "other_names", ())]).casefold()
    config = load_rule_config("modality_rules.yaml")
    for rule in config.get("rules", []):
        keywords = [str(keyword).casefold() for keyword in rule.get("keywords", [])]
        if any(keyword in text for keyword in keywords):
            return str(rule.get("modality")), str(rule.get("id") or "modality_rule")
    return str(config.get("default", "unknown")), None


def _infer_indication(trial: ClinicalTrialRecord) -> tuple[str | None, str | None, str | None]:
    condition_match = _match_indication_text(" | ".join(trial.conditions))
    if condition_match is not None:
        return condition_match
    title_match = _match_indication_text(" | ".join([trial.brief_title or "", trial.official_title or ""]))
    if title_match is not None:
        return title_match
    if len(trial.conditions) == 1:
        return trial.conditions[0], None, None
    return None, None, None


def _match_indication_text(text: str) -> tuple[str | None, str | None, str | None] | None:
    text = text.casefold()
    if not text.strip():
        return None
    for rule in load_rule_config("indication_rules.yaml").get("rules", []):
        all_terms = [str(term).casefold() for term in rule.get("all_terms", [])]
        terms = [str(term).casefold() for term in rule.get("terms", [])]
        if (all_terms and all(_term_matches(text, term) for term in all_terms)) or any(_term_matches(text, term) for term in terms):
            return (
                str(rule.get("normalized_indication")),
                str(rule.get("therapeutic_area")),
                str(rule.get("id") or "indication_rule"),
            )
    return None


def _secondary_indications(trial: ClinicalTrialRecord) -> tuple[str, ...]:
    matches: list[str] = []
    for text in (
        " | ".join([trial.brief_title or "", trial.official_title or ""]),
        " | ".join(trial.conditions),
    ):
        lowered = text.casefold()
        if not lowered.strip():
            continue
        for rule in load_rule_config("indication_rules.yaml").get("rules", []):
            all_terms = [str(term).casefold() for term in rule.get("all_terms", [])]
            terms = [str(term).casefold() for term in rule.get("terms", [])]
            if (all_terms and all(_term_matches(lowered, term) for term in all_terms)) or any(_term_matches(lowered, term) for term in terms):
                value = rule.get("normalized_indication")
                if value:
                    matches.append(str(value))
    return tuple(dict.fromkeys(matches))


def _term_matches(text: str, term: str) -> bool:
    if len(term) <= 3 or re.fullmatch(r"[a-z0-9-]+", term):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text))
    return term in text
