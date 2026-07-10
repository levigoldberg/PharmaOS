from __future__ import annotations

from pharma_os.schemas import ClinicalTrialRecord, TrialIntervention, TrialSponsor
from pharma_os.tools.asset_identity import resolve_asset_identity


class NoMatchRxNormClient:
    def normalize(self, drug_name: str):
        from pharma_os.schemas import SourceMetadata

        return None, SourceMetadata(
            source_id=f"rxnorm:{drug_name.casefold()}",
            title=f"RxNorm normalization for {drug_name}",
            provenance="test",
            source_type="drug_normalization",
        )


def test_asset_identity_extracts_development_code_alias_from_title() -> None:
    trial = ClinicalTrialRecord(
        nct_id="NCT05966480",
        brief_title="Phase 2 Study of ESK-001 in Active Systemic Lupus Erythematosus",
        official_title="A Study of Multiple Dose Levels of ESK-001",
        phases=("PHASE2",),
        conditions=("SLE",),
        interventions=(
            TrialIntervention(name="Envudeucitinib", type="DRUG"),
            TrialIntervention(name="Placebo", type="DRUG"),
        ),
        lead_sponsor=TrialSponsor(name="Alumis Inc"),
        source_id="ctgov:NCT05966480",
    )

    output, _ = resolve_asset_identity(trial, rxnorm_client=NoMatchRxNormClient())

    assert output.asset_name == "Envudeucitinib"
    assert "ESK-001" in output.aliases


def test_asset_identity_normalizes_multi_term_atopic_dermatitis_conditions() -> None:
    trial = ClinicalTrialRecord(
        nct_id="NCT07011706",
        brief_title="A Study of ATI-045 in Atopic Dermatitis",
        official_title="A Study of ATI-045 in Adult Participants With AD",
        phases=("PHASE2",),
        conditions=("Atopic Dermatitis", "Atopic", "Dermatitis", "AD", "Eczema"),
        interventions=(
            TrialIntervention(name="ATI-045", type="DRUG"),
            TrialIntervention(name="Placebo", type="DRUG"),
        ),
        lead_sponsor=TrialSponsor(name="Aclaris Therapeutics, Inc."),
        source_id="ctgov:NCT07011706",
    )

    output, _ = resolve_asset_identity(trial, rxnorm_client=NoMatchRxNormClient())

    assert output.normalized_indication == "atopic dermatitis"
    assert output.therapeutic_area == "immunology/dermatology"
    assert "indication_atopic_dermatitis" in output.rule_ids
    assert not any(flag.flag_id == "asset-missing-indication" for flag in output.missing_data_flags)
