from __future__ import annotations

from pharma_os.tools import asset_identity as asset_identity_module
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


def test_asset_identity_prefers_sle_condition_over_adult_title_acronym() -> None:
    trial = ClinicalTrialRecord(
        nct_id="NCT05966480",
        brief_title="Phase 2 Study of ESK-001 in Active Systemic Lupus Erythematosus",
        official_title="Randomized Double-blind Study of ESK-001 in Adult Patients With Systemic Lupus Erythematosus",
        phases=("PHASE2",),
        conditions=("SLE",),
        interventions=(
            TrialIntervention(name="Envudeucitinib", type="DRUG", description="Oral tablet"),
            TrialIntervention(name="Placebo", type="DRUG"),
        ),
        lead_sponsor=TrialSponsor(name="Alumis Inc"),
        source_id="ctgov:NCT05966480",
    )

    output, _ = resolve_asset_identity(trial, rxnorm_client=NoMatchRxNormClient())

    assert output.normalized_indication == "systemic lupus erythematosus"
    assert output.therapeutic_area == "immunology"
    assert "indication_systemic_lupus_erythematosus" in output.rule_ids
    assert output.secondary_indications == ()


def test_asset_identity_preserves_other_asset_indications_as_secondary_context() -> None:
    trial = ClinicalTrialRecord(
        nct_id="NCT05966480",
        brief_title="Study of ESK-001 in Atopic Dermatitis",
        official_title="Study of ESK-001 in Adult Patients With Atopic Dermatitis",
        phases=("PHASE2",),
        conditions=("SLE",),
        interventions=(
            TrialIntervention(name="Envudeucitinib", type="DRUG", description="Oral tablet"),
            TrialIntervention(name="Placebo", type="DRUG"),
        ),
        lead_sponsor=TrialSponsor(name="Alumis Inc"),
        source_id="ctgov:NCT05966480",
    )

    output, _ = resolve_asset_identity(trial, rxnorm_client=NoMatchRxNormClient())

    assert output.normalized_indication == "systemic lupus erythematosus"
    assert output.secondary_indications == ("atopic dermatitis",)


def test_asset_identity_explicit_indication_override_still_wins(monkeypatch) -> None:
    monkeypatch.setattr(
        asset_identity_module,
        "human_override",
        lambda nct_id: {"indication": "atopic dermatitis", "therapeutic_area": "immunology/dermatology"}
        if nct_id == "NCT05966480"
        else {},
    )
    trial = ClinicalTrialRecord(
        nct_id="NCT05966480",
        brief_title="Phase 2 Study of ESK-001 in Active Systemic Lupus Erythematosus",
        phases=("PHASE2",),
        conditions=("SLE",),
        interventions=(
            TrialIntervention(name="Envudeucitinib", type="DRUG", description="Oral tablet"),
            TrialIntervention(name="Placebo", type="DRUG"),
        ),
        lead_sponsor=TrialSponsor(name="Alumis Inc"),
        source_id="ctgov:NCT05966480",
    )

    output, _ = resolve_asset_identity(trial, rxnorm_client=NoMatchRxNormClient())

    assert output.normalized_indication == "atopic dermatitis"
    assert output.therapeutic_area == "immunology/dermatology"
    assert "human_override" in output.rule_ids


def test_asset_identity_dedupes_multiple_dose_arms_of_same_active_asset() -> None:
    trial = ClinicalTrialRecord(
        nct_id="NCT04999839",
        brief_title="Study of NDI-034858 in Plaque Psoriasis",
        phases=("PHASE2",),
        conditions=("Plaque Psoriasis",),
        interventions=(
            TrialIntervention(name="NDI-034858 2 mg tablet", type="DRUG"),
            TrialIntervention(name="NDI-034858 10 mg tablet", type="DRUG"),
            TrialIntervention(name="NDI-034858 30 mg tablet", type="DRUG"),
            TrialIntervention(name="Placebo", type="DRUG"),
        ),
        lead_sponsor=TrialSponsor(name="Nimbus Lakshmi, Inc."),
        source_id="ctgov:NCT04999839",
    )

    output, _ = resolve_asset_identity(trial, rxnorm_client=NoMatchRxNormClient())

    assert output.asset_name == "NDI-034858 2 mg tablet"
    assert output.normalized_indication == "psoriasis"
    assert not any(flag.flag_id == "asset-multiple-candidates" for flag in output.missing_data_flags)
