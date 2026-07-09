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
