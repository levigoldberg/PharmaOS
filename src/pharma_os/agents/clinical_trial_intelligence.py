"""Compatibility wrappers for the Agent 3 trial-landscape component."""

from __future__ import annotations

from typing import Any

from pharma_os.components.trial_landscape import build_trial_landscape_output, search_trial_landscape
from pharma_os.schemas import (
    ClinicalTrialIntelligenceInput,
    ClinicalTrialIntelligenceOutput,
    ClinicalTrialsSearchResult,
)
from pharma_os.tools.clinicaltrials import ClinicalTrialsGovClient


AGENT_NAME = "agent3_trial_landscape_component"


def build_clinical_trial_intelligence_agent() -> None:
    """Clinical trial intelligence is no longer a standalone LLM agent."""

    return None


def run_clinical_trial_intelligence_agent(
    input_data: ClinicalTrialIntelligenceInput,
    *,
    run_id: str,
    agent: Any | None = None,
) -> tuple[ClinicalTrialIntelligenceOutput, dict[str, str | int | float | bool | None]]:
    """Run Agent 3 landscape mode through the deterministic internal component."""

    if agent is not None:
        raise ValueError("trial_intelligence no longer accepts a standalone LLM agent")
    output = search_trial_landscape(
        disease=input_data.disease,
        drug=input_data.drug,
        target=input_data.target,
        phase=input_data.phase,
        limit=input_data.limit,
        run_id=run_id,
        client=ClinicalTrialsGovClient(),
    )
    return output, {"mode": "agent3_trial_landscape_component"}


def deterministic_trial_intelligence_output(
    *,
    run_id: str,
    input_data: ClinicalTrialIntelligenceInput,
    search_result: ClinicalTrialsSearchResult,
) -> ClinicalTrialIntelligenceOutput:
    """Build compatibility output from tool data for tests or offline fallback injection."""

    return build_trial_landscape_output(
        run_id=run_id,
        input_data=input_data,
        search_result=search_result,
    )
