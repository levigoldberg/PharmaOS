"""Clinical Trial Intelligence Agent."""

from __future__ import annotations

from typing import Any

from pharma_os.agents.runtime import load_agents_sdk, run_agent
from pharma_os.schemas import (
    ClinicalTrialIntelligenceInput,
    ClinicalTrialIntelligenceOutput,
    ClinicalTrialRecord,
    ClinicalTrialsSearchResult,
)
from pharma_os.tools.clinicaltrials import ClinicalTrialsGovClient


AGENT_NAME = "clinical_trial_intelligence_agent"


def build_clinical_trial_intelligence_agent() -> Any:
    """Build the one-agent Clinical Trial Intelligence loop."""

    Agent, AgentOutputSchema, _, function_tool = load_agents_sdk()

    @function_tool
    def search_clinical_trials(
        disease: str,
        drug: str | None = None,
        target: str | None = None,
        phase: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Search ClinicalTrials.gov for trials matching disease and optional filters."""

        result = ClinicalTrialsGovClient().search_trials(
            ClinicalTrialIntelligenceInput(
                disease=disease,
                drug=drug,
                target=target,
                phase=phase,
                limit=limit,
            )
        )
        return result.model_dump(mode="json")

    @function_tool
    def fetch_clinical_trial(nct_id: str) -> dict[str, Any]:
        """Fetch one ClinicalTrials.gov trial by NCT ID."""

        record = ClinicalTrialsGovClient().fetch_trial(nct_id)
        return record.model_dump(mode="json")

    return Agent(
        name=AGENT_NAME,
        output_type=AgentOutputSchema(
            ClinicalTrialIntelligenceOutput,
            strict_json_schema=False,
        ),
        tools=[search_clinical_trials, fetch_clinical_trial],
        instructions="""You are the Clinical Trial Intelligence Agent for PharmaOS.

Use the ClinicalTrials.gov tools for factual trial data. Return only structured output.

You must:
- accept disease, optional drug, optional target, optional phase, and limit from the input payload
- call the ClinicalTrials.gov search tool before writing the output
- normalize returned trials into the supplied schema
- cite source_id on every factual claim
- summarize trial landscape, statuses, phases, sponsors, endpoints, and populations conservatively
- flag obvious risks: terminated or withdrawn trials, missing results, very small enrollment, outdated status, unclear endpoints

You must not:
- make scientific go/no-go recommendations
- nominate targets
- approve protocols
- make safety decisions
- make investment recommendations
- calculate or mention rNPV
- invent facts not returned by tools
""",
    )


def run_clinical_trial_intelligence_agent(
    input_data: ClinicalTrialIntelligenceInput,
    *,
    run_id: str,
    agent: Any | None = None,
) -> tuple[ClinicalTrialIntelligenceOutput, dict[str, str | int | float | bool | None]]:
    """Run the Clinical Trial Intelligence Agent and return typed output plus trace metadata."""

    payload = {
        "run_id": run_id,
        "input": input_data.model_dump(mode="json"),
        "output_requirements": {
            "source_grounded_claims": True,
            "avoid_high_risk_recommendations": True,
        },
    }
    result = run_agent(
        agent or build_clinical_trial_intelligence_agent(),
        payload,
        ClinicalTrialIntelligenceOutput,
    )
    output = result.output
    if not isinstance(output, ClinicalTrialIntelligenceOutput):
        output = ClinicalTrialIntelligenceOutput.model_validate(output)
    return output, result.trace_metadata


def deterministic_trial_intelligence_output(
    *,
    run_id: str,
    input_data: ClinicalTrialIntelligenceInput,
    search_result: ClinicalTrialsSearchResult,
) -> ClinicalTrialIntelligenceOutput:
    """Build a conservative output from tool data for tests or offline fallback injection."""

    trials = search_result.trials
    claims = _claims(run_id, trials)
    risks = _risks(run_id, trials)
    return ClinicalTrialIntelligenceOutput(
        output_id=f"cti-output-{run_id}",
        run_id=run_id,
        input=input_data,
        trials=trials,
        sources=search_result.sources,
        claims=claims,
        risk_flags=risks,
        landscape_summary=f"Found {len(trials)} ClinicalTrials.gov records for {input_data.disease}.",
        status_summary=_count_summary("status", [trial.overall_status for trial in trials]),
        phase_summary=_count_summary("phase", [phase for trial in trials for phase in trial.phases]),
        sponsor_summary=_count_summary(
            "sponsor",
            [trial.lead_sponsor.name for trial in trials if trial.lead_sponsor],
        ),
        endpoint_summary=_endpoint_summary(trials),
        population_summary=_population_summary(trials),
        confidence=0.75 if trials else 0.4,
    )


def _claims(run_id: str, trials: tuple[ClinicalTrialRecord, ...]) -> tuple[Any, ...]:
    from pharma_os.schemas import EvidenceClaim

    claims = []
    for trial in trials:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-{trial.nct_id}-status",
                claim_text=f"{trial.nct_id} has overall status {trial.overall_status or 'unknown'}.",
                source_ids=(trial.source_id,),
                provenance="clinical_trial_intelligence.deterministic_claims",
                confidence=0.9,
                confidence_level="high",
            )
        )
        if trial.enrollment_count is not None:
            claims.append(
                EvidenceClaim(
                    claim_id=f"claim-{run_id}-{trial.nct_id}-enrollment",
                    claim_text=f"{trial.nct_id} reports enrollment of {trial.enrollment_count}.",
                    source_ids=(trial.source_id,),
                    provenance="clinical_trial_intelligence.deterministic_claims",
                    confidence=0.9,
                    confidence_level="high",
                )
            )
    return tuple(claims)


def _risks(run_id: str, trials: tuple[ClinicalTrialRecord, ...]) -> tuple[Any, ...]:
    from pharma_os.schemas import TrialLandscapeRisk

    risks = []
    for trial in trials:
        status = (trial.overall_status or "").upper()
        if status in {"TERMINATED", "WITHDRAWN", "SUSPENDED"}:
            risks.append(
                TrialLandscapeRisk(
                    risk_id=f"risk-{run_id}-{trial.nct_id}-status",
                    trial_id=trial.nct_id,
                    risk_type="terminated_or_withdrawn",
                    description=f"{trial.nct_id} has status {trial.overall_status}.",
                    severity="high",
                    source_ids=(trial.source_id,),
                )
            )
        if not trial.results_available:
            risks.append(
                TrialLandscapeRisk(
                    risk_id=f"risk-{run_id}-{trial.nct_id}-missing-results",
                    trial_id=trial.nct_id,
                    risk_type="missing_results",
                    description=f"{trial.nct_id} has no posted results in ClinicalTrials.gov.",
                    severity="medium",
                    source_ids=(trial.source_id,),
                )
            )
        if trial.enrollment_count is not None and trial.enrollment_count < 30:
            risks.append(
                TrialLandscapeRisk(
                    risk_id=f"risk-{run_id}-{trial.nct_id}-small-enrollment",
                    trial_id=trial.nct_id,
                    risk_type="small_enrollment",
                    description=f"{trial.nct_id} reports enrollment below 30.",
                    severity="medium",
                    source_ids=(trial.source_id,),
                )
            )
        if not trial.primary_endpoints:
            risks.append(
                TrialLandscapeRisk(
                    risk_id=f"risk-{run_id}-{trial.nct_id}-unclear-endpoints",
                    trial_id=trial.nct_id,
                    risk_type="unclear_endpoints",
                    description=f"{trial.nct_id} has no normalized primary endpoints.",
                    severity="medium",
                    source_ids=(trial.source_id,),
                )
            )
    return tuple(risks)


def _count_summary(label: str, values: list[str | None]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return f"No {label} values were available."
    return "; ".join(f"{key}: {value}" for key, value in sorted(counts.items()))


def _endpoint_summary(trials: tuple[ClinicalTrialRecord, ...]) -> str:
    total = sum(len(trial.primary_endpoints) for trial in trials)
    return f"{total} primary endpoints were normalized across {len(trials)} trials."


def _population_summary(trials: tuple[ClinicalTrialRecord, ...]) -> str:
    enrollments = [trial.enrollment_count for trial in trials if trial.enrollment_count is not None]
    if not enrollments:
        return "No enrollment counts were available."
    return f"Enrollment counts range from {min(enrollments)} to {max(enrollments)}."
