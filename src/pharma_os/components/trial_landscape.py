"""Deterministic Agent 3 trial-landscape component."""

from __future__ import annotations

from pharma_os.schemas import (
    ClinicalTrialIntelligenceInput,
    ClinicalTrialIntelligenceOutput,
    ClinicalTrialRecord,
    ClinicalTrialsSearchResult,
    EvidenceClaim,
    TrialLandscapeRisk,
)
from pharma_os.tools.clinicaltrials import ClinicalTrialsGovClient


def search_trial_landscape(
    *,
    disease: str,
    run_id: str,
    client: ClinicalTrialsGovClient | None = None,
    drug: str | None = None,
    target: str | None = None,
    phase: str | None = None,
    limit: int = 10,
) -> ClinicalTrialIntelligenceOutput:
    """Search CT.gov and return a typed, deterministic trial-landscape summary."""

    input_data = ClinicalTrialIntelligenceInput(
        disease=disease,
        drug=drug,
        target=target,
        phase=phase,
        limit=limit,
    )
    result = (client or ClinicalTrialsGovClient()).search_trials(input_data)
    return build_trial_landscape_output(run_id=run_id, input_data=input_data, search_result=result)


def build_trial_landscape_output(
    *,
    run_id: str,
    input_data: ClinicalTrialIntelligenceInput,
    search_result: ClinicalTrialsSearchResult,
) -> ClinicalTrialIntelligenceOutput:
    """Build conservative landscape output from normalized CT.gov records."""

    trials = search_result.trials
    claims = landscape_claims(run_id, trials)
    risks = landscape_risk_flags(run_id, trials)
    return ClinicalTrialIntelligenceOutput(
        output_id=f"trial-landscape-output-{run_id}",
        run_id=run_id,
        input=input_data,
        trials=trials,
        sources=search_result.sources,
        claims=claims,
        risk_flags=risks,
        landscape_summary=f"Found {len(trials)} ClinicalTrials.gov records for {input_data.disease}.",
        status_summary=count_summary("status", [trial.overall_status for trial in trials]),
        phase_summary=count_summary("phase", [phase for trial in trials for phase in trial.phases]),
        sponsor_summary=count_summary(
            "sponsor",
            [trial.lead_sponsor.name for trial in trials if trial.lead_sponsor],
        ),
        endpoint_summary=endpoint_summary(trials),
        population_summary=population_summary(trials),
        confidence=0.75 if trials else 0.4,
    )


def landscape_claims(run_id: str, trials: tuple[ClinicalTrialRecord, ...]) -> tuple[EvidenceClaim, ...]:
    """Generate source-backed claims for landscape records."""

    claims: list[EvidenceClaim] = []
    for trial in trials:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-{trial.nct_id}-status",
                claim_text=f"{trial.nct_id} has overall status {trial.overall_status or 'unknown'}.",
                source_ids=(trial.source_id,),
                provenance="trial_landscape.deterministic_claims",
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
                    provenance="trial_landscape.deterministic_claims",
                    confidence=0.9,
                    confidence_level="high",
                )
            )
    return tuple(claims)


def landscape_risk_flags(run_id: str, trials: tuple[ClinicalTrialRecord, ...]) -> tuple[TrialLandscapeRisk, ...]:
    """Flag basic registry risks in a trial landscape."""

    risks: list[TrialLandscapeRisk] = []
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


def count_summary(label: str, values: list[str | None]) -> str:
    """Summarize categorical values as sorted counts."""

    counts: dict[str, int] = {}
    for value in values:
        key = value or "unknown"
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return f"No {label} values were available."
    return "; ".join(f"{key}: {value}" for key, value in sorted(counts.items()))


def endpoint_summary(trials: tuple[ClinicalTrialRecord, ...]) -> str:
    """Summarize normalized primary endpoint availability."""

    total = sum(len(trial.primary_endpoints) for trial in trials)
    return f"{total} primary endpoints were normalized across {len(trials)} trials."


def population_summary(trials: tuple[ClinicalTrialRecord, ...]) -> str:
    """Summarize enrollment ranges across landscape records."""

    enrollments = [trial.enrollment_count for trial in trials if trial.enrollment_count is not None]
    if not enrollments:
        return "No enrollment counts were available."
    return f"Enrollment counts range from {min(enrollments)} to {max(enrollments)}."
