"""Compatibility workflow for Agent 3 trial-landscape mode."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from pharma_os.agents.clinical_trial_intelligence import run_clinical_trial_intelligence_agent
from pharma_os.memory import MemoryStore
from pharma_os.report import build_report
from pharma_os.schemas import (
    AgentOutput,
    ClinicalTrialIntelligenceInput,
    ClinicalTrialIntelligenceOutput,
    WorkflowRun,
)
from pharma_os.validators import (
    aggregate_validation_status,
    assign_human_gate,
    generate_confidence_flags,
    validate_numeric_provenance,
    validate_schema,
    validate_source_coverage,
)


AgentRunner = Callable[
    [ClinicalTrialIntelligenceInput, str],
    tuple[ClinicalTrialIntelligenceOutput, dict[str, str | int | float | bool | None]],
]


def run_trial_intelligence_workflow(
    input_data: ClinicalTrialIntelligenceInput,
    *,
    memory: MemoryStore | None = None,
    agent_runner: AgentRunner | None = None,
) -> ClinicalTrialIntelligenceOutput:
    """Run the Agent 3 trial-landscape component through the legacy route."""

    store = memory or MemoryStore()
    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    run = WorkflowRun(
        run_id=run_id,
        workflow_name="trial_intelligence",
        status="running",
        started_at=started_at,
        input_provenance="cli.trial_intelligence.agent3_landscape_mode",
        metadata={"limit": input_data.limit},
    )
    store.save_run(run, input_payload=input_data)

    runner = agent_runner or _default_agent_runner
    output, trace_metadata = runner(input_data, run_id)
    output = output.model_copy(update={"run_id": run_id})

    validation_results = (
        validate_schema(
            target_id=output.output_id,
            payload=output,
            schema_type=ClinicalTrialIntelligenceOutput,
            run_id=run_id,
        ),
        validate_source_coverage(
            target_id=output.output_id,
            claims=output.claims,
            source_ids={source.source_id for source in output.sources},
            run_id=run_id,
        ),
        validate_numeric_provenance(
            target_id=output.output_id,
            claims=output.claims,
            run_id=run_id,
        ),
    )
    output_text = "\n".join(
        [
            output.landscape_summary,
            output.status_summary,
            output.phase_summary,
            output.sponsor_summary,
            output.endpoint_summary,
            output.population_summary,
            *(claim.claim_text for claim in output.claims),
        ]
    )
    human_gate = assign_human_gate(
        run_id=run_id,
        workflow_name="trial_intelligence",
        validation_results=validation_results,
        output_text=output_text,
    )
    confidence_flags = generate_confidence_flags(
        run_id=run_id,
        validation_results=validation_results,
        risk_flags=output.risk_flags,
    )
    validation_status = aggregate_validation_status(validation_results)
    if human_gate and validation_status == "passed":
        validation_status = "needs_human_review"

    output = output.model_copy(
        update={
            "validation_results": validation_results,
            "confidence_flags": confidence_flags,
            "human_gate": human_gate,
            "validation_status": validation_status,
            "trace_metadata": trace_metadata,
        }
    )

    agent_output = AgentOutput(
        output_id=f"agent-output-{run_id}",
        agent_name="agent3_trial_landscape_component",
        run_id=run_id,
        provenance="PharmaOS deterministic Agent 3 trial-landscape component",
        claims=output.claims,
        sources=output.sources,
        confidence=output.confidence,
        validation_status=validation_status,
        gate_reason=human_gate.gate_reason if human_gate else None,
    )

    store.save_sources(run_id, output.sources)
    store.save_claims(run_id, output.claims)
    store.save_agent_output(agent_output, payload=output)
    store.save_validation_results(run_id, validation_results)
    store.save_confidence_flags(run_id, confidence_flags)
    store.save_human_gate(run_id, human_gate)

    completed_run = run.model_copy(
        update={
            "status": "completed" if validation_status != "failed" else "blocked",
            "completed_at": datetime.now(timezone.utc),
            "source_ids": tuple(source.source_id for source in output.sources),
            "validation_status": validation_status,
            "gate_reason": human_gate.gate_reason if human_gate else None,
        }
    )
    store.save_run(
        completed_run,
        input_payload=input_data,
        output_payload=output,
        trace_metadata=trace_metadata,
    )
    build_report(run_id, memory=store)
    return output


def _default_agent_runner(
    input_data: ClinicalTrialIntelligenceInput,
    run_id: str,
) -> tuple[ClinicalTrialIntelligenceOutput, dict[str, str | int | float | bool | None]]:
    return run_clinical_trial_intelligence_agent(input_data, run_id=run_id)
