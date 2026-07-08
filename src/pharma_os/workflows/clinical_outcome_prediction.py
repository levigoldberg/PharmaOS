"""Clinical Outcome Prediction workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from pharma_os.agents.clinical_outcome_prediction import run_clinical_outcome_prediction_agent
from pharma_os.memory import MemoryStore
from pharma_os.report import build_report
from pharma_os.schemas import (
    AgentOutput,
    ClinicalOutcomePredictionInput,
    ClinicalOutcomePredictionOutput,
    HumanGate,
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


AgentRunner = Callable[[ClinicalOutcomePredictionInput, str], ClinicalOutcomePredictionOutput]


def run_clinical_outcome_prediction_workflow(
    input_data: ClinicalOutcomePredictionInput,
    *,
    memory: MemoryStore | None = None,
    agent_runner: AgentRunner | None = None,
) -> ClinicalOutcomePredictionOutput:
    """Run the Agent 3 clinical outcome prediction workflow end to end."""

    store = memory or MemoryStore()
    run_id = str(uuid4())
    run = WorkflowRun(
        run_id=run_id,
        workflow_name="clinical_outcome_prediction",
        status="running",
        started_at=datetime.now(timezone.utc),
        input_provenance="cli.clinical_outcome_prediction",
        metadata={"nct_id": input_data.nct_id},
    )
    store.save_run(run, input_payload=input_data)

    runner = agent_runner or _default_agent_runner
    output = runner(input_data, run_id).model_copy(update={"run_id": run_id})

    validation_results = (
        validate_schema(
            target_id=output.output_id,
            payload=output,
            schema_type=ClinicalOutcomePredictionOutput,
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
    output_text = "\n".join([*(claim.claim_text for claim in output.claims), *(flag.reason for flag in output.missing_data_flags)])
    gate = assign_human_gate(
        run_id=run_id,
        workflow_name="clinical_outcome_prediction",
        validation_results=validation_results,
        output_text=output_text,
    )
    if gate is None and any(flag.severity in {"high", "critical"} for flag in output.missing_data_flags):
        gate = HumanGate(
            gate_id=f"gate-{run_id}",
            decision="needs_human_review",
            gate_reason="clinical_outcome_prediction requires human review because clinical-risk inputs are missing or low confidence.",
            required_roles=("clinical_lead", "regulatory_reviewer"),
            source_ids=tuple(source.source_id for source in output.sources),
            provenance="pharma_os.workflows.clinical_outcome_prediction.missing_data_gate",
        )
    confidence_flags = generate_confidence_flags(
        run_id=run_id,
        validation_results=validation_results,
        risk_flags=output.missing_data_flags,
    )
    validation_status = aggregate_validation_status(validation_results)
    if gate and validation_status == "passed":
        validation_status = "needs_human_review"

    output = output.model_copy(
        update={
            "validation_results": validation_results,
            "confidence_flags": confidence_flags,
            "human_gate": gate,
            "validation_status": validation_status,
        }
    )

    agent_output = AgentOutput(
        output_id=f"agent-output-{run_id}",
        agent_name="clinical_outcome_prediction_agent",
        run_id=run_id,
        provenance="PharmaOS deterministic clinical_outcome_prediction workflow",
        claims=output.claims,
        sources=output.sources,
        confidence=output.confidence,
        validation_status=validation_status,
        gate_reason=gate.gate_reason if gate else None,
    )
    store.save_sources(run_id, output.sources)
    store.save_claims(run_id, output.claims)
    store.save_agent_output(agent_output, payload=output)
    store.save_validation_results(run_id, validation_results)
    store.save_confidence_flags(run_id, confidence_flags)
    store.save_human_gate(run_id, gate)

    completed_run = run.model_copy(
        update={
            "status": "completed" if validation_status != "failed" else "blocked",
            "completed_at": datetime.now(timezone.utc),
            "source_ids": tuple(source.source_id for source in output.sources),
            "validation_status": validation_status,
            "gate_reason": gate.gate_reason if gate else None,
        }
    )
    store.save_run(completed_run, input_payload=input_data, output_payload=output)
    build_report(run_id, memory=store)
    return output


def _default_agent_runner(input_data: ClinicalOutcomePredictionInput, run_id: str) -> ClinicalOutcomePredictionOutput:
    return run_clinical_outcome_prediction_agent(input_data, run_id=run_id)
