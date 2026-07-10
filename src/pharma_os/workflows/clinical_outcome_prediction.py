"""Clinical Outcome Prediction workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel

from pharma_os.agents.clinical_outcome_prediction import (
    ClinicalOutcomePredictionAgentResult,
    run_clinical_outcome_prediction_agent_result,
)
from pharma_os.execution_modes import (
    execution_mode_for_payload,
    execution_mode_summary_for_mode,
    primary_execution_mode,
    summarize_execution_modes,
)
from pharma_os.human_readable import build_human_readable_module_output
from pharma_os.memory import MemoryStore
from pharma_os.report import build_report
from pharma_os.schemas import (
    AgentOutput,
    ClinicalOutcomePredictionInput,
    ClinicalOutcomePredictionOutput,
    HumanGate,
    SourceMetadata,
    WorkflowRun,
)
from pharma_os.validators import (
    aggregate_validation_status,
    assign_human_gate,
    generate_confidence_flags,
    validate_clinical_outcome_constraints,
    validate_numeric_provenance,
    validate_schema,
    validate_source_coverage,
)


AgentRunner = Callable[[ClinicalOutcomePredictionInput, str], ClinicalOutcomePredictionOutput | ClinicalOutcomePredictionAgentResult]


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
    runner_result = runner(input_data, run_id)
    if isinstance(runner_result, ClinicalOutcomePredictionAgentResult):
        output = runner_result.output.model_copy(update={"run_id": run_id})
        subagent_payloads = runner_result.subagent_payloads
        agent_traces = runner_result.traces
    else:
        output = runner_result.model_copy(update={"run_id": run_id})
        subagent_payloads = ()
        agent_traces = ()

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
        *validate_clinical_outcome_constraints(
            run_id=run_id,
            output=output,
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
    human_readable_result = build_human_readable_module_output(
        module_name="clinical_outcome_prediction",
        module_display_name="Agent 3 Clinical Outcome Prediction",
        run_id=run_id,
        typed_output=output,
    )
    all_agent_traces = (*agent_traces, human_readable_result.trace)
    execution_mode_summary = summarize_execution_modes(all_agent_traces)
    human_readable_output = human_readable_result.output.model_copy(
        update={"execution_mode": human_readable_result.trace.execution_mode}
    )
    output = output.model_copy(
        update={
            "human_readable_summary": human_readable_output,
            "execution_mode_summary": execution_mode_summary,
        }
    )

    agent_output = AgentOutput(
        output_id=f"agent-output-{run_id}",
        agent_name="clinical_outcome_prediction_agent3_workflow",
        run_id=run_id,
        provenance="PharmaOS Agent 3 clinical_outcome_prediction workflow with SDK-backed clinical reasoning and deterministic retrieval/math",
        claims=output.claims,
        sources=output.sources,
        confidence=output.confidence,
        validation_status=validation_status,
        gate_reason=gate.gate_reason if gate else None,
        execution_mode=primary_execution_mode(execution_mode_summary),
        execution_mode_summary=execution_mode_summary,
    )
    store.save_sources(run_id, output.sources)
    store.save_claims(run_id, output.claims)
    for payload in subagent_payloads:
        store.save_agent_output(
            _subagent_output_envelope(
                run_id=run_id,
                payload=payload,
                sources=output.sources,
                validation_status=validation_status,
                traces=agent_traces,
            ),
            payload=payload,
        )
    store.save_agent_traces(agent_traces)
    store.save_agent_trace(human_readable_result.trace)
    store.save_agent_output(
        _human_readable_output_envelope(
            run_id=run_id,
            payload=human_readable_result.output,
            sources=output.sources,
            validation_status=validation_status,
            execution_mode=human_readable_result.trace.execution_mode,
        ),
        payload=human_readable_output,
    )
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
    store.save_run(
        completed_run,
        input_payload=input_data,
        output_payload=output,
        trace_metadata={
            "manager_agent": "ClinicalOutcomeManagerAgent",
            "subagent_trace_count": len(agent_traces),
            "subagent_output_count": len(subagent_payloads),
            "human_readable_summary_output_id": human_readable_output.output_id,
            "execution_mode_summary": execution_mode_summary.model_dump(mode="json"),
        },
    )
    if completed_run.status == "completed" and validation_status != "failed":
        store.mark_workflow_output_current(
            workflow_name="clinical_outcome_prediction",
            nct_id=input_data.nct_id,
            current_run_id=run_id,
            current_output_id=output.output_id,
        )
    build_report(run_id, memory=store)
    return output


def _default_agent_runner(input_data: ClinicalOutcomePredictionInput, run_id: str) -> ClinicalOutcomePredictionAgentResult:
    return run_clinical_outcome_prediction_agent_result(input_data, run_id=run_id)


def _subagent_output_envelope(
    *,
    run_id: str,
    payload: BaseModel,
    sources: tuple[SourceMetadata, ...],
    validation_status: str,
    traces: tuple[object, ...],
) -> AgentOutput:
    known_sources = {source.source_id: source for source in sources}
    payload_source_ids = tuple(source_id for source_id in getattr(payload, "source_ids", ()) if source_id in known_sources)
    agent_name = {
        "ClinicalOutcomeManagerPlan": "ClinicalOutcomeManagerAgent",
        "AssetIdentityAdjudication": "AssetIdentityAdjudicatorAgent",
        "EndpointRiskAssessment": "EndpointRiskAgent",
        "ComparatorRelevanceOutput": "ComparatorRelevanceAgent",
        "EnrollmentDurationRisk": "EnrollmentFeasibilityAgent",
        "SafetyContext": "SafetyContextAgent",
        "FailureModeClassification": "FailureModeSynthesisAgent",
    }.get(payload.__class__.__name__, payload.__class__.__name__)
    output_id = getattr(payload, "output_id", None) or f"{agent_name}-{payload.__class__.__name__}"
    execution_mode = execution_mode_for_payload(payload, traces)
    return AgentOutput(
        output_id=f"agent-output-{run_id}-{output_id}",
        agent_name=agent_name,
        run_id=run_id,
        provenance="PharmaOS Agent 3 subagent typed output",
        claims=(),
        sources=tuple(known_sources[source_id] for source_id in payload_source_ids),
        confidence=float(getattr(payload, "confidence", 0.5) or 0.5),
        validation_status=validation_status,  # type: ignore[arg-type]
        execution_mode=execution_mode,
        execution_mode_summary=execution_mode_summary_for_mode(execution_mode),
    )


def _human_readable_output_envelope(
    *,
    run_id: str,
    payload: BaseModel,
    sources: tuple[SourceMetadata, ...],
    validation_status: str,
    execution_mode: str,
) -> AgentOutput:
    known_sources = {source.source_id: source for source in sources}
    payload_source_ids = tuple(source_id for source_id in getattr(payload, "source_ids", ()) if source_id in known_sources)
    return AgentOutput(
        output_id=f"agent-output-{run_id}-{payload.output_id}",
        agent_name="Agent3HumanReadableSummaryAgent",
        run_id=run_id,
        provenance="PharmaOS Agent 3 human-readable structured output",
        claims=(),
        sources=tuple(known_sources[source_id] for source_id in payload_source_ids),
        confidence=float(getattr(payload, "confidence", 0.5) or 0.5),
        validation_status=validation_status,  # type: ignore[arg-type]
        execution_mode=execution_mode,  # type: ignore[arg-type]
        execution_mode_summary=execution_mode_summary_for_mode(execution_mode),  # type: ignore[arg-type]
    )
