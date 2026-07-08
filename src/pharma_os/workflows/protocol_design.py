"""Protocol Design Brief Agent 5 workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from pharma_os.agents.protocol_design import (
    build_eligibility_and_schedule_sections,
    build_protocol_strategy_sections,
    build_search_strategy,
    review_protocol_design,
    select_analog_trials,
)
from pharma_os.memory import MemoryStore
from pharma_os.report import build_report
from pharma_os.schemas import (
    Agent3HandoffReference,
    Agent4HandoffReference,
    AgentOutput,
    ClinicalOutcomePredictionInput,
    ClinicalOutcomePredictionOutput,
    DueDiligenceInput,
    DueDiligenceOutput,
    HumanGate,
    ProtocolDesignInput,
    ProtocolDesignOutput,
    SourceMetadata,
    WorkflowRun,
)
from pharma_os.tools.protocol_design import (
    build_benchmark_summary,
    build_protocol_design_brief,
    build_protocol_design_claims,
    calculate_analog_benchmark,
    execute_ctgov_search_plan,
)
from pharma_os.validators import (
    aggregate_validation_status,
    generate_confidence_flags,
    validate_numeric_provenance,
    validate_protocol_design_constraints,
    validate_schema,
    validate_source_coverage,
)
from pharma_os.workflows.clinical_outcome_prediction import run_clinical_outcome_prediction_workflow
from pharma_os.workflows.due_diligence import run_due_diligence_workflow


def run_protocol_design_workflow(
    input_data: ProtocolDesignInput,
    *,
    memory: MemoryStore | None = None,
) -> ProtocolDesignOutput:
    """Run Agent 5 and return a source-grounded draft ProtocolDesignBrief."""

    store = memory or MemoryStore()
    run_id = str(uuid4())
    run = WorkflowRun(
        run_id=run_id,
        workflow_name="protocol_design",
        status="running",
        started_at=datetime.now(timezone.utc),
        input_provenance="cli.protocol_design",
        metadata={"nct_id": input_data.nct_id},
    )
    store.save_run(run, input_payload=input_data)

    agent3_output, agent3_handoff = _get_or_run_agent3(input_data, memory=store)
    agent4_output, agent4_handoff = _get_or_run_agent4(input_data, memory=store)
    target_trial = agent4_output.target_trial
    target_source = _source_for_trial(target_trial)
    agent3_source = _source_for_agent_output(
        source_id=f"agent_output:clinical_outcome_prediction:{agent3_output.output_id}",
        title="Agent 3 clinical outcome prediction output",
        output_id=agent3_output.output_id,
        run_id=agent3_output.run_id,
    )
    agent4_source = _source_for_agent_output(
        source_id=f"agent_output:due_diligence:{agent4_output.output_id}",
        title="Agent 4 due diligence output",
        output_id=agent4_output.output_id,
        run_id=agent4_output.run_id,
    )
    search_plan = build_search_strategy(
        run_id=run_id,
        target_trial=target_trial,
        agent3_output=agent3_output,
        agent4_output=agent4_output,
    )
    analog_candidates, analog_sources, retrieval_flags = execute_ctgov_search_plan(
        search_plan=search_plan,
        target_nct_id=target_trial.nct_id,
    )
    selection = select_analog_trials(
        run_id=run_id,
        target_trial=target_trial,
        candidates=analog_candidates,
        agent3_output=agent3_output,
        agent4_output=agent4_output,
        search_plan=search_plan,
        top_k=input_data.analog_top_k,
    )
    benchmark_bundle = calculate_analog_benchmark(
        run_id=run_id,
        target_trial=target_trial,
        candidates=analog_candidates,
        selection=selection,
        search_plan=search_plan,
    )
    benchmark_summary = build_benchmark_summary(benchmark_bundle)
    source_ids = tuple(
        dict.fromkeys(
            (
                target_trial.source_id,
                agent3_source.source_id,
                agent4_source.source_id,
                *benchmark_bundle.source_ids,
            )
        )
    )
    strategy_sections = build_protocol_strategy_sections(
        run_id=run_id,
        target_trial=target_trial,
        source_ids=source_ids,
        benchmark_summary=benchmark_summary,
        agent3_output=agent3_output,
        agent4_output=agent4_output,
    )
    eligibility_sections = build_eligibility_and_schedule_sections(
        run_id=run_id,
        source_ids=benchmark_bundle.source_ids or source_ids,
        inclusion_themes=benchmark_bundle.inclusion_themes,
        exclusion_themes=benchmark_bundle.exclusion_themes,
        safety_themes=benchmark_bundle.safety_exclusion_themes,
    )
    reviewer_critique = review_protocol_design(
        run_id=run_id,
        source_ids=source_ids,
        analog_limitations=benchmark_bundle.limitations,
        agent3_output=agent3_output,
        agent4_output=agent4_output,
    )
    sources = _dedupe_sources(
        (
            target_source,
            agent3_source,
            agent4_source,
            *analog_sources,
        )
    )
    source_ids = tuple(source.source_id for source in sources)
    missing_data_flags = _dedupe_missing_flags(
        (
            *retrieval_flags,
            *benchmark_bundle.missing_data_flags,
            *agent3_output.missing_data_flags,
            *agent4_output.missing_data_flags,
        )
    )
    benchmark_bundle = benchmark_bundle.model_copy(
        update={
            "source_ids": tuple(source_id for source_id in benchmark_bundle.source_ids if source_id in set(source_ids)),
            "missing_data_flags": tuple(dict.fromkeys((*retrieval_flags, *benchmark_bundle.missing_data_flags))),
        }
    )
    claims = build_protocol_design_claims(
        run_id=run_id,
        target_trial=target_trial,
        benchmark_bundle=benchmark_bundle,
        source_ids=benchmark_bundle.source_ids or source_ids,
    )
    assumptions = tuple(
        assumption
        for assumption in (*agent3_output.assumptions, *agent4_output.assumptions)
        if assumption.requires_human_review or assumption.assumption_type in {"calculated", "missing", "user_reviewed"}
    )
    gate = HumanGate(
        gate_id=f"gate-{run_id}",
        decision="needs_human_review",
        gate_reason="protocol_design produces a draft strategy artifact requiring human clinical, statistical, and regulatory review.",
        required_roles=("clinical_lead", "biostatistician", "regulatory_reviewer"),
        source_ids=source_ids,
        provenance="pharma_os.workflows.protocol_design.mandatory_human_gate",
    )
    brief = build_protocol_design_brief(
        run_id=run_id,
        target_trial=target_trial,
        strategy_sections=strategy_sections,
        eligibility_sections=eligibility_sections,
        reviewer_critique=reviewer_critique,
        benchmark_bundle=benchmark_bundle,
        claims=claims,
        assumptions=assumptions,
        missing_data_flags=missing_data_flags,
        source_ids=source_ids,
    )
    output = ProtocolDesignOutput(
        output_id=f"protocol-design-output-{run_id}",
        run_id=run_id,
        input=input_data,
        target_trial=target_trial,
        agent3_handoff=agent3_handoff,
        agent4_handoff=agent4_handoff,
        analog_candidates=analog_candidates,
        analog_benchmark_bundle=benchmark_bundle,
        protocol_design_brief=brief,
        sources=sources,
        claims=claims,
        assumptions=assumptions,
        missing_data_flags=missing_data_flags,
        human_gate=gate,
        confidence=_confidence(missing_data_flags, benchmark_bundle.confidence),
    )
    validation_results = (
        validate_schema(
            target_id=output.output_id,
            payload=output,
            schema_type=ProtocolDesignOutput,
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
        *validate_protocol_design_constraints(run_id=run_id, output=output),
    )
    confidence_flags = generate_confidence_flags(
        run_id=run_id,
        validation_results=validation_results,
        risk_flags=missing_data_flags,
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
        agent_name="protocol_design_agent5_workflow",
        run_id=run_id,
        provenance="PharmaOS deterministic protocol_design workflow",
        claims=claims,
        sources=sources,
        confidence=output.confidence,
        validation_status=validation_status,
        gate_reason=gate.gate_reason,
    )
    store.save_sources(run_id, sources)
    store.save_claims(run_id, claims)
    store.save_agent_output(agent_output, payload=output)
    store.save_validation_results(run_id, validation_results)
    store.save_confidence_flags(run_id, confidence_flags)
    store.save_human_gate(run_id, gate)

    completed_run = run.model_copy(
        update={
            "status": "completed" if validation_status != "failed" else "blocked",
            "completed_at": datetime.now(timezone.utc),
            "source_ids": source_ids,
            "validation_status": validation_status,
            "gate_reason": gate.gate_reason,
        }
    )
    store.save_run(completed_run, input_payload=input_data, output_payload=output)
    build_report(run_id, memory=store)
    return output


def _get_or_run_agent3(
    input_data: ProtocolDesignInput,
    *,
    memory: MemoryStore,
) -> tuple[ClinicalOutcomePredictionOutput, Agent3HandoffReference]:
    latest = None if input_data.refresh_agent3 else memory.get_latest_workflow_output(
        workflow_name="clinical_outcome_prediction",
        nct_id=input_data.nct_id,
    )
    if latest is not None:
        agent3_run, payload = latest
        output = ClinicalOutcomePredictionOutput.model_validate_json(json.dumps(payload))
        return output, Agent3HandoffReference(
            agent3_run_id=agent3_run.run_id,
            agent3_output_id=output.output_id,
            nct_id=output.input.nct_id,
            generated_or_reused="reused",
            retrieved_from_memory=True,
            source_ids=tuple(source.source_id for source in output.sources),
            confidence=output.confidence,
        )

    output = run_clinical_outcome_prediction_workflow(
        ClinicalOutcomePredictionInput(
            nct_id=input_data.nct_id,
            pos_workbook_path=input_data.pos_workbook_path,
        ),
        memory=memory,
    )
    return output, Agent3HandoffReference(
        agent3_run_id=output.run_id,
        agent3_output_id=output.output_id,
        nct_id=output.input.nct_id,
        generated_or_reused="generated",
        retrieved_from_memory=False,
        source_ids=tuple(source.source_id for source in output.sources),
        confidence=output.confidence,
    )


def _get_or_run_agent4(
    input_data: ProtocolDesignInput,
    *,
    memory: MemoryStore,
) -> tuple[DueDiligenceOutput, Agent4HandoffReference]:
    latest = None if input_data.refresh_agent4 else memory.get_latest_workflow_output(
        workflow_name="due_diligence",
        nct_id=input_data.nct_id,
    )
    if latest is not None:
        agent4_run, payload = latest
        output = DueDiligenceOutput.model_validate_json(json.dumps(payload))
        return output, Agent4HandoffReference(
            agent4_run_id=agent4_run.run_id,
            agent4_output_id=output.output_id,
            nct_id=output.input.nct_id,
            generated_or_reused="reused",
            retrieved_from_memory=True,
            source_ids=tuple(source.source_id for source in output.sources),
            confidence=output.confidence,
        )

    output = run_due_diligence_workflow(
        DueDiligenceInput(
            nct_id=input_data.nct_id,
            pos_workbook_path=input_data.pos_workbook_path,
            wac_data_path=input_data.wac_data_path,
            annual_patients=input_data.annual_patients,
            peak_penetration=input_data.peak_penetration,
            gross_to_net=input_data.gross_to_net,
            operating_margin=input_data.operating_margin,
            discount_rate=input_data.discount_rate,
            development_cost=input_data.development_cost,
            launch_year=input_data.launch_year,
            loe_year=input_data.loe_year,
            refresh_agent3=input_data.refresh_agent3,
        ),
        memory=memory,
    )
    return output, Agent4HandoffReference(
        agent4_run_id=output.run_id,
        agent4_output_id=output.output_id,
        nct_id=output.input.nct_id,
        generated_or_reused="generated",
        retrieved_from_memory=False,
        source_ids=tuple(source.source_id for source in output.sources),
        confidence=output.confidence,
    )


def _source_for_trial(trial: object) -> SourceMetadata:
    return SourceMetadata(
        source_id=getattr(trial, "source_id"),
        title=getattr(trial, "brief_title", None) or getattr(trial, "official_title", None) or getattr(trial, "nct_id"),
        url=f"https://clinicaltrials.gov/study/{getattr(trial, 'nct_id')}",
        authors=tuple(
            sponsor.name
            for sponsor in (getattr(trial, "lead_sponsor", None), *getattr(trial, "collaborators", ()))
            if sponsor is not None
        ),
        retrieved_at=datetime.now(timezone.utc),
        provenance="ClinicalTrials.gov API v2 protocolSection",
        source_type="clinical_trial_registry",
        version="v2",
    )


def _source_for_agent_output(*, source_id: str, title: str, output_id: str, run_id: str) -> SourceMetadata:
    return SourceMetadata(
        source_id=source_id,
        title=title,
        provenance=f"Scientific Memory workflow output {output_id} from run {run_id}",
        source_type="agent_output",
        version="local",
    )


def _dedupe_sources(sources: tuple[SourceMetadata, ...]) -> tuple[SourceMetadata, ...]:
    deduped: dict[str, SourceMetadata] = {}
    for source in sources:
        deduped[source.source_id] = source
    return tuple(deduped.values())


def _dedupe_missing_flags(flags: tuple[object, ...]) -> tuple[object, ...]:
    deduped: dict[str, object] = {}
    for flag in flags:
        deduped[getattr(flag, "flag_id")] = flag
    return tuple(deduped.values())


def _confidence(flags: tuple[object, ...], benchmark_confidence: float) -> float:
    if any(getattr(flag, "severity", None) == "critical" for flag in flags):
        return 0.15
    high = sum(1 for flag in flags if getattr(flag, "severity", None) == "high")
    medium = sum(1 for flag in flags if getattr(flag, "severity", None) == "medium")
    return max(0.15, min(0.8, benchmark_confidence - high * 0.12 - medium * 0.04))
