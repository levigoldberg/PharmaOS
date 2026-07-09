"""Protocol Design Brief Agent 5 workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import ValidationError

from pharma_os.agents.protocol_design import build_search_strategy, run_protocol_design_manager_agent, select_analog_trials
from pharma_os.human_readable import build_human_readable_module_output
from pharma_os.memory import MemoryStore
from pharma_os.report import build_report
from pharma_os.schemas import (
    Agent3HandoffReference,
    Agent4HandoffReference,
    AgentOutput,
    AnalogBenchmarkBundle,
    BenchmarkFrequency,
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
    initial_source_ids = tuple(
        dict.fromkeys(
            (
                target_trial.source_id,
                agent3_source.source_id,
                agent4_source.source_id,
            )
        )
    )
    assumptions = tuple(
        assumption
        for assumption in (*agent3_output.assumptions, *agent4_output.assumptions)
        if assumption.requires_human_review or assumption.assumption_type in {"calculated", "missing", "user_reviewed"}
    )
    upstream_missing_flags = _dedupe_missing_flags((*agent3_output.missing_data_flags, *agent4_output.missing_data_flags))

    manager_result = run_protocol_design_manager_agent(
        run_id=run_id,
        target_trial=target_trial,
        agent3_output=agent3_output,
        agent4_output=agent4_output,
        source_ids=initial_source_ids,
        assumptions=assumptions,
        missing_data_flags=upstream_missing_flags,
        claims=(),
        top_k=input_data.analog_top_k,
        execute_search_plan=lambda search_plan, target_nct_id: execute_ctgov_search_plan(
            search_plan=search_plan,
            target_nct_id=target_nct_id,
        ),
        calculate_benchmark=lambda trial, candidates, selection, search_plan: calculate_analog_benchmark(
            run_id=run_id,
            target_trial=trial,
            candidates=candidates,
            selection=selection,
            search_plan=search_plan,
        ),
    )
    analog_candidates = manager_result.analog_candidates
    benchmark_bundle = manager_result.benchmark_bundle
    sources = _dedupe_sources(
        (
            target_source,
            agent3_source,
            agent4_source,
            *manager_result.analog_sources,
        )
    )
    source_ids = tuple(source.source_id for source in sources)
    missing_data_flags = _dedupe_missing_flags(
        (
            *manager_result.retrieval_flags,
            *benchmark_bundle.missing_data_flags,
            *agent3_output.missing_data_flags,
            *agent4_output.missing_data_flags,
        )
    )
    benchmark_bundle = benchmark_bundle.model_copy(
        update={
            "source_ids": tuple(source_id for source_id in benchmark_bundle.source_ids if source_id in set(source_ids)),
            "missing_data_flags": tuple(dict.fromkeys((*manager_result.retrieval_flags, *benchmark_bundle.missing_data_flags))),
        }
    )
    benchmark_bundle = _filter_benchmark_source_ids(benchmark_bundle, known_source_ids=set(source_ids))
    claims = build_protocol_design_claims(
        run_id=run_id,
        target_trial=target_trial,
        benchmark_bundle=benchmark_bundle,
        source_ids=benchmark_bundle.source_ids or source_ids,
    )
    gate = HumanGate(
        gate_id=f"gate-{run_id}",
        decision="needs_human_review",
        gate_reason="protocol_design produces a draft strategy artifact requiring human clinical, statistical, and regulatory review.",
        required_roles=("clinical_lead", "biostatistician", "regulatory_reviewer"),
        source_ids=source_ids,
        provenance="pharma_os.workflows.protocol_design.mandatory_human_gate",
    )
    brief = manager_result.protocol_design_brief.model_copy(
        update={
            "source_backed_claim_ids": tuple(claim.claim_id for claim in claims),
            "assumptions": assumptions,
            "missing_data_flags": missing_data_flags,
            "source_ids": source_ids,
            "reviewer_critique": manager_result.reviewer_critique,
        }
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
    human_readable_result = build_human_readable_module_output(
        module_name="protocol_design",
        module_display_name="Agent 5 Protocol Design",
        run_id=run_id,
        typed_output=output,
    )
    output = output.model_copy(update={"human_readable_summary": human_readable_result.output})

    agent_output = AgentOutput(
        output_id=f"agent-output-{run_id}",
        agent_name="protocol_design_agent5_workflow",
        run_id=run_id,
        provenance="PharmaOS Agent 5 protocol_design workflow with SDK-backed subagents and deterministic retrieval/math",
        claims=claims,
        sources=sources,
        confidence=output.confidence,
        validation_status=validation_status,
        gate_reason=gate.gate_reason,
    )
    store.save_sources(run_id, sources)
    store.save_claims(run_id, claims)
    for payload in manager_result.subagent_payloads:
        store.save_agent_output(
            _subagent_output_envelope(
                run_id=run_id,
                payload=payload,
                sources=sources,
                validation_status=validation_status,
            ),
            payload=payload,
        )
    store.save_agent_traces(manager_result.traces)
    store.save_agent_trace(human_readable_result.trace)
    store.save_agent_output(
        _human_readable_output_envelope(
            run_id=run_id,
            payload=human_readable_result.output,
            sources=sources,
            validation_status=validation_status,
        ),
        payload=human_readable_result.output,
    )
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
    store.save_run(
        completed_run,
        input_payload=_expanded_protocol_design_input(
            input_data=input_data,
            target_trial=target_trial,
            agent3_output=agent3_output,
            agent3_handoff=agent3_handoff,
            agent4_output=agent4_output,
            agent4_handoff=agent4_handoff,
            assumptions=assumptions,
            missing_data_flags=upstream_missing_flags,
            source_ids=initial_source_ids,
        ),
        output_payload=output,
        trace_metadata={
            "manager_agent": "ProtocolDesignManagerAgent",
            "subagent_trace_count": len(manager_result.traces),
            "subagent_output_count": len(manager_result.subagent_payloads),
            "agent_runtime_mode": _agent_runtime_mode(manager_result.traces),
            "human_readable_summary_output_id": human_readable_result.output.output_id,
        },
    )
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
        try:
            output = ClinicalOutcomePredictionOutput.model_validate_json(json.dumps(payload))
        except ValidationError:
            output = None
        if output is not None:
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
        try:
            output = DueDiligenceOutput.model_validate_json(json.dumps(payload))
        except ValidationError:
            output = None
        if output is not None:
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


def _expanded_protocol_design_input(
    *,
    input_data: ProtocolDesignInput,
    target_trial: object,
    agent3_output: ClinicalOutcomePredictionOutput,
    agent3_handoff: Agent3HandoffReference,
    agent4_output: DueDiligenceOutput,
    agent4_handoff: Agent4HandoffReference,
    assumptions: tuple[object, ...],
    missing_data_flags: tuple[object, ...],
    source_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "cli_input": input_data.model_dump(mode="json"),
        "expanded_pipeline_input": {
            "target_trial": _compact_trial_input(target_trial),
            "agent3_handoff": agent3_handoff.model_dump(mode="json"),
            "agent3_output": _compact_agent3_input(agent3_output),
            "agent4_handoff": agent4_handoff.model_dump(mode="json"),
            "agent4_output": _compact_agent4_input(agent4_output),
            "assumptions": [
                assumption.model_dump(mode="json") if hasattr(assumption, "model_dump") else str(assumption)
                for assumption in assumptions
            ],
            "missing_data_flags": [
                flag.model_dump(mode="json") if hasattr(flag, "model_dump") else str(flag)
                for flag in missing_data_flags
            ],
            "initial_source_ids": source_ids,
        },
    }


def _compact_trial_input(target_trial: object) -> dict[str, object]:
    if not hasattr(target_trial, "model_dump"):
        return {"summary": str(target_trial)}
    payload = target_trial.model_dump(mode="json")
    locations = payload.get("locations") if isinstance(payload.get("locations"), list) else []
    countries = sorted({item.get("country") for item in locations if isinstance(item, dict) and item.get("country")})
    return {
        "nct_id": payload.get("nct_id"),
        "brief_title": payload.get("brief_title"),
        "overall_status": payload.get("overall_status"),
        "phases": payload.get("phases"),
        "conditions": payload.get("conditions"),
        "enrollment_count": payload.get("enrollment_count"),
        "primary_endpoint_count": len(payload.get("primary_endpoints") or []),
        "secondary_endpoint_count": len(payload.get("secondary_endpoints") or []),
        "site_count": len(locations),
        "countries": countries[:30],
        "source_id": payload.get("source_id"),
    }


def _compact_agent3_input(output: ClinicalOutcomePredictionOutput) -> dict[str, object]:
    return {
        "output_id": output.output_id,
        "run_id": output.run_id,
        "nct_id": output.input.nct_id,
        "asset_name": output.asset_identity.asset_name,
        "endpoint_risk_level": output.endpoint_risk_assessment.risk_level,
        "enrollment_duration_risk_level": output.enrollment_duration_risk.risk_level,
        "historical_pos": output.historical_pos_estimate.probability_of_success,
        "historical_pos_lookup_key": output.historical_pos_estimate.lookup_key,
        "missing_data_flag_ids": [flag.flag_id for flag in output.missing_data_flags],
        "source_ids": [source.source_id for source in output.sources],
        "confidence": output.confidence,
        "validation_status": output.validation_status,
    }


def _compact_agent4_input(output: DueDiligenceOutput) -> dict[str, object]:
    return {
        "output_id": output.output_id,
        "run_id": output.run_id,
        "nct_id": output.input.nct_id,
        "asset_name": output.asset_identity.asset_name,
        "red_flag_ids": [flag.flag_id for flag in output.red_flags],
        "missing_data_flag_ids": [flag.flag_id for flag in output.missing_data_flags],
        "pos": {
            "probability_of_success": output.pos.probability_of_success,
            "lookup_key": output.pos.lookup_key,
        },
        "patent_loe_year": output.patent_exclusivity.estimated_loe_year,
        "annual_wac": output.pricing.annual_wac,
        "commercial_calculable": output.commercial_model.calculable,
        "rnpv_calculable": output.rnpv.calculable,
        "source_ids": [source.source_id for source in output.sources],
        "confidence": output.confidence,
        "validation_status": output.validation_status,
    }


def _agent_runtime_mode(traces: tuple[object, ...]) -> str:
    provenances = {getattr(trace, "provenance", "") for trace in traces}
    if "pharma_os.agent_runtime.openai_agents_sdk" in provenances:
        return "openai_agents_sdk"
    if "pharma_os.agent_runtime.offline" in provenances:
        return "offline"
    return "unknown"


def _filter_benchmark_source_ids(
    bundle: AnalogBenchmarkBundle,
    *,
    known_source_ids: set[str],
) -> AnalogBenchmarkBundle:
    def filtered(source_ids: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(source_id for source_id in source_ids if source_id in known_source_ids)

    def filtered_freqs(rows: tuple[BenchmarkFrequency, ...]) -> tuple[BenchmarkFrequency, ...]:
        return tuple(row.model_copy(update={"source_ids": filtered(row.source_ids)}) for row in rows)

    return bundle.model_copy(
        update={
            "source_ids": filtered(bundle.source_ids),
            "enrollment": bundle.enrollment.model_copy(update={"source_ids": filtered(bundle.enrollment.source_ids)}),
            "planned_duration_months": bundle.planned_duration_months.model_copy(update={"source_ids": filtered(bundle.planned_duration_months.source_ids)}),
            "site_count": bundle.site_count.model_copy(update={"source_ids": filtered(bundle.site_count.source_ids)}),
            "randomized_frequency": filtered_freqs(bundle.randomized_frequency),
            "blinding_frequency": filtered_freqs(bundle.blinding_frequency),
            "arm_count_distribution": filtered_freqs(bundle.arm_count_distribution),
            "primary_endpoint_family_frequency": filtered_freqs(bundle.primary_endpoint_family_frequency),
            "secondary_endpoint_family_frequency": filtered_freqs(bundle.secondary_endpoint_family_frequency),
            "comparator_categories": filtered_freqs(bundle.comparator_categories),
            "country_distribution": filtered_freqs(bundle.country_distribution),
            "status_distribution": filtered_freqs(bundle.status_distribution),
            "results_availability": filtered_freqs(bundle.results_availability),
        }
    )


def _subagent_output_envelope(
    *,
    run_id: str,
    payload: object,
    sources: tuple[SourceMetadata, ...],
    validation_status: str,
) -> AgentOutput:
    known_sources = {source.source_id: source for source in sources}
    payload_source_ids = tuple(source_id for source_id in getattr(payload, "source_ids", ()) if source_id in known_sources)
    agent_name = getattr(payload, "agent_name", None) or {
        "ProtocolDesignManagerPlan": "ProtocolDesignManagerAgent",
        "AnalogSearchPlanOutput": "AnalogSearchPlannerAgent",
        "AnalogTrialSelectionOutput": "AnalogSelectionAgent",
        "BenchmarkInterpretation": "AnalogBenchmarkInterpreterAgent",
        "ProtocolReviewerCritique": "RegulatoryCriticAgent",
        "ProtocolDesignBrief": "ProtocolBriefWriterAgent",
    }.get(payload.__class__.__name__, payload.__class__.__name__)
    return AgentOutput(
        output_id=f"agent-output-{run_id}-{getattr(payload, 'output_id', getattr(payload, 'brief_id', payload.__class__.__name__))}",
        agent_name=agent_name,
        run_id=run_id,
        provenance="PharmaOS Agent 5 subagent typed output",
        claims=(),
        sources=tuple(known_sources[source_id] for source_id in payload_source_ids),
        confidence=float(getattr(payload, "confidence", 0.5) or 0.5),
        validation_status=validation_status,  # type: ignore[arg-type]
    )


def _human_readable_output_envelope(
    *,
    run_id: str,
    payload: object,
    sources: tuple[SourceMetadata, ...],
    validation_status: str,
) -> AgentOutput:
    known_sources = {source.source_id: source for source in sources}
    payload_source_ids = tuple(source_id for source_id in getattr(payload, "source_ids", ()) if source_id in known_sources)
    return AgentOutput(
        output_id=f"agent-output-{run_id}-{getattr(payload, 'output_id', payload.__class__.__name__)}",
        agent_name="Agent5HumanReadableSummaryAgent",
        run_id=run_id,
        provenance="PharmaOS Agent 5 human-readable structured output",
        claims=(),
        sources=tuple(known_sources[source_id] for source_id in payload_source_ids),
        confidence=float(getattr(payload, "confidence", 0.5) or 0.5),
        validation_status=validation_status,  # type: ignore[arg-type]
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
