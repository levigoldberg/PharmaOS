"""PharmaOS-native Trial Intelligence + Due Diligence workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pharma_os.memory import MemoryStore
from pharma_os.report import build_report
from pharma_os.schemas import (
    AgentOutput,
    DueDiligenceInput,
    DueDiligenceOutput,
    EvidenceClaim,
    SourceMetadata,
    WorkflowRun,
)
from pharma_os.tools.clinicaltrials import ClinicalTrialsGovClient
from pharma_os.tools.due_diligence import (
    build_commercial_model,
    build_rnpv,
    lookup_pos,
    lookup_pricing,
    resolve_asset_identity,
    search_patent_exclusivity,
)
from pharma_os.tools.rules import config_source
from pharma_os.validators import (
    aggregate_validation_status,
    assign_human_gate,
    generate_confidence_flags,
    validate_numeric_provenance,
    validate_schema,
    validate_source_coverage,
)


def run_due_diligence_workflow(
    input_data: DueDiligenceInput,
    *,
    memory: MemoryStore | None = None,
) -> DueDiligenceOutput:
    """Run a deterministic due-diligence workflow with real source retrieval."""

    store = memory or MemoryStore()
    run_id = str(uuid4())
    run = WorkflowRun(
        run_id=run_id,
        workflow_name="due_diligence",
        status="running",
        started_at=datetime.now(timezone.utc),
        input_provenance="cli.due_diligence",
        metadata={"nct_id": input_data.nct_id},
    )
    store.save_run(run, input_payload=input_data)

    trial = ClinicalTrialsGovClient().fetch_trial(input_data.nct_id)
    asset_identity, identity_sources = resolve_asset_identity(trial)
    patent_exclusivity, patent_sources = search_patent_exclusivity(
        asset_identity,
        loe_year_override=input_data.loe_year,
    )
    pos, pos_source = lookup_pos(
        trial,
        asset_identity,
        workbook_path=input_data.pos_workbook_path,
    )
    pricing, pricing_sources = lookup_pricing(
        asset_identity,
        wac_data_path=input_data.wac_data_path,
    )
    commercial_model = build_commercial_model(
        annual_patients=input_data.annual_patients,
        peak_penetration=input_data.peak_penetration,
        gross_to_net=input_data.gross_to_net,
        pricing=pricing,
    )
    rnpv = build_rnpv(
        commercial=commercial_model,
        pos=pos,
        patent=patent_exclusivity,
        launch_year=input_data.launch_year,
        loe_year=input_data.loe_year,
        discount_rate=input_data.discount_rate,
        operating_margin=input_data.operating_margin,
        development_cost=input_data.development_cost,
        phase=pos.current_phase or (trial.phases[0] if trial.phases else None),
    )

    user_source = SourceMetadata(
        source_id=f"user_input:{run_id}",
        title="Reviewed due-diligence CLI assumptions",
        provenance="CLI supplied due_diligence assumptions",
        source_type="human_input",
        version="local",
    )
    config_sources = _config_sources_from_assumptions((*commercial_model.assumptions, *rnpv.assumptions))
    sources = _dedupe_sources((*identity_sources, *patent_sources, pos_source, *pricing_sources, *config_sources, user_source))
    missing_data_flags = (
        *asset_identity.missing_data_flags,
        *patent_exclusivity.missing_data_flags,
        *pos.missing_data_flags,
        *pricing.missing_data_flags,
        *commercial_model.missing_data_flags,
        *rnpv.missing_data_flags,
    )
    assumptions = tuple(
        assumption.model_copy(update={"source_ids": (user_source.source_id,)})
        if not assumption.source_ids and (assumption.provenance.startswith("cli.") or "override" in assumption.provenance.casefold())
        else assumption
        for assumption in (*commercial_model.assumptions, *rnpv.assumptions)
    )
    commercial_model = commercial_model.model_copy(
        update={
            "assumptions": tuple(
                assumption.model_copy(update={"source_ids": (user_source.source_id,)})
                if not assumption.source_ids
                else assumption
                for assumption in commercial_model.assumptions
            )
        }
    )
    rnpv = rnpv.model_copy(
        update={
            "assumptions": tuple(
                assumption.model_copy(update={"source_ids": (user_source.source_id,)})
                if not assumption.source_ids and (assumption.provenance.startswith("cli.") or "override" in assumption.provenance.casefold())
                else assumption
                for assumption in rnpv.assumptions
            ),
            "source_ids": tuple(dict.fromkeys((*rnpv.source_ids, user_source.source_id))),
        }
    )
    claims = _claims(
        run_id=run_id,
        trial=trial,
        asset=asset_identity,
        patent=patent_exclusivity,
        pos=pos,
        pricing=pricing,
        commercial=commercial_model,
        rnpv=rnpv,
    )
    output = DueDiligenceOutput(
        output_id=f"due-diligence-output-{run_id}",
        run_id=run_id,
        input=input_data,
        trial=trial,
        asset_identity=asset_identity,
        patent_exclusivity=patent_exclusivity,
        pos=pos,
        pricing=pricing,
        commercial_model=commercial_model,
        rnpv=rnpv,
        sources=sources,
        claims=claims,
        assumptions=assumptions,
        missing_data_flags=missing_data_flags,
        confidence=_confidence(missing_data_flags),
    )

    validation_results = (
        validate_schema(
            target_id=output.output_id,
            payload=output,
            schema_type=DueDiligenceOutput,
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
    output_text = "\n".join([*(claim.claim_text for claim in claims), *(flag.reason for flag in missing_data_flags)])
    gate = assign_human_gate(
        run_id=run_id,
        workflow_name="due_diligence",
        validation_results=validation_results,
        output_text=output_text,
    )
    if missing_data_flags and gate is None:
        from pharma_os.schemas import HumanGate

        gate = HumanGate(
            gate_id=f"gate-{run_id}",
            decision="needs_human_review",
            gate_reason="due_diligence requires human review because diligence-critical inputs are missing or low confidence.",
            required_roles=("clinical_lead", "commercial_lead", "ip_counsel"),
            source_ids=tuple(source.source_id for source in sources),
            provenance="pharma_os.workflows.due_diligence.missing_data_gate",
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
        agent_name="due_diligence_deterministic_workflow",
        run_id=run_id,
        provenance="PharmaOS deterministic due_diligence workflow",
        claims=claims,
        sources=sources,
        confidence=output.confidence,
        validation_status=validation_status,
        gate_reason=gate.gate_reason if gate else None,
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
            "source_ids": tuple(source.source_id for source in sources),
            "validation_status": validation_status,
            "gate_reason": gate.gate_reason if gate else None,
        }
    )
    store.save_run(completed_run, input_payload=input_data, output_payload=output)
    build_report(run_id, memory=store)
    return output


def _claims(
    *,
    run_id: str,
    trial: object,
    asset: object,
    patent: object,
    pos: object,
    pricing: object,
    commercial: object,
    rnpv: object,
) -> tuple[EvidenceClaim, ...]:
    claims: list[EvidenceClaim] = [
        EvidenceClaim(
            claim_id=f"claim-{run_id}-trial-status",
            claim_text=f"{trial.nct_id} has ClinicalTrials.gov status {trial.overall_status or 'unknown'}.",
            source_ids=(trial.source_id,),
            provenance="due_diligence.ctgov",
            confidence=0.95,
            confidence_level="very_high",
        )
    ]
    if asset.asset_name:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-asset",
                claim_text=f"The inferred asset for {trial.nct_id} is {asset.asset_name}.",
                source_ids=asset.source_ids[:1] or (trial.source_id,),
                provenance="due_diligence.asset_identity",
                confidence=asset.confidence,
                confidence_level="high" if asset.confidence >= 0.7 else "low",
            )
        )
    if pos.probability_of_success is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-pos",
                claim_text=f"Workbook PoS for {pos.disease_area} {pos.current_phase} is {pos.probability_of_success:.3f}.",
                source_ids=pos.source_ids,
                provenance="due_diligence.pos_workbook",
                confidence=pos.confidence,
                confidence_level="high",
            )
        )
    if pricing.annual_wac is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-pricing",
                claim_text=f"Annual WAC benchmark is {pricing.annual_wac:.2f} USD based on local WAC and openFDA dosing evidence.",
                source_ids=pricing.source_ids,
                provenance="due_diligence.pricing",
                confidence=pricing.confidence,
                confidence_level="medium",
            )
        )
    if commercial.peak_net_sales is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-commercial",
                claim_text=f"Deterministic peak net sales are {commercial.peak_net_sales:.2f} USD under reviewed assumptions.",
                source_ids=commercial.source_ids,
                provenance="due_diligence.commercial_model",
                confidence=commercial.confidence,
                confidence_level="medium",
            )
        )
    if rnpv.rnpv is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-rnpv",
                claim_text=f"Deterministic rNPV is {rnpv.rnpv:.2f} USD under reviewed assumptions.",
                source_ids=rnpv.source_ids,
                provenance="due_diligence.rnpv",
                confidence=rnpv.confidence,
                confidence_level="medium",
            )
        )
    if patent.estimated_loe_year is not None:
        claims.append(
            EvidenceClaim(
                claim_id=f"claim-{run_id}-loe",
                claim_text=f"Reviewed LOE year is {patent.estimated_loe_year}.",
                source_ids=patent.source_ids or (trial.source_id,),
                provenance="due_diligence.patent_exclusivity",
                confidence=patent.confidence,
                confidence_level="medium" if patent.source_ids else "low",
            )
        )
    return tuple(claims)


def _dedupe_sources(sources: tuple[object, ...]) -> tuple[object, ...]:
    deduped: dict[str, object] = {}
    for source in sources:
        deduped[getattr(source, "source_id")] = source
    return tuple(deduped.values())


def _confidence(flags: tuple[object, ...]) -> float:
    if any(getattr(flag, "severity", None) == "critical" for flag in flags):
        return 0.15
    high = sum(1 for flag in flags if getattr(flag, "severity", None) == "high")
    medium = sum(1 for flag in flags if getattr(flag, "severity", None) == "medium")
    return max(0.1, 0.85 - high * 0.15 - medium * 0.05)


def _config_sources_from_assumptions(assumptions: tuple[object, ...]) -> tuple[SourceMetadata, ...]:
    sources = []
    for assumption in assumptions:
        for source_id in getattr(assumption, "source_ids", ()):
            if source_id == "config:due_diligence:default_archetypes":
                sources.append(config_source("default_archetypes.yaml", section="due_diligence"))
            elif source_id == "config:due_diligence:rnpv_assumptions_config":
                sources.append(config_source("rnpv_assumptions_config.yaml", section="due_diligence"))
    return _dedupe_sources(tuple(sources))  # type: ignore[return-value]
