"""Human-readable structured summaries for PharmaOS workflow modules."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from pharma_os.agent_runtime import AgentRuntimeConfig, StructuredAgentResult, run_structured_llm_call, runtime_config_for_live_agents
from pharma_os.schemas import HumanReadableFinding, HumanReadableModuleOutput


ModuleName = Literal["clinical_outcome_prediction", "due_diligence", "protocol_design", "trial_intelligence"]


def build_human_readable_module_output(
    *,
    module_name: ModuleName,
    module_display_name: str,
    run_id: str,
    typed_output: BaseModel,
) -> StructuredAgentResult:
    """Generate a source-grounded human-facing summary from a typed workflow output."""

    source_output_id = str(getattr(typed_output, "output_id", f"{module_name}-{run_id}"))
    source_ids = _source_ids(typed_output)
    existing = getattr(typed_output, "human_readable_summary", None)
    if isinstance(existing, HumanReadableModuleOutput) and existing.source_output_id == source_output_id:
        return run_structured_llm_call(
            agent_name=f"{_module_agent_prefix(module_name)}HumanReadableSummaryAgent",
            instructions=_instructions(module_display_name),
            payload={"summary_reuse": True, "source_output_id": source_output_id},
            output_type=HumanReadableModuleOutput,
            run_id=run_id,
            input_summary=f"Reuse existing human-readable structured summary for {module_display_name}.",
            config=AgentRuntimeConfig(
                model="deterministic-reuse",
                model_route="human_summary",
                disabled=True,
                provenance="pharma_os.human_readable.reuse_existing_summary",
            ),
            offline_output=existing,
            source_ids=source_ids,
            confidence=existing.confidence,
            rationale_summary=f"{module_display_name} human-readable summary already matched this typed output.",
        )
    fallback = _fallback_summary(
        module_name=module_name,
        module_display_name=module_display_name,
        run_id=run_id,
        source_output_id=source_output_id,
        typed_output=typed_output,
        source_ids=source_ids,
    )
    return run_structured_llm_call(
        agent_name=f"{_module_agent_prefix(module_name)}HumanReadableSummaryAgent",
        instructions=_instructions(module_display_name),
        payload={
            "module_name": module_name,
            "module_display_name": module_display_name,
            "typed_workflow_output": typed_output.model_dump(mode="json"),
            "summary_requirements": {
                "audience": "human reviewer examining PharmaOS module outputs",
                "use_only_supplied_data": True,
                "preserve_source_ids": True,
                "avoid_hidden_reasoning": True,
                "avoid_final_decisions": True,
            },
        },
        output_type=HumanReadableModuleOutput,
        run_id=run_id,
        input_summary=f"Create a human-readable structured summary for {module_display_name}.",
        config=runtime_config_for_live_agents(
            disabled_provenance="pharma_os.human_readable",
            model_route="human_summary",
        ),
        offline_output=fallback,
        source_ids=source_ids,
        confidence=fallback.confidence,
        rationale_summary=f"{module_display_name} human-readable summary generated from typed workflow output only.",
    )


def _instructions(module_display_name: str) -> str:
    return (
        f"You are writing a human-readable structured summary for {module_display_name}. "
        "Return only the requested strict structured output. Use only the supplied typed workflow output and source IDs. "
        "Do not invent facts, rates, clinical conclusions, legal conclusions, investment recommendations, approval predictions, "
        "go/no-go decisions, or final protocol decisions. Preserve important caveats, missing evidence, validation status, "
        "human gates, and handoff context. Keep the text readable for a human reviewer and do not expose hidden reasoning."
    )


def _fallback_summary(
    *,
    module_name: ModuleName,
    module_display_name: str,
    run_id: str,
    source_output_id: str,
    typed_output: BaseModel,
    source_ids: tuple[str, ...],
) -> HumanReadableModuleOutput:
    values = typed_output.model_dump(mode="json")
    confidence = float(values.get("confidence") or 0.5)
    validation_status = str(values.get("validation_status") or "not_run")
    limitations = _limitations(values)
    review_questions = _review_questions(values)
    findings = _findings(module_name, values, source_ids, confidence)
    takeaways = tuple(finding.title for finding in findings[:5]) or (f"{module_display_name} typed output is available for review.",)
    return HumanReadableModuleOutput(
        output_id=f"human-readable-{module_name}-{run_id}",
        run_id=run_id,
        module_name=module_name,
        module_display_name=module_display_name,
        source_output_id=source_output_id,
        headline=_headline(module_name, values, validation_status),
        plain_language_summary=_plain_summary(module_name, values, validation_status),
        key_takeaways=takeaways,
        key_findings=tuple(findings),
        handoff_summary=_handoff_summary(module_name, values),
        limitations=limitations,
        human_review_questions=review_questions,
        source_ids=source_ids,
        confidence=max(0.0, min(1.0, confidence)),
        provenance="pharma_os.human_readable.offline_fallback_from_typed_output",
    )


def _source_ids(output: BaseModel) -> tuple[str, ...]:
    explicit = tuple(str(source_id) for source_id in getattr(output, "source_ids", ()) if source_id)
    sources = tuple(str(getattr(source, "source_id", "")) for source in getattr(output, "sources", ()) if getattr(source, "source_id", ""))
    return tuple(dict.fromkeys((*explicit, *sources)))


def _headline(module_name: ModuleName, values: dict[str, Any], validation_status: str) -> str:
    nct_id = _nct_id(values)
    if module_name == "clinical_outcome_prediction":
        risk = _nested(values, "endpoint_risk_assessment", "risk_level") or "unknown endpoint risk"
        return f"Agent 3 clinical outcome summary for {nct_id}: {risk}; validation {validation_status}."
    if module_name == "due_diligence":
        red_flags = len(values.get("red_flags") or ())
        return f"Agent 4 diligence summary for {nct_id}: {red_flags} red flags; validation {validation_status}."
    if module_name == "protocol_design":
        analogs = len(_nested(values, "analog_benchmark_bundle", "selected_analog_ids") or ())
        proposed = _nested(values, "next_study_intent", "proposed_next_stage") or _nested(values, "protocol_design_brief", "next_study_intent", "proposed_next_stage") or "proposed next study"
        return f"Agent 5 protocol design summary for {nct_id}: {proposed}; {analogs} selected analogs; validation {validation_status}."
    trials = len(values.get("trials") or ())
    return f"Trial intelligence summary for {nct_id}: {trials} trials; validation {validation_status}."


def _plain_summary(module_name: ModuleName, values: dict[str, Any], validation_status: str) -> str:
    gate = values.get("human_gate") or {}
    gate_reason = gate.get("gate_reason") if isinstance(gate, dict) else None
    suffix = f" Human review gate: {gate_reason}" if gate_reason else ""
    if module_name == "clinical_outcome_prediction":
        asset = _nested(values, "asset_identity", "asset_name") or "unknown asset"
        failure = _nested(values, "failure_mode_classification", "summary") or "failure-mode summary unavailable"
        return f"Agent 3 summarizes clinical risk context for {asset}. {failure} Validation status is {validation_status}.{suffix}"
    if module_name == "due_diligence":
        memo = _nested(values, "asset_memo", "summary") or "Asset memo summary unavailable."
        return f"Agent 4 combines clinical, safety, IP, pricing, commercial, and rNPV diligence. {memo} Validation status is {validation_status}.{suffix}"
    if module_name == "protocol_design":
        title = _nested(values, "protocol_design_brief", "title") or "draft protocol design brief"
        objective = _nested(values, "next_study_intent", "development_objective") or _nested(values, "protocol_design_brief", "next_study_intent", "development_objective")
        objective_sentence = f" Development objective: {objective}." if objective else ""
        return f"Agent 5 produced {title} from Agent 3/4 handoffs and analog benchmarking.{objective_sentence} Validation status is {validation_status}.{suffix}"
    return f"Trial intelligence generated a source-backed landscape summary. Validation status is {validation_status}.{suffix}"


def _handoff_summary(module_name: ModuleName, values: dict[str, Any]) -> str:
    if module_name == "clinical_outcome_prediction":
        return "This Agent 3 output can be consumed by Agent 4 as clinical risk, asset, source, comparator, safety, and missing-data context."
    if module_name == "due_diligence":
        handoff = values.get("agent3_handoff") or {}
        agent3_run = handoff.get("agent3_run_id", "unknown") if isinstance(handoff, dict) else "unknown"
        return f"This Agent 4 output consumed Agent 3 run {agent3_run} and can be consumed by Agent 5 as diligence, red-flag, assumption, and source context."
    if module_name == "protocol_design":
        handoff3 = values.get("agent3_handoff") or {}
        handoff4 = values.get("agent4_handoff") or {}
        agent3_run = handoff3.get("agent3_run_id", "unknown") if isinstance(handoff3, dict) else "unknown"
        agent4_run = handoff4.get("agent4_run_id", "unknown") if isinstance(handoff4, dict) else "unknown"
        return f"This Agent 5 output consumed Agent 3 run {agent3_run} and Agent 4 run {agent4_run}; it is a draft strategy artifact requiring review."
    return "This compatibility output can be inspected as a deterministic trial-landscape artifact."


def _limitations(values: dict[str, Any]) -> tuple[str, ...]:
    reasons = []
    for flag in values.get("missing_data_flags") or ():
        if isinstance(flag, dict) and flag.get("reason"):
            reasons.append(str(flag["reason"]))
    for validation in values.get("validation_results") or ():
        if isinstance(validation, dict) and validation.get("status") in {"failed", "warning", "needs_human_review"}:
            reasons.append(str(validation.get("message") or validation.get("gate_reason") or "Validation requires review."))
    return tuple(dict.fromkeys(reasons))[:8]


def _review_questions(values: dict[str, Any]) -> tuple[str, ...]:
    questions = []
    for output in values.get("synthesis_outputs") or ():
        if isinstance(output, dict):
            questions.extend(str(question) for question in output.get("review_questions") or ())
    memo_questions = _nested(values, "asset_memo", "review_questions") or ()
    brief_questions = _nested(values, "protocol_design_brief", "human_review_questions") or ()
    reviewer_questions = tuple(_nested(values, "protocol_design_brief", "reviewer_critique", key) or () for key in ("statistical_questions", "regulatory_questions"))
    for collection in (memo_questions, brief_questions, *reviewer_questions):
        questions.extend(str(question) for question in collection)
    if not questions and values.get("human_gate"):
        questions.append("Human reviewers should resolve the workflow gate before using this output operationally.")
    return tuple(dict.fromkeys(questions))[:10]


def _findings(
    module_name: ModuleName,
    values: dict[str, Any],
    source_ids: tuple[str, ...],
    confidence: float,
) -> list[HumanReadableFinding]:
    if module_name == "clinical_outcome_prediction":
        return [
            _finding("Asset identity", _nested(values, "asset_identity", "summary") or _nested(values, "asset_identity", "asset_name") or "Asset identity unavailable.", source_ids, confidence),
            _finding("Endpoint risk", _nested(values, "endpoint_risk_assessment", "rationale") or "Endpoint risk rationale unavailable.", source_ids, confidence),
            _finding("Enrollment feasibility", _nested(values, "enrollment_duration_risk", "rationale") or "Enrollment feasibility rationale unavailable.", source_ids, confidence),
            _finding("Comparator benchmark", _nested(values, "comparator_benchmarking", "benchmark_summary") or "Comparator benchmark summary unavailable.", source_ids, confidence),
            _finding("Safety context", _nested(values, "safety_context", "summary") or "Safety context unavailable.", source_ids, confidence),
        ]
    if module_name == "due_diligence":
        return [
            _finding("Clinical evidence", _nested(values, "clinical_evidence", "ctgov_summary") or "Clinical evidence summary unavailable.", source_ids, confidence),
            _finding("Competitive landscape", _nested(values, "competitive_landscape", "benchmark_summary") or "Competitive landscape summary unavailable.", source_ids, confidence),
            _finding("Safety label", _nested(values, "safety_label_summary", "warnings_summary") or "Safety label summary unavailable.", source_ids, confidence),
            _finding("Patent and LOE", _nested(values, "patent_loe_review", "review_summary") or "Patent review summary unavailable.", source_ids, confidence),
            _finding("Asset memo", _nested(values, "asset_memo", "summary") or "Asset memo summary unavailable.", source_ids, confidence),
        ]
    if module_name == "protocol_design":
        intent_detail = _next_study_intent_summary(values)
        return [
            _finding("Next study intent", intent_detail, source_ids, confidence),
            _finding("Analog benchmark", _analog_summary(values), source_ids, confidence),
            _finding("Executive synopsis", _nested(values, "protocol_design_brief", "executive_synopsis", "body") or "Executive synopsis unavailable.", source_ids, confidence),
            _finding("Endpoint strategy", _nested(values, "protocol_design_brief", "endpoint_strategy", "body") or "Endpoint strategy unavailable.", source_ids, confidence),
            _finding("Safety monitoring", _nested(values, "protocol_design_brief", "safety_monitoring_outline", "body") or "Safety monitoring outline unavailable.", source_ids, confidence),
            _finding("Reviewer critique", "; ".join(_nested(values, "protocol_design_brief", "reviewer_critique", "limitations") or ()) or "Reviewer critique limitations unavailable.", source_ids, confidence),
        ]
    return [_finding("Landscape summary", str(values.get("landscape_summary") or "Landscape summary unavailable."), source_ids, confidence)]


def _finding(title: str, detail: str, source_ids: tuple[str, ...], confidence: float) -> HumanReadableFinding:
    return HumanReadableFinding(title=title, detail=detail[:1200], source_ids=source_ids, confidence=max(0.0, min(1.0, confidence)))


def _analog_summary(values: dict[str, Any]) -> str:
    bundle = values.get("analog_benchmark_bundle") or {}
    if not isinstance(bundle, dict):
        return "Analog benchmark unavailable."
    selected = ", ".join(bundle.get("selected_analog_ids") or ()) or "none selected"
    limitations = "; ".join(bundle.get("limitations") or ())
    return f"Selected analogs: {selected}. {limitations}".strip()


def _next_study_intent_summary(values: dict[str, Any]) -> str:
    intent = values.get("next_study_intent") or _nested(values, "protocol_design_brief", "next_study_intent") or {}
    if not isinstance(intent, dict):
        return "Next-study intent unavailable."
    parts = [
        f"Proposed next study: {intent.get('proposed_next_stage') or 'unknown'}",
        f"role: {intent.get('study_role') or 'unknown'}",
        f"objective: {intent.get('development_objective') or 'unknown'}",
        f"key question: {intent.get('key_clinical_question') or 'unknown'}",
    ]
    alternatives = "; ".join(intent.get("alternatives_considered") or ())
    if alternatives:
        parts.append(f"alternatives considered: {alternatives}")
    return ". ".join(parts) + "."


def _nct_id(values: dict[str, Any]) -> str:
    return str(_nested(values, "input", "nct_id") or _nested(values, "trial_identity", "nct_id") or _nested(values, "target_trial", "nct_id") or "unknown NCT")


def _nested(values: dict[str, Any], *path: str) -> Any:
    current: Any = values
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _module_agent_prefix(module_name: ModuleName) -> str:
    return {
        "clinical_outcome_prediction": "Agent3",
        "due_diligence": "Agent4",
        "protocol_design": "Agent5",
        "trial_intelligence": "TrialIntelligence",
    }[module_name]
