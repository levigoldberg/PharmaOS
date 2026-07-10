"""AI-first request understanding for Control Tower orchestration."""

from __future__ import annotations

import os
from typing import Any

from pharma_os.agent_runtime import AgentRuntimeConfig, AgentRuntimeError, run_structured_llm_call, runtime_config_for_live_agents
from pharma_os.registry import WorkflowRegistry
from pharma_os.schemas import RequestUnderstandingOutput


class RequestUnderstandingError(RuntimeError):
    """Raised when a natural-language orchestration goal cannot be parsed."""


def understand_orchestration_goal(
    *,
    goal: str,
    explicit_fields: dict[str, Any],
    registry: WorkflowRegistry | None = None,
    config: AgentRuntimeConfig | None = None,
) -> RequestUnderstandingOutput:
    """Parse a natural-language orchestration goal into structured request fields."""

    effective_registry = registry or WorkflowRegistry.default()
    runtime_config = config or runtime_config_for_live_agents(
        disabled_provenance="pharma_os.request_understanding",
        model_route="request_understanding",
    )
    payload = {
        "raw_goal": goal,
        "explicit_cli_fields": explicit_fields,
        "implemented_capabilities": [
            capability.model_dump(mode="json")
            for capability in effective_registry.capabilities()
            if capability.implementation_status == "implemented" and capability.executable
        ],
        "registered_capabilities": [capability.model_dump(mode="json") for capability in effective_registry.capabilities()],
        "accepted_orchestration_fields": (
            "nct_id",
            "asset_name",
            "indication",
            "assumptions",
            "force_refresh",
            "decision_type",
            "target_capability",
            "requested_outputs",
            "execution_scope",
        ),
        "constraints": (
            "Return only structured RequestUnderstandingOutput. Extract NCT IDs exactly when present. "
            "Only put reviewed execution assumptions in assumptions: pos_workbook_path, wac_data_path, "
            "annual_patients, peak_penetration, gross_to_net, operating_margin, discount_rate, "
            "development_cost, launch_year, or loe_year. Do not put workflow intent, rationale, or inferred "
            "task labels in assumptions. "
            "If the goal maps to a registered but non-executable capability, set target_capability to that capability "
            "and do not invent missing implementation details. For executable clinical workflows, nct_id is the only "
            "hard required goal-only field. Commercial assumptions are optional reviewed inputs for due_diligence and "
            "protocol_design; extract them when present, but do not mark reviewed_commercial_assumptions or individual "
            "commercial assumptions as missing_required_fields when an NCT ID is present. Do not ask the user to choose "
            "output formats, full-vs-limited deliverables, reuse behavior, or commercial defaults for an implemented "
            "workflow; map to the registered capability and let Control Tower plan run/reuse/refresh from memory. Do not "
            "put run/reuse/refresh intent into assumptions. Use force_refresh only when the user explicitly requests a "
            "fresh rerun or refresh; otherwise leave it empty and let Control Tower decide from Scientific Memory. If an "
            "explicit fresh rerun/refresh is scoped with words like only, just, solely, or specifically, set execution_scope "
            "to target_only for the requested target capability. If the user asks for the full dependency chain or all related "
            "workflows, set execution_scope to full_path. If dependencies may be reused but not refreshed unless required, set "
            "execution_scope to target_with_dependencies. "
            "executable clinical workflow lacks hard required inputs, populate missing_required_fields and clarifying_questions. "
            "If the user explicitly asks to reuse existing memory/artifacts, reflect that in the rationale and leave force_refresh empty."
        ),
    }
    try:
        result = run_structured_llm_call(
            agent_name="RequestUnderstandingAgent",
            instructions=_instructions(),
            payload=payload,
            output_type=RequestUnderstandingOutput,
            run_id="request-understanding",
            input_summary=f"Parse orchestration goal: {goal[:160]}",
            config=runtime_config,
            offline_output=None,
            source_ids=(),
            confidence=None,
            rationale_summary="Parse natural-language orchestration request into typed Control Tower fields.",
        )
    except AgentRuntimeError as exc:
        raise RequestUnderstandingError(_request_understanding_error_message(exc, runtime_config)) from exc
    return result.output


def _request_understanding_error_message(exc: AgentRuntimeError, runtime_config: AgentRuntimeConfig) -> str:
    route = runtime_config.model_route
    model = runtime_config.model
    if runtime_config.disabled:
        if not os.getenv("OPENAI_API_KEY"):
            return (
                "Natural-language goal parsing requires live AI, but OPENAI_API_KEY is not visible to this Python process. "
                "Run from the PharmaOS project directory with .env present, export OPENAI_API_KEY, or pass --input-json "
                "with a complete OrchestrationRequest."
            )
        if _truthy_env("PHARMA_OS_AGENTS_DISABLED") or _truthy_env("PHARMA_OS_OFFLINE"):
            return (
                "Natural-language goal parsing requires live AI, but live agents are disabled "
                f"({runtime_config.provenance}). Set PHARMA_OS_AGENTS_DISABLED=false and PHARMA_OS_OFFLINE=false/unset, "
                "or pass --input-json with a complete OrchestrationRequest."
            )
        live_setting = os.getenv("PHARMA_OS_ENABLE_LIVE_AGENTS")
        if live_setting is not None and live_setting.strip() and not _truthy_env("PHARMA_OS_ENABLE_LIVE_AGENTS"):
            return (
                "Natural-language goal parsing requires live AI, but PHARMA_OS_ENABLE_LIVE_AGENTS is set to false. "
                "Set PHARMA_OS_ENABLE_LIVE_AGENTS=true or pass --input-json with a complete OrchestrationRequest."
            )
        return (
            "Natural-language goal parsing requires live AI, but the request-understanding route is disabled "
            f"({runtime_config.provenance}). Pass --input-json with a complete OrchestrationRequest."
        )
    return (
        "Natural-language goal parsing attempted a live OpenAI call but it failed. "
        f"Route={route}; model={model}; error={exc.__class__.__name__}: {str(exc)[:700]}. "
        "Check model access/availability for this account, API key validity, billing/project permissions, and rate limits. "
        "You can temporarily bypass AI parsing with --input-json containing a complete OrchestrationRequest."
    )


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip().casefold() in {"1", "true", "yes", "on"}


def _instructions() -> str:
    return (
        "You are RequestUnderstandingAgent for PharmaOS. Parse the user's orchestration goal into the strict output schema. "
        "Identify the intended registered capability, decision type, NCT ID, asset name, indication, reviewed assumptions, "
        "refresh or skip hints, and requested output formats. Do not execute workflows. Do not fabricate identifiers or "
        "assumptions. Reviewed assumptions are only user-provided execution parameters such as commercial assumptions, "
        "workbook paths, launch year, or LOE year; they are not summaries of the user's intent. Prefer target_capability "
        "values from the registry. For currently executable clinical workflows, an NCT ID is required. For unsupported "
        "registered capabilities, return the capability and let the Control Tower block. If the user asks to reuse existing "
        "memory/artifacts, do not convert that into a fresh-run request. If the user explicitly asks to remake, rerun, refresh, "
        "regenerate, or redo a capability, put that capability in force_refresh. If the user scopes that request to only one "
        "capability, set execution_scope to target_only; otherwise distinguish target_with_dependencies, full_path, or unspecified."
    )
