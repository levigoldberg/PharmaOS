"""AI-first request understanding for Control Tower orchestration."""

from __future__ import annotations

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
    runtime_config = config or runtime_config_for_live_agents(disabled_provenance="pharma_os.request_understanding")
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
        ),
        "constraints": (
            "Return only structured RequestUnderstandingOutput. Extract NCT IDs exactly when present. "
            "If the goal maps to a registered but non-executable capability, set target_capability to that capability "
            "and do not invent missing implementation details. If an executable clinical workflow lacks required inputs, "
            "populate missing_required_fields and clarifying_questions."
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
        raise RequestUnderstandingError(
            "Natural-language goal parsing requires live AI. Provide OPENAI_API_KEY with live agents enabled, "
            "or pass explicit structured flags such as --nct-id, or use --input-json."
        ) from exc
    return result.output


def _instructions() -> str:
    return (
        "You are RequestUnderstandingAgent for PharmaOS. Parse the user's orchestration goal into the strict output schema. "
        "Identify the intended registered capability, decision type, NCT ID, asset name, indication, reviewed assumptions, "
        "refresh or skip hints, and requested output formats. Do not execute workflows. Do not fabricate identifiers or "
        "assumptions. Prefer target_capability values from the registry. For currently executable clinical workflows, an "
        "NCT ID is required. For unsupported registered capabilities, return the capability and let the Control Tower block."
    )
