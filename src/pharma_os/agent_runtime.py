"""Reusable OpenAI Agents SDK runtime foundation for PharmaOS.

Agent 3, Agent 4, Agent 5, the compatibility trial-landscape route, and the
Control Tower all use this shared layer to run structured SDK calls when live
agents are enabled and validated deterministic fallbacks when offline.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from pharma_os.schemas import AgentRunTrace, AgentStepTrace, AgentToolCallTrace, ExecutionMode, StrictSchema


T = TypeVar("T", bound=BaseModel)


class AgentRuntimeError(RuntimeError):
    """Raised when an agent run cannot complete."""


class AgentRuntimeConfig(StrictSchema):
    """Runtime settings for OpenAI Agents SDK-backed workflow components."""

    model: str = Field(default="gpt-5.6-terra", min_length=1)
    model_route: str = Field(default="default", min_length=1)
    max_turns: int = Field(default=8, ge=1, le=50)
    disabled: bool = False
    provenance: str = "pharma_os.agent_runtime.env"


class StructuredAgentResult(StrictSchema):
    """Structured output and safe trace from one agent run."""

    output: BaseModel
    trace: AgentRunTrace
    trace_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


MODEL_TIER_DEFAULTS = {
    "fast": "gpt-5.6-luna",
    "balanced": "gpt-5.6-terra",
    "deep": "gpt-5.6-sol",
}

MODEL_ROUTE_ENV = {
    "request_understanding": "PHARMA_OS_MODEL_REQUEST_UNDERSTANDING",
    "control_tower": "PHARMA_OS_MODEL_CONTROL_TOWER",
    "human_summary": "PHARMA_OS_MODEL_HUMAN_SUMMARY",
    "agent3_manager": "PHARMA_OS_MODEL_AGENT3_MANAGER",
    "agent3_subagent": "PHARMA_OS_MODEL_AGENT3_SUBAGENT",
    "agent4_manager": "PHARMA_OS_MODEL_AGENT4_MANAGER",
    "agent4_subagent": "PHARMA_OS_MODEL_AGENT4_SUBAGENT",
    "agent5_manager": "PHARMA_OS_MODEL_AGENT5_MANAGER",
    "agent5_subagent": "PHARMA_OS_MODEL_AGENT5_SUBAGENT",
}

MODEL_ROUTE_TIERS = {
    "default": "balanced",
    "request_understanding": "fast",
    "control_tower": "balanced",
    "human_summary": "fast",
    "agent3_manager": "balanced",
    "agent3_subagent": "fast",
    "agent4_manager": "balanced",
    "agent4_subagent": "balanced",
    "agent5_manager": "deep",
    "agent5_subagent": "balanced",
}


def runtime_config_from_env(*, model_route: str = "default") -> AgentRuntimeConfig:
    """Read runtime configuration from environment variables."""

    disabled = _truthy(os.getenv("PHARMA_OS_AGENTS_DISABLED")) or _truthy(os.getenv("PHARMA_OS_OFFLINE"))
    max_turns_text = os.getenv("PHARMA_OS_AGENT_MAX_TURNS")
    try:
        max_turns = int(max_turns_text) if max_turns_text else 8
    except ValueError:
        max_turns = 8
    route = _normalize_model_route(model_route)
    return AgentRuntimeConfig(
        model=resolve_model_for_route(route),
        model_route=route,
        max_turns=max_turns,
        disabled=disabled,
    )


def runtime_config_for_live_agents(*, disabled_provenance: str, model_route: str = "default") -> AgentRuntimeConfig:
    """Resolve live/offline agent mode from environment.

    Live OpenAI Agents SDK calls are enabled when an API key is present, unless
    the operator explicitly disables agents or sets PHARMA_OS_ENABLE_LIVE_AGENTS=false.
    """

    env_config = runtime_config_from_env(model_route=model_route)
    if env_config.disabled:
        return env_config
    live_setting = os.getenv("PHARMA_OS_ENABLE_LIVE_AGENTS")
    if live_setting is not None and live_setting.strip() and not _truthy(live_setting):
        return env_config.model_copy(update={"disabled": True, "provenance": f"{disabled_provenance}.disabled_by_env"})
    if os.getenv("OPENAI_API_KEY"):
        return env_config
    return env_config.model_copy(update={"disabled": True, "provenance": f"{disabled_provenance}.missing_openai_api_key"})


def runtime_config_for_route(
    *,
    model_route: str,
    disabled_provenance: str,
    config: AgentRuntimeConfig | None = None,
) -> AgentRuntimeConfig:
    """Return route-specific runtime config unless the caller supplied an explicit config."""

    if config is not None:
        if config.model_route == "default":
            return config.model_copy(update={"model_route": _normalize_model_route(model_route)})
        return config
    return runtime_config_for_live_agents(disabled_provenance=disabled_provenance, model_route=model_route)


def resolve_model_for_route(model_route: str = "default") -> str:
    """Resolve the selected model for a logical agent/workflow route."""

    route = _normalize_model_route(model_route)
    route_env = MODEL_ROUTE_ENV.get(route)
    if route_env and os.getenv(route_env):
        return str(os.getenv(route_env))
    if os.getenv("PHARMA_OS_MODEL"):
        return str(os.getenv("PHARMA_OS_MODEL"))
    tier = MODEL_ROUTE_TIERS.get(route, MODEL_ROUTE_TIERS["default"])
    return MODEL_TIER_DEFAULTS[tier]


def load_agents_sdk() -> tuple[Any, Any, Any, Any]:
    """Import OpenAI Agents SDK lazily so tests can run without live calls."""

    try:
        from agents import Agent, AgentOutputSchema, Runner, function_tool
    except ModuleNotFoundError as exc:
        raise AgentRuntimeError(
            "OpenAI Agents SDK is not installed. Install project dependencies before live agent runs."
        ) from exc
    return Agent, AgentOutputSchema, Runner, function_tool


def agents_sdk_output_schema(output_type: type[BaseModel]) -> Any:
    """Build an Agents SDK output schema compatible with PharmaOS Pydantic models.

    PharmaOS schemas intentionally include typed metadata dictionaries for
    provenance, assumptions, and trace details. The Agents SDK strict schema
    preflight rejects those dynamic object fields, so live SDK agents use the
    non-strict SDK wrapper and PharmaOS validates the final output with the
    original strict Pydantic model after the call.
    """

    _, AgentOutputSchema, _, _ = load_agents_sdk()
    return AgentOutputSchema(output_type, strict_json_schema=False)


def run_structured_agent(
    *,
    agent: Any,
    payload: BaseModel | dict[str, Any],
    output_type: type[T],
    agent_name: str,
    run_id: str,
    input_summary: str,
    config: AgentRuntimeConfig | None = None,
    max_turns: int | None = None,
    offline_output: T | dict[str, Any] | None = None,
    source_ids: tuple[str, ...] = (),
    confidence: float | None = None,
    rationale_summary: str | None = None,
) -> StructuredAgentResult:
    """Run an SDK agent with structured output and safe trace metadata."""

    settings = config or runtime_config_from_env()
    effective_max_turns = max_turns or settings.max_turns
    started_at = datetime.now(timezone.utc)
    if settings.disabled:
        if offline_output is None:
            raise AgentRuntimeError("Agent runtime is disabled/offline and no offline_output was supplied.")
        output = _validate_output(offline_output, output_type)
        completed_at = datetime.now(timezone.utc)
        return StructuredAgentResult(
            output=output,
            trace=_trace(
                run_id=run_id,
                agent_name=agent_name,
                input_summary=input_summary,
                output=output,
                started_at=started_at,
                completed_at=completed_at,
                source_ids=source_ids,
                confidence=confidence,
                rationale_summary=rationale_summary or "Offline structured output was validated without a live agent call.",
                tool_calls=(),
                provenance="pharma_os.agent_runtime.offline",
                execution_mode="deterministic_fallback",
                runtime_metadata=_trace_runtime_metadata(settings, {"retry_count": 0, "retry_attempts": 1, "retry_exhausted": False, "fallback_cause": "disabled"}),
            ),
            trace_metadata={
                "agent_name": agent_name,
                "model": settings.model,
                "model_route": settings.model_route,
                "max_turns": effective_max_turns,
                "disabled": True,
                "execution_mode": "deterministic_fallback",
                "retry_count": 0,
                "fallback_cause": "disabled",
            },
        )

    _, _, Runner, _ = load_agents_sdk()
    try:
        response, retry_metadata = _execute_with_retries(
            lambda: _run_agent_once(Runner, agent, payload, effective_max_turns),
            operation="Agents SDK run",
        )
    except Exception as exc:
        if _agent_fallbacks_disabled():
            raise AgentRuntimeError(f"Agent run failed with fallbacks disabled: {exc.__class__.__name__}: {exc}") from exc
        if offline_output is None:
            raise AgentRuntimeError(f"Agent run failed: {exc.__class__.__name__}: {exc}") from exc
        output = _validate_output(offline_output, output_type)
        completed_at = datetime.now(timezone.utc)
        retry_metadata = _retry_metadata_for_exception(exc)
        return StructuredAgentResult(
            output=output,
            trace=_trace(
                run_id=run_id,
                agent_name=agent_name,
                input_summary=input_summary,
                output=output,
                started_at=started_at,
                completed_at=completed_at,
                source_ids=source_ids,
                confidence=confidence,
                rationale_summary=rationale_summary
                or "Agents SDK call failed; offline structured output was validated as fallback.",
                tool_calls=(),
                provenance="pharma_os.agent_runtime.openai_agents_sdk_fallback",
                execution_mode="deterministic_fallback",
                runtime_metadata=_trace_runtime_metadata(settings, retry_metadata, fallback_cause=_fallback_cause(exc)),
            ),
            trace_metadata={
                "agent_name": agent_name,
                "model": settings.model,
                "model_route": settings.model_route,
                "max_turns": effective_max_turns,
                "disabled": False,
                "fallback": True,
                "execution_mode": "deterministic_fallback",
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:500],
                "fallback_cause": _fallback_cause(exc),
                **retry_metadata,
            },
        )

    parsed = getattr(response, "final_output", response)
    output = _validate_output(parsed, output_type)
    completed_at = datetime.now(timezone.utc)
    tool_calls = _extract_tool_calls(
        response=response,
        run_id=run_id,
        agent_name=agent_name,
        started_at=started_at,
        completed_at=completed_at,
    )
    return StructuredAgentResult(
        output=output,
        trace=_trace(
            run_id=run_id,
            agent_name=agent_name,
            input_summary=input_summary,
            output=output,
            started_at=started_at,
            completed_at=completed_at,
            source_ids=source_ids,
            confidence=confidence,
            rationale_summary=rationale_summary or "Structured agent output was validated; hidden reasoning was not stored.",
            tool_calls=tool_calls,
            provenance="pharma_os.agent_runtime.openai_agents_sdk",
            execution_mode="live_agent",
            runtime_metadata=_trace_runtime_metadata(settings, retry_metadata),
        ),
        trace_metadata={
            **_response_metadata(response),
            "agent_name": agent_name,
            "model": settings.model,
            "model_route": settings.model_route,
            "max_turns": effective_max_turns,
            "disabled": False,
            "execution_mode": "live_agent",
            **retry_metadata,
        },
    )


def run_structured_llm_call(
    *,
    agent_name: str,
    instructions: str,
    payload: BaseModel | dict[str, Any],
    output_type: type[T],
    run_id: str,
    input_summary: str,
    config: AgentRuntimeConfig | None = None,
    offline_output: T | dict[str, Any] | None = None,
    source_ids: tuple[str, ...] = (),
    confidence: float | None = None,
    rationale_summary: str | None = None,
) -> StructuredAgentResult:
    """Run one direct OpenAI structured-output call with PharmaOS tracing."""

    settings = _direct_llm_runtime_config(config)
    started_at = datetime.now(timezone.utc)
    if settings.disabled:
        return _offline_structured_result(
            offline_output=offline_output,
            output_type=output_type,
            run_id=run_id,
            agent_name=agent_name,
            input_summary=input_summary,
            started_at=started_at,
            settings=settings,
            source_ids=source_ids,
            confidence=confidence,
            rationale_summary=rationale_summary or "Offline structured output was validated without a live direct OpenAI call.",
            trace_metadata={"direct_api": True},
        )

    try:
        (parsed, response_metadata), retry_metadata = _execute_with_retries(
            lambda: _call_openai_structured_output(
                model=settings.model,
                instructions=instructions,
                payload=payload,
                output_type=output_type,
            ),
            operation="Direct OpenAI structured output call",
        )
    except Exception as exc:
        if _agent_fallbacks_disabled():
            if isinstance(exc, AgentRuntimeError):
                raise
            raise AgentRuntimeError(f"Direct OpenAI structured output call failed with fallbacks disabled: {exc.__class__.__name__}: {exc}") from exc
        if offline_output is None:
            if isinstance(exc, AgentRuntimeError):
                raise
            raise AgentRuntimeError(f"Direct OpenAI structured output call failed: {exc.__class__.__name__}: {exc}") from exc
        output = _validate_output(offline_output, output_type)
        completed_at = datetime.now(timezone.utc)
        retry_metadata = _retry_metadata_for_exception(exc)
        return StructuredAgentResult(
            output=output,
            trace=_trace(
                run_id=run_id,
                agent_name=agent_name,
                input_summary=input_summary,
                output=output,
                started_at=started_at,
                completed_at=completed_at,
                source_ids=source_ids,
                confidence=confidence,
                rationale_summary=rationale_summary
                or "Direct OpenAI call failed; offline structured output was validated as fallback.",
                tool_calls=(),
                provenance="pharma_os.agent_runtime.direct_openai_api_fallback",
                execution_mode="deterministic_fallback",
                runtime_metadata=_trace_runtime_metadata(settings, retry_metadata, fallback_cause=_fallback_cause(exc)),
            ),
            trace_metadata={
                "agent_name": agent_name,
                "model": settings.model,
                "model_route": settings.model_route,
                "disabled": False,
                "direct_api": True,
                "fallback": True,
                "execution_mode": "deterministic_fallback",
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:500],
                "fallback_cause": _fallback_cause(exc),
                **retry_metadata,
            },
        )

    output = _validate_output(parsed, output_type)
    completed_at = datetime.now(timezone.utc)
    return StructuredAgentResult(
        output=output,
        trace=_trace(
            run_id=run_id,
            agent_name=agent_name,
            input_summary=input_summary,
            output=output,
            started_at=started_at,
            completed_at=completed_at,
            source_ids=source_ids,
            confidence=confidence,
            rationale_summary=rationale_summary or "Structured direct OpenAI output was validated; hidden reasoning was not stored.",
            tool_calls=(),
            provenance="pharma_os.agent_runtime.openai_api_structured_output",
            execution_mode="direct_llm",
            runtime_metadata=_trace_runtime_metadata(settings, retry_metadata),
        ),
        trace_metadata={
            **response_metadata,
            "agent_name": agent_name,
            "model": settings.model,
            "model_route": settings.model_route,
            "disabled": False,
            "direct_api": True,
            "execution_mode": "direct_llm",
            **retry_metadata,
        },
    )


def _trace(
    *,
    run_id: str,
    agent_name: str,
    input_summary: str,
    output: BaseModel,
    started_at: datetime,
    completed_at: datetime,
    source_ids: tuple[str, ...],
    confidence: float | None,
    rationale_summary: str,
    tool_calls: tuple[AgentToolCallTrace, ...],
    provenance: str,
    execution_mode: ExecutionMode,
    runtime_metadata: dict[str, str | int | float | bool | None] | None = None,
) -> AgentRunTrace:
    output_id = getattr(output, "output_id", None) or getattr(output, "brief_id", None)
    output_summary = _summarize_model(output)
    step = AgentStepTrace(
        run_id=run_id,
        agent_name=agent_name,
        step_id=f"step-{uuid4()}",
        input_summary=input_summary,
        output_summary=output_summary,
        tool_calls=tool_calls,
        source_ids=source_ids,
        confidence=confidence,
        started_at=started_at,
        completed_at=completed_at,
        provenance=provenance,
        execution_mode=execution_mode,
    )
    return AgentRunTrace(
        trace_id=f"trace-{uuid4()}",
        run_id=run_id,
        agent_name=agent_name,
        input_summary=input_summary,
        output_id=str(output_id) if output_id else None,
        output_type=output.__class__.__name__,
        output_summary=output_summary,
        steps=(step,),
        tool_calls=tool_calls,
        source_ids=source_ids,
        confidence=confidence,
        rationale_summary=rationale_summary[:1000],
        started_at=started_at,
        completed_at=completed_at,
        provenance=provenance,
        execution_mode=execution_mode,
        **(runtime_metadata or {}),
    )


def _offline_structured_result(
    *,
    offline_output: T | dict[str, Any] | None,
    output_type: type[T],
    run_id: str,
    agent_name: str,
    input_summary: str,
    started_at: datetime,
    settings: AgentRuntimeConfig,
    source_ids: tuple[str, ...],
    confidence: float | None,
    rationale_summary: str,
    trace_metadata: dict[str, str | int | float | bool | None] | None = None,
) -> StructuredAgentResult:
    if offline_output is None:
        raise AgentRuntimeError("Agent runtime is disabled/offline and no offline_output was supplied.")
    output = _validate_output(offline_output, output_type)
    completed_at = datetime.now(timezone.utc)
    return StructuredAgentResult(
        output=output,
        trace=_trace(
            run_id=run_id,
            agent_name=agent_name,
            input_summary=input_summary,
            output=output,
            started_at=started_at,
            completed_at=completed_at,
            source_ids=source_ids,
            confidence=confidence,
            rationale_summary=rationale_summary,
            tool_calls=(),
            provenance="pharma_os.agent_runtime.offline",
            execution_mode="deterministic_fallback",
            runtime_metadata=_trace_runtime_metadata(settings, {"retry_count": 0, "retry_attempts": 1, "retry_exhausted": False, "fallback_cause": "disabled"}),
        ),
        trace_metadata={
            "agent_name": agent_name,
            "model": settings.model,
            "model_route": settings.model_route,
            "max_turns": settings.max_turns,
            "disabled": True,
            "execution_mode": "deterministic_fallback",
            "retry_count": 0,
            "fallback_cause": "disabled",
            **(trace_metadata or {}),
        },
    )


def _run_agent_once(Runner: Any, agent: Any, payload: BaseModel | dict[str, Any], max_turns: int) -> Any:
    try:
        return Runner.run_sync(
            agent,
            _payload_json(payload),
            max_turns=max_turns,
        )
    except TypeError:
        return Runner.run_sync(agent, _payload_json(payload))


def _execute_with_retries(operation_call: Any, *, operation: str) -> tuple[Any, dict[str, str | int | float | bool | None]]:
    policy = _retry_policy_from_env()
    attempts = 0
    last_exc: Exception | None = None
    while attempts < policy["max_attempts"]:
        attempts += 1
        try:
            result = operation_call()
            return result, {
                "retry_count": attempts - 1,
                "retry_attempts": attempts,
                "retry_exhausted": False,
            }
        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc) or attempts >= policy["max_attempts"]:
                break
            delay = _retry_delay_seconds(exc, attempt=attempts, policy=policy)
            time.sleep(delay)
    assert last_exc is not None
    if _is_transient_error(last_exc) and attempts >= policy["max_attempts"]:
        raise AgentRuntimeError(
            f"{operation} failed after {attempts} attempts due to transient error: "
            f"{last_exc.__class__.__name__}: {last_exc}"
        ) from last_exc
    raise last_exc


def _retry_policy_from_env() -> dict[str, float | int]:
    return {
        "max_attempts": _env_int("PHARMA_OS_LLM_MAX_RETRIES", 4, minimum=1, maximum=12),
        "initial_delay": _env_float("PHARMA_OS_LLM_RETRY_INITIAL_DELAY_SECONDS", 1.0, minimum=0.0, maximum=120.0),
        "max_delay": _env_float("PHARMA_OS_LLM_RETRY_MAX_DELAY_SECONDS", 30.0, minimum=0.0, maximum=300.0),
    }


def _retry_delay_seconds(exc: Exception, *, attempt: int, policy: dict[str, float | int]) -> float:
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(float(policy["max_delay"]), max(0.0, retry_after))
    base = float(policy["initial_delay"]) * (2 ** max(0, attempt - 1))
    jitter = random.uniform(0.0, min(1.0, base * 0.25)) if base > 0 else 0.0
    return min(float(policy["max_delay"]), base + jitter)


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    response = getattr(exc, "response", None)
    if headers is None and response is not None:
        headers = getattr(response, "headers", None)
    if not headers:
        return None
    retry_after_ms = _header_value(headers, "retry-after-ms")
    if retry_after_ms is not None:
        try:
            return float(retry_after_ms) / 1000.0
        except ValueError:
            return None
    retry_after = _header_value(headers, "retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            return None
    return None


def _header_value(headers: Any, key: str) -> str | None:
    if hasattr(headers, "get"):
        value = headers.get(key) or headers.get(key.title())
        return str(value) if value is not None else None
    return None


def _is_transient_error(exc: Exception) -> bool:
    status_code = _status_code(exc)
    if status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    name = exc.__class__.__name__.casefold()
    text = str(exc).casefold()
    transient_markers = ("ratelimit", "rate_limit", "timeout", "connection", "temporarily", "service unavailable")
    return any(marker in name or marker in text for marker in transient_markers)


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if status_code is None and getattr(exc, "response", None) is not None:
        status_code = getattr(exc.response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


def _retry_metadata_for_exception(exc: Exception) -> dict[str, str | int | float | bool | None]:
    retry_count = 0
    retry_attempts = 1
    retry_exhausted = False
    cause_exc: Exception = exc
    if isinstance(exc, AgentRuntimeError) and exc.__cause__ is not None and isinstance(exc.__cause__, Exception):
        cause_exc = exc.__cause__
        if "failed after" in str(exc):
            retry_exhausted = True
            retry_attempts = _retry_policy_from_env()["max_attempts"]  # type: ignore[assignment]
            retry_count = int(retry_attempts) - 1
    metadata: dict[str, str | int | float | bool | None] = {
        "retry_count": retry_count,
        "retry_attempts": retry_attempts,
        "retry_exhausted": retry_exhausted,
    }
    if retry_exhausted:
        metadata["final_retry_reason"] = f"{cause_exc.__class__.__name__}: {str(cause_exc)[:500]}"
    return metadata


def _trace_runtime_metadata(
    settings: AgentRuntimeConfig,
    retry_metadata: dict[str, str | int | float | bool | None],
    *,
    fallback_cause: str | None = None,
) -> dict[str, str | int | float | bool | None]:
    values: dict[str, str | int | float | bool | None] = {
        "model": settings.model,
        "model_route": settings.model_route,
        "retry_count": int(retry_metadata.get("retry_count") or 0),
        "retry_attempts": int(retry_metadata.get("retry_attempts") or 1),
        "retry_exhausted": bool(retry_metadata.get("retry_exhausted") or False),
        "fallback_cause": fallback_cause or retry_metadata.get("fallback_cause"),
        "final_retry_reason": retry_metadata.get("final_retry_reason"),
    }
    return {key: value for key, value in values.items() if value is not None}


def _fallback_cause(exc: Exception) -> str:
    cause = exc.__cause__ if isinstance(exc, AgentRuntimeError) and isinstance(exc.__cause__, Exception) else exc
    if _status_code(cause) == 429 or "ratelimit" in cause.__class__.__name__.casefold() or "rate limit" in str(cause).casefold():
        return "rate_limit"
    if "timeout" in cause.__class__.__name__.casefold() or "timeout" in str(cause).casefold():
        return "timeout"
    if _is_transient_error(cause):
        return "transient_error"
    if isinstance(exc, AgentRuntimeError):
        return "runtime_error"
    return "sdk_error"


def _extract_tool_calls(
    *,
    response: Any,
    run_id: str,
    agent_name: str,
    started_at: datetime,
    completed_at: datetime,
) -> tuple[AgentToolCallTrace, ...]:
    calls = []
    raw_calls = getattr(response, "tool_calls", None) or getattr(response, "tools", None) or ()
    if isinstance(raw_calls, dict):
        raw_calls = raw_calls.values()
    for index, call in enumerate(raw_calls, start=1):
        name = getattr(call, "name", None) or getattr(call, "tool_name", None)
        if not name and isinstance(call, dict):
            name = call.get("name") or call.get("tool_name")
        calls.append(
            AgentToolCallTrace(
                run_id=run_id,
                agent_name=agent_name,
                step_id=f"step-tool-{index}",
                tool_name=str(name or f"tool_{index}"),
                input_summary=_safe_summary(getattr(call, "input", None) if not isinstance(call, dict) else call.get("input")),
                output_summary=_safe_summary(getattr(call, "output", None) if not isinstance(call, dict) else call.get("output")),
                source_ids=(),
                started_at=started_at,
                completed_at=completed_at,
                provenance="pharma_os.agent_runtime.tool_call_summary",
                execution_mode="live_agent",
            )
        )
    return tuple(calls)


def _response_metadata(response: Any) -> dict[str, str | int | float | bool | None]:
    metadata: dict[str, str | int | float | bool | None] = {}
    for name in ("last_response_id", "trace_id", "workflow_name"):
        value = getattr(response, name, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[name] = value
    usage = getattr(response, "usage", None)
    if usage is not None:
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(usage, name, None)
            if isinstance(value, (int, float)):
                metadata[f"usage_{name}"] = value
    return {key: value for key, value in metadata.items() if value is not None}


def _direct_llm_runtime_config(config: AgentRuntimeConfig | None) -> AgentRuntimeConfig:
    env_config = runtime_config_for_live_agents(disabled_provenance="pharma_os.agent_runtime.direct_openai_api")
    if config is None:
        return env_config
    if config.disabled:
        return config
    if env_config.disabled:
        return config.model_copy(update={"disabled": True, "provenance": env_config.provenance})
    return config


def _call_openai_structured_output(
    *,
    model: str,
    instructions: str,
    payload: BaseModel | dict[str, Any],
    output_type: type[T],
) -> tuple[Any, dict[str, str | int | float | bool | None]]:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise AgentRuntimeError("OpenAI Python SDK is not installed. Install project dependencies before live LLM calls.") from exc

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    payload_text = _payload_json(payload)

    if hasattr(client, "responses") and hasattr(client.responses, "parse"):
        response = client.responses.parse(
            model=model,
            instructions=instructions,
            input=payload_text,
            text_format=output_type,
        )
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise AgentRuntimeError("Direct OpenAI structured output response did not include parsed output.")
        return parsed, _response_metadata(response)

    beta = getattr(client, "beta", None)
    chat = getattr(beta, "chat", None)
    completions = getattr(chat, "completions", None)
    if completions is not None and hasattr(completions, "parse"):
        response = completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": payload_text},
            ],
            response_format=output_type,
        )
        parsed = getattr(response.choices[0].message, "parsed", None)
        if parsed is None:
            raise AgentRuntimeError("Direct OpenAI chat structured output response did not include parsed output.")
        return parsed, _response_metadata(response)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": payload_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": output_type.__name__,
                "schema": output_type.model_json_schema(),
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise AgentRuntimeError("Direct OpenAI structured output response was empty.")
    return json.loads(content), _response_metadata(response)


def _validate_output(value: T | dict[str, Any] | Any, output_type: type[T]) -> T:
    if isinstance(value, output_type):
        return value
    return output_type.model_validate(value)


def _payload_json(payload: BaseModel | dict[str, Any]) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    return json.dumps(payload, ensure_ascii=False)


def _summarize_model(model: BaseModel) -> str:
    values = model.model_dump(mode="json")
    for key in ("summary", "landscape_summary", "title", "output_id", "brief_id"):
        value = values.get(key)
        if isinstance(value, str) and value:
            return value[:1000]
    return f"{model.__class__.__name__} structured output validated."


def _safe_summary(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:1000]
    try:
        return json.dumps(value, ensure_ascii=False, default=str)[:1000]
    except TypeError:
        return str(value)[:1000]


def _normalize_model_route(model_route: str | None) -> str:
    route = str(model_route or "default").strip().casefold()
    return route if route in MODEL_ROUTE_TIERS else "default"


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name) or default)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name) or default)
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def agent_fallbacks_disabled() -> bool:
    """Return whether live agent fallback paths should fail closed."""

    return _truthy(os.getenv("PHARMA_OS_DISABLE_AGENT_FALLBACKS"))


def _agent_fallbacks_disabled() -> bool:
    return agent_fallbacks_disabled()
