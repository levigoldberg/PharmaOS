"""Reusable OpenAI Agents SDK runtime foundation for PharmaOS.

This module is intentionally not wired into Agent 3, Agent 4, Agent 5, or
trial_intelligence yet. It provides a safe, testable layer for future agent
adoption while preserving deterministic workflow control.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from pharma_os.schemas import AgentRunTrace, AgentStepTrace, AgentToolCallTrace, StrictSchema


T = TypeVar("T", bound=BaseModel)


class AgentRuntimeError(RuntimeError):
    """Raised when an agent run cannot complete."""


class AgentRuntimeConfig(StrictSchema):
    """Runtime settings for future OpenAI Agents SDK usage."""

    model: str = Field(default="gpt-5.5", min_length=1)
    max_turns: int = Field(default=8, ge=1, le=50)
    disabled: bool = False
    provenance: str = "pharma_os.agent_runtime.env"


class StructuredAgentResult(StrictSchema):
    """Structured output and safe trace from one agent run."""

    output: BaseModel
    trace: AgentRunTrace
    trace_metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


def runtime_config_from_env() -> AgentRuntimeConfig:
    """Read runtime configuration from environment variables."""

    disabled = _truthy(os.getenv("PHARMA_OS_AGENTS_DISABLED")) or _truthy(os.getenv("PHARMA_OS_OFFLINE"))
    max_turns_text = os.getenv("PHARMA_OS_AGENT_MAX_TURNS")
    try:
        max_turns = int(max_turns_text) if max_turns_text else 8
    except ValueError:
        max_turns = 8
    return AgentRuntimeConfig(
        model=os.getenv("PHARMA_OS_MODEL") or "gpt-5.5",
        max_turns=max_turns,
        disabled=disabled,
    )


def runtime_config_for_live_agents(*, disabled_provenance: str) -> AgentRuntimeConfig:
    """Resolve live/offline agent mode from environment.

    Live OpenAI Agents SDK calls are enabled when an API key is present, unless
    the operator explicitly disables agents or sets PHARMA_OS_ENABLE_LIVE_AGENTS=false.
    """

    env_config = runtime_config_from_env()
    if env_config.disabled:
        return env_config
    live_setting = os.getenv("PHARMA_OS_ENABLE_LIVE_AGENTS")
    if live_setting is not None and live_setting.strip() and not _truthy(live_setting):
        return env_config.model_copy(update={"disabled": True, "provenance": f"{disabled_provenance}.disabled_by_env"})
    if os.getenv("OPENAI_API_KEY"):
        return env_config
    return env_config.model_copy(update={"disabled": True, "provenance": f"{disabled_provenance}.missing_openai_api_key"})


def load_agents_sdk() -> tuple[Any, Any, Any, Any]:
    """Import OpenAI Agents SDK lazily so tests can run without live calls."""

    try:
        from agents import Agent, AgentOutputSchema, Runner, function_tool
    except ModuleNotFoundError as exc:
        raise AgentRuntimeError(
            "OpenAI Agents SDK is not installed. Install project dependencies before live agent runs."
        ) from exc
    return Agent, AgentOutputSchema, Runner, function_tool


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
            ),
            trace_metadata={
                "agent_name": agent_name,
                "model": settings.model,
                "max_turns": effective_max_turns,
                "disabled": True,
            },
        )

    _, _, Runner, _ = load_agents_sdk()
    try:
        response = Runner.run_sync(
            agent,
            _payload_json(payload),
            max_turns=effective_max_turns,
        )
    except TypeError:
        response = Runner.run_sync(agent, _payload_json(payload))
    except Exception as exc:
        raise AgentRuntimeError(f"Agent run failed: {exc.__class__.__name__}: {exc}") from exc

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
        ),
        trace_metadata={
            **_response_metadata(response),
            "agent_name": agent_name,
            "model": settings.model,
            "max_turns": effective_max_turns,
            "disabled": False,
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
        parsed, response_metadata = _call_openai_structured_output(
            model=settings.model,
            instructions=instructions,
            payload=payload,
            output_type=output_type,
        )
    except Exception as exc:
        if offline_output is None:
            if isinstance(exc, AgentRuntimeError):
                raise
            raise AgentRuntimeError(f"Direct OpenAI structured output call failed: {exc.__class__.__name__}: {exc}") from exc
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
                rationale_summary=rationale_summary
                or "Direct OpenAI call failed; offline structured output was validated as fallback.",
                tool_calls=(),
                provenance="pharma_os.agent_runtime.direct_openai_api_fallback",
            ),
            trace_metadata={
                "agent_name": agent_name,
                "model": settings.model,
                "disabled": False,
                "direct_api": True,
                "fallback": True,
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:500],
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
        ),
        trace_metadata={
            **response_metadata,
            "agent_name": agent_name,
            "model": settings.model,
            "disabled": False,
            "direct_api": True,
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
        ),
        trace_metadata={
            "agent_name": agent_name,
            "model": settings.model,
            "max_turns": settings.max_turns,
            "disabled": True,
            **(trace_metadata or {}),
        },
    )


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


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}
