"""Small OpenAI Agents SDK runtime wrapper."""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class AgentRuntimeError(RuntimeError):
    """Raised when an Agents SDK run cannot complete."""


class AgentRunResult(BaseModel):
    """Normalized result from an Agents SDK run."""

    output: BaseModel
    trace_metadata: dict[str, str | int | float | bool | None] = {}


def load_agents_sdk() -> tuple[Any, Any, Any, Any]:
    """Import Agents SDK objects lazily so tests can run without live SDK imports."""

    try:
        from agents import Agent, AgentOutputSchema, Runner, function_tool
    except ModuleNotFoundError as exc:
        raise AgentRuntimeError(
            "OpenAI Agents SDK is not installed. Install project dependencies before live agent runs."
        ) from exc
    return Agent, AgentOutputSchema, Runner, function_tool


def run_agent(agent: Any, payload: dict[str, Any], output_type: type[T]) -> AgentRunResult:
    """Run one agent synchronously and validate its structured output."""

    _, _, Runner, _ = load_agents_sdk()
    try:
        response = Runner.run_sync(agent, json.dumps(payload, ensure_ascii=False))
        parsed = getattr(response, "final_output", response)
        if not isinstance(parsed, output_type):
            parsed = output_type.model_validate(parsed)
        return AgentRunResult(
            output=parsed,
            trace_metadata=_trace_metadata(response),
        )
    except AgentRuntimeError:
        raise
    except Exception as exc:
        raise AgentRuntimeError(f"Agent run failed: {exc.__class__.__name__}: {exc}") from exc


def _trace_metadata(response: Any) -> dict[str, str | int | float | bool | None]:
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
