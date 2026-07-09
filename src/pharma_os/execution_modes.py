"""Execution-mode aggregation for visible AI audit reporting."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pharma_os.schemas import AgentRunTrace, ExecutionMode, ExecutionModeSummary


def summarize_execution_modes(
    traces: tuple[AgentRunTrace, ...] = (),
    *,
    reused_artifacts: int = 0,
) -> ExecutionModeSummary:
    """Build a visible execution-mode summary from persisted traces."""

    live_agent = sum(1 for trace in traces if trace.execution_mode == "live_agent")
    direct_llm = sum(1 for trace in traces if trace.execution_mode == "direct_llm")
    fallbacks = sum(1 for trace in traces if trace.execution_mode == "deterministic_fallback")
    live_ai = live_agent + direct_llm
    requested = live_ai + fallbacks
    return ExecutionModeSummary(
        requested_reasoning_steps=requested,
        live_agent_calls_completed=live_agent,
        direct_llm_calls_completed=direct_llm,
        live_ai_calls_completed=live_ai,
        deterministic_fallbacks_used=fallbacks,
        reused_artifacts_used=max(0, reused_artifacts),
        summary=_summary_text(
            requested=requested,
            live_ai=live_ai,
            fallbacks=fallbacks,
            reused=max(0, reused_artifacts),
        ),
    )


def combine_execution_mode_summaries(*summaries: ExecutionModeSummary, reused_artifacts: int = 0) -> ExecutionModeSummary:
    """Combine execution summaries from nested workflow outputs."""

    requested = sum(summary.requested_reasoning_steps for summary in summaries)
    live_agent = sum(summary.live_agent_calls_completed for summary in summaries)
    direct_llm = sum(summary.direct_llm_calls_completed for summary in summaries)
    live_ai = sum(summary.live_ai_calls_completed for summary in summaries)
    fallbacks = sum(summary.deterministic_fallbacks_used for summary in summaries)
    reused = sum(summary.reused_artifacts_used for summary in summaries) + max(0, reused_artifacts)
    return ExecutionModeSummary(
        requested_reasoning_steps=requested,
        live_agent_calls_completed=live_agent,
        direct_llm_calls_completed=direct_llm,
        live_ai_calls_completed=live_ai,
        deterministic_fallbacks_used=fallbacks,
        reused_artifacts_used=reused,
        summary=_summary_text(requested=requested, live_ai=live_ai, fallbacks=fallbacks, reused=reused),
    )


def primary_execution_mode(summary: ExecutionModeSummary) -> ExecutionMode:
    """Return the most important visible mode for a combined output envelope."""

    if summary.deterministic_fallbacks_used:
        return "deterministic_fallback"
    if summary.live_agent_calls_completed:
        return "live_agent"
    if summary.direct_llm_calls_completed:
        return "direct_llm"
    if summary.reused_artifacts_used:
        return "reused_artifact"
    return "deterministic_fallback"


def execution_mode_summary_for_mode(mode: ExecutionMode) -> ExecutionModeSummary:
    """Build a one-output summary for an explicit execution mode."""

    if mode == "live_agent":
        return summarize_execution_modes(
            (
                AgentRunTrace(
                    trace_id="summary-live-agent",
                    run_id="summary",
                    agent_name="summary",
                    provenance="pharma_os.execution_modes.summary",
                    execution_mode="live_agent",
                ),
            )
        )
    if mode == "direct_llm":
        return summarize_execution_modes(
            (
                AgentRunTrace(
                    trace_id="summary-direct-llm",
                    run_id="summary",
                    agent_name="summary",
                    provenance="pharma_os.execution_modes.summary",
                    execution_mode="direct_llm",
                ),
            )
        )
    if mode == "reused_artifact":
        return summarize_execution_modes((), reused_artifacts=1)
    return summarize_execution_modes(
        (
            AgentRunTrace(
                trace_id="summary-deterministic-fallback",
                run_id="summary",
                agent_name="summary",
                provenance="pharma_os.execution_modes.summary",
                execution_mode="deterministic_fallback",
            ),
        )
    )


def execution_mode_for_payload(payload: Any, traces: tuple[AgentRunTrace, ...]) -> ExecutionMode:
    """Find the execution mode for a typed subagent payload."""

    output_id = _payload_output_id(payload)
    output_type = payload.__class__.__name__
    for trace in traces:
        if output_id and trace.output_id == output_id:
            return trace.execution_mode
    for trace in traces:
        if trace.output_type == output_type:
            return trace.execution_mode
    return "deterministic_fallback"


def reused_artifacts_from_output(output: Any) -> int:
    """Count explicit reused-artifact handoffs in a workflow output payload."""

    if isinstance(output, BaseModel):
        output = output.model_dump(mode="json")
    if not isinstance(output, dict):
        return 0
    count = 0
    for key in ("agent3_handoff", "agent4_handoff"):
        handoff = output.get(key)
        if isinstance(handoff, dict) and handoff.get("retrieved_from_memory") is True:
            count += 1
        if isinstance(handoff, dict) and handoff.get("generated_or_reused") == "reused":
            count += 1 if handoff.get("retrieved_from_memory") is not True else 0
    for step in output.get("step_results") or ():
        if isinstance(step, dict) and step.get("status") == "reused":
            count += 1
    return count


def _payload_output_id(payload: Any) -> str | None:
    for attr in ("output_id", "brief_id", "memo_id"):
        value = getattr(payload, attr, None)
        if value:
            return str(value)
    return None


def _summary_text(*, requested: int, live_ai: int, fallbacks: int, reused: int) -> str:
    text = (
        f"{requested} reasoning steps requested, "
        f"{live_ai} live AI calls completed, "
        f"{fallbacks} deterministic fallbacks used."
    )
    if reused:
        text += f" {reused} reused artifacts used."
    return text
