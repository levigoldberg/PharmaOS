from __future__ import annotations

import pytest
from pydantic import Field

from pharma_os.agent_runtime import AgentRuntimeConfig, AgentRuntimeError, run_structured_agent
from pharma_os.schemas import StrictSchema


class FixtureOutput(StrictSchema):
    output_id: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0, le=1)


def test_run_structured_agent_offline_validates_output_and_trace() -> None:
    result = run_structured_agent(
        agent=object(),
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        agent_name="fixture_agent",
        run_id="RUN",
        input_summary="Fixture input.",
        config=AgentRuntimeConfig(model="test-model", max_turns=3, disabled=True),
        offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
        source_ids=("ctgov:NCT12345678",),
        confidence=0.7,
        rationale_summary="Fixture rationale summary.",
    )

    assert isinstance(result.output, FixtureOutput)
    assert result.trace.agent_name == "fixture_agent"
    assert result.trace.output_id == "OUT"
    assert result.trace.output_type == "FixtureOutput"
    assert result.trace.rationale_summary == "Fixture rationale summary."
    assert result.trace_metadata["disabled"] is True
    assert "chain" not in result.trace.model_dump_json().casefold()


def test_run_structured_agent_offline_requires_fixture_output() -> None:
    with pytest.raises(AgentRuntimeError, match="disabled/offline"):
        run_structured_agent(
            agent=object(),
            payload={},
            output_type=FixtureOutput,
            agent_name="fixture_agent",
            run_id="RUN",
            input_summary="Fixture input.",
            config=AgentRuntimeConfig(disabled=True),
        )
