from __future__ import annotations

from typing import Any

import pytest
from pydantic import Field

from pharma_os.agent_runtime import AgentRuntimeConfig, StructuredAgentResult, run_structured_llm_call
from pharma_os.agents import clinical_outcome_prediction, due_diligence, protocol_design
from pharma_os.schemas import StrictSchema


class FixtureOutput(StrictSchema):
    output_id: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0, le=1)


def _direct_result(**kwargs: Any) -> StructuredAgentResult:
    return run_structured_llm_call(
        agent_name=kwargs["agent_name"],
        instructions=kwargs.get("instructions", "Return FixtureOutput."),
        payload=kwargs["payload"],
        output_type=kwargs["output_type"],
        run_id=kwargs["run_id"],
        input_summary=kwargs["input_summary"],
        config=AgentRuntimeConfig(disabled=True),
        offline_output=kwargs["offline_output"],
        source_ids=kwargs["source_ids"],
        confidence=kwargs["confidence"],
        rationale_summary=kwargs["rationale_summary"],
    )


def _run_private_typed_agent(module: Any, agent_name: str) -> StructuredAgentResult:
    return module._run_typed_agent(
        agent_name=agent_name,
        instructions="Return FixtureOutput.",
        output_type=FixtureOutput,
        run_id="RUN",
        input_summary="Fixture routing input.",
        payload={"prompt": "fixture"},
        fallback_output=FixtureOutput(output_id=f"{agent_name}-OUT", summary="Fixture summary.", confidence=0.7),
        source_ids=("ctgov:NCT12345678",),
        confidence=0.7,
        config=AgentRuntimeConfig(model="test-model", disabled=False),
        rationale_summary="Fixture routing rationale.",
    )


@pytest.mark.parametrize(
    ("module", "agent_name"),
    [
        (clinical_outcome_prediction, "AssetIdentityAdjudicatorAgent"),
        (clinical_outcome_prediction, "EndpointRiskAgent"),
        (clinical_outcome_prediction, "ComparatorRelevanceAgent"),
        (clinical_outcome_prediction, "EnrollmentFeasibilityAgent"),
        (clinical_outcome_prediction, "SafetyContextAgent"),
        (clinical_outcome_prediction, "FailureModeSynthesisAgent"),
        (due_diligence, "ClinicalEvidenceSynthesisAgent"),
        (due_diligence, "CompetitiveLandscapeAgent"),
        (due_diligence, "SafetyDiligenceAgent"),
        (protocol_design, "EndpointStrategyAgent"),
        (protocol_design, "PopulationEligibilityAgent"),
        (protocol_design, "ComparatorDesignAgent"),
        (protocol_design, "SafetyMonitoringAgent"),
        (protocol_design, "StatisticalSkeletonAgent"),
    ],
)
def test_converted_allowlist_uses_direct_api_without_agents_sdk(module: Any, agent_name: str, monkeypatch) -> None:
    direct_calls: list[str] = []

    def fake_direct(**kwargs: Any) -> StructuredAgentResult:
        direct_calls.append(kwargs["agent_name"])
        return _direct_result(**kwargs)

    monkeypatch.setattr(module, "run_structured_llm_call", fake_direct)
    monkeypatch.setattr(module, "run_structured_agent", lambda **_: pytest.fail("Agents SDK runtime should not run"))
    monkeypatch.setattr(module, "load_agents_sdk", lambda: pytest.fail("Agent should not be instantiated"))

    result = _run_private_typed_agent(module, agent_name)

    assert isinstance(result.output, FixtureOutput)
    assert direct_calls == [agent_name]
    assert result.trace.agent_name == agent_name


@pytest.mark.parametrize(
    ("module", "agent_name"),
    [
        (due_diligence, "DiligenceRedTeamAgent"),
        (due_diligence, "AssetMemoAgent"),
        (protocol_design, "RegulatoryCriticAgent"),
        (protocol_design, "ProtocolBriefWriterAgent"),
    ],
)
def test_borderline_agents_remain_on_agents_sdk_path(module: Any, agent_name: str, monkeypatch) -> None:
    instantiated: list[str] = []
    sdk_calls: list[str] = []

    class FakeAgent:
        def __init__(self, *, name: str, instructions: str, model: str, output_type: type[Any]) -> None:
            del instructions, model, output_type
            self.name = name
            instantiated.append(name)

    def fake_sdk_runtime(**kwargs: Any) -> StructuredAgentResult:
        sdk_calls.append(kwargs["agent_name"])
        assert kwargs["agent"].name == agent_name
        return _direct_result(**kwargs)

    monkeypatch.setattr(module, "run_structured_llm_call", lambda **_: pytest.fail("Direct API path should not run"))
    monkeypatch.setattr(module, "load_agents_sdk", lambda: (FakeAgent, object, object, object))
    monkeypatch.setattr(module, "run_structured_agent", fake_sdk_runtime)

    result = _run_private_typed_agent(module, agent_name)

    assert isinstance(result.output, FixtureOutput)
    assert instantiated == [agent_name]
    assert sdk_calls == [agent_name]
