from __future__ import annotations

from typing import Any

import pytest
from pydantic import Field

from pharma_os.agent_runtime import AgentRuntimeConfig, StructuredAgentResult, run_structured_llm_call
from pharma_os.agents import clinical_outcome_prediction, due_diligence, protocol_design
from pharma_os.human_readable import build_human_readable_module_output
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


@pytest.mark.parametrize(
    ("module", "agent_name", "expected_route"),
    [
        (clinical_outcome_prediction, "ClinicalOutcomeManagerAgent", "agent3_manager"),
        (clinical_outcome_prediction, "EndpointRiskAgent", "agent3_subagent"),
        (due_diligence, "DueDiligenceManagerAgent", "agent4_manager"),
        (due_diligence, "DiligenceRedTeamAgent", "agent4_subagent"),
        (protocol_design, "ProtocolDesignManagerAgent", "agent5_manager"),
        (protocol_design, "DevelopmentStrategyAgent", "agent5_manager"),
        (protocol_design, "ProtocolBriefWriterAgent", "agent5_manager"),
        (protocol_design, "AnalogSearchPlannerAgent", "agent5_subagent"),
    ],
)
def test_workflow_agents_use_expected_model_routes(module: Any, agent_name: str, expected_route: str, monkeypatch) -> None:
    seen_routes: list[str] = []

    def fake_direct(**kwargs: Any) -> StructuredAgentResult:
        seen_routes.append(kwargs["config"].model_route)
        return _direct_result(**kwargs)

    def fake_sdk(**kwargs: Any) -> StructuredAgentResult:
        seen_routes.append(kwargs["config"].model_route)
        return _direct_result(**kwargs)

    class FakeAgent:
        def __init__(self, *, name: str, instructions: str, model: str, output_type: type[Any]) -> None:
            del name, instructions, model, output_type

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("PHARMA_OS_MODEL", raising=False)
    monkeypatch.setattr(module, "run_structured_llm_call", fake_direct)
    monkeypatch.setattr(module, "run_structured_agent", fake_sdk)
    monkeypatch.setattr(module, "load_agents_sdk", lambda: (FakeAgent, object, object, object))

    result = module._run_typed_agent(
        agent_name=agent_name,
        instructions="Return FixtureOutput.",
        output_type=FixtureOutput,
        run_id="RUN",
        input_summary="Fixture routing input.",
        payload={"prompt": "fixture"},
        fallback_output=FixtureOutput(output_id=f"{agent_name}-OUT", summary="Fixture summary.", confidence=0.7),
        source_ids=("ctgov:NCT12345678",),
        confidence=0.7,
        config=None,
        rationale_summary="Fixture routing rationale.",
    )

    assert isinstance(result.output, FixtureOutput)
    assert seen_routes == [expected_route]


def test_human_readable_summary_uses_human_summary_route(monkeypatch) -> None:
    seen_routes: list[str] = []

    def fake_direct(**kwargs: Any) -> StructuredAgentResult:
        seen_routes.append(kwargs["config"].model_route)
        return _direct_result(**kwargs)

    monkeypatch.setattr("pharma_os.human_readable.run_structured_llm_call", fake_direct)

    result = build_human_readable_module_output(
        module_name="trial_intelligence",
        module_display_name="Trial Intelligence",
        run_id="RUN",
        typed_output=FixtureOutput(output_id="OUT", summary="Fixture summary.", confidence=0.7),
    )

    assert result.output.output_id == "human-readable-trial_intelligence-RUN"
    assert seen_routes == ["human_summary"]
