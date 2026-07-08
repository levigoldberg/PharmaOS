from __future__ import annotations

from types import ModuleType, SimpleNamespace

from pharma_os.agents.runtime import run_agent
from pharma_os.schemas import ClinicalTrialIntelligenceInput, ClinicalTrialIntelligenceOutput


def test_run_agent_validates_final_output(monkeypatch) -> None:
    output = ClinicalTrialIntelligenceOutput(
        output_id="output-1",
        run_id="RUN",
        input=ClinicalTrialIntelligenceInput(disease="glioblastoma"),
        landscape_summary="No trials found.",
        status_summary="No status values were available.",
        phase_summary="No phase values were available.",
        sponsor_summary="No sponsor values were available.",
        endpoint_summary="0 primary endpoints were normalized across 0 trials.",
        population_summary="No enrollment counts were available.",
    )

    class Runner:
        @staticmethod
        def run_sync(agent, payload):
            return SimpleNamespace(final_output=output, trace_id="trace-1")

    module = ModuleType("agents")
    module.Agent = object
    module.AgentOutputSchema = object
    module.Runner = Runner
    module.function_tool = lambda func: func
    monkeypatch.setitem(__import__("sys").modules, "agents", module)

    result = run_agent(object(), {"input": "test"}, ClinicalTrialIntelligenceOutput)

    assert result.output == output
    assert result.trace_metadata["trace_id"] == "trace-1"
