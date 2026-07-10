from __future__ import annotations

import pytest
from pydantic import Field, ValidationError

from pharma_os.agent_runtime import (
    AgentRuntimeConfig,
    AgentRuntimeError,
    StructuredAgentResult,
    resolve_model_for_route,
    run_structured_agent,
    run_structured_llm_call,
    runtime_config_from_env,
    runtime_config_for_live_agents,
)
from pharma_os.schemas import StrictSchema


class FixtureOutput(StrictSchema):
    output_id: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0, le=1)


class TransientRateLimitError(RuntimeError):
    status_code = 429


def test_model_route_specific_env_overrides_global_model(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OS_MODEL", "global-model")
    monkeypatch.setenv("PHARMA_OS_MODEL_CONTROL_TOWER", "control-model")

    config = runtime_config_from_env(model_route="control_tower")

    assert config.model == "control-model"
    assert config.model_route == "control_tower"
    assert resolve_model_for_route("control_tower") == "control-model"


def test_global_model_preserves_backward_compatible_override(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OS_MODEL", "global-model")
    monkeypatch.delenv("PHARMA_OS_MODEL_REQUEST_UNDERSTANDING", raising=False)

    config = runtime_config_from_env(model_route="request_understanding")

    assert config.model == "global-model"
    assert config.model_route == "request_understanding"


def test_model_route_tier_defaults_apply_without_env(monkeypatch) -> None:
    for key in (
        "PHARMA_OS_MODEL",
        "PHARMA_OS_MODEL_REQUEST_UNDERSTANDING",
        "PHARMA_OS_MODEL_CONTROL_TOWER",
        "PHARMA_OS_MODEL_AGENT5_MANAGER",
    ):
        monkeypatch.delenv(key, raising=False)

    assert runtime_config_from_env(model_route="request_understanding").model == "gpt-5.6-luna"
    assert runtime_config_from_env(model_route="control_tower").model == "gpt-5.6-terra"
    assert runtime_config_from_env(model_route="agent5_manager").model == "gpt-5.6-sol"


def test_unknown_model_route_falls_back_safely(monkeypatch) -> None:
    monkeypatch.delenv("PHARMA_OS_MODEL", raising=False)

    config = runtime_config_from_env(model_route="not-a-real-route")

    assert config.model_route == "default"
    assert config.model == "gpt-5.6-terra"


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
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace.steps[0].execution_mode == "deterministic_fallback"
    assert result.trace_metadata["disabled"] is True
    assert result.trace_metadata["execution_mode"] == "deterministic_fallback"
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


def test_run_structured_llm_call_offline_validates_output_and_trace() -> None:
    result = run_structured_llm_call(
        agent_name="fixture_direct_agent",
        instructions="Return FixtureOutput.",
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        run_id="RUN",
        input_summary="Fixture direct input.",
        config=AgentRuntimeConfig(model="test-model", disabled=True),
        offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
        source_ids=("ctgov:NCT12345678",),
        confidence=0.7,
        rationale_summary="Fixture direct rationale summary.",
    )

    assert isinstance(result.output, FixtureOutput)
    assert result.trace.agent_name == "fixture_direct_agent"
    assert result.trace.output_id == "OUT"
    assert result.trace.output_type == "FixtureOutput"
    assert result.trace.provenance == "pharma_os.agent_runtime.offline"
    assert result.trace.steps[0].provenance == "pharma_os.agent_runtime.offline"
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace_metadata["direct_api"] is True
    assert result.trace_metadata["disabled"] is True


def test_run_structured_llm_call_uses_offline_output_when_api_key_missing(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)
    monkeypatch.delenv("PHARMA_OS_ENABLE_LIVE_AGENTS", raising=False)

    result = run_structured_llm_call(
        agent_name="fixture_direct_agent",
        instructions="Return FixtureOutput.",
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        run_id="RUN",
        input_summary="Fixture direct input.",
        config=AgentRuntimeConfig(model="test-model", disabled=False),
        offline_output=FixtureOutput(output_id="OUT", summary="Fixture summary.", confidence=0.7),
    )

    assert isinstance(result.output, FixtureOutput)
    assert result.trace_metadata["disabled"] is True
    assert result.trace_metadata["direct_api"] is True


def test_run_structured_llm_call_validates_output_type() -> None:
    with pytest.raises(ValidationError):
        run_structured_llm_call(
            agent_name="fixture_direct_agent",
            instructions="Return FixtureOutput.",
            payload={"prompt": "fixture"},
            output_type=FixtureOutput,
            run_id="RUN",
            input_summary="Fixture direct input.",
            config=AgentRuntimeConfig(disabled=True),
            offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 3.0},
        )


def test_run_structured_llm_call_live_path_returns_structured_result(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)
    monkeypatch.delenv("PHARMA_OS_ENABLE_LIVE_AGENTS", raising=False)

    def fake_call(**kwargs):
        assert kwargs["model"] == "test-model"
        assert kwargs["instructions"] == "Return FixtureOutput."
        return {"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7}, {"last_response_id": "resp_123"}

    monkeypatch.setattr("pharma_os.agent_runtime._call_openai_structured_output", fake_call)

    result = run_structured_llm_call(
        agent_name="fixture_direct_agent",
        instructions="Return FixtureOutput.",
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        run_id="RUN",
        input_summary="Fixture direct input.",
        config=AgentRuntimeConfig(model="test-model", disabled=False),
        offline_output={"output_id": "OFFLINE", "summary": "Offline summary.", "confidence": 0.5},
        source_ids=("ctgov:NCT12345678",),
    )

    assert isinstance(result, StructuredAgentResult)
    assert isinstance(result.output, FixtureOutput)
    assert result.output.output_id == "OUT"
    assert result.trace.provenance == "pharma_os.agent_runtime.openai_api_structured_output"
    assert result.trace.execution_mode == "direct_llm"
    assert result.trace_metadata["last_response_id"] == "resp_123"
    assert result.trace_metadata["execution_mode"] == "direct_llm"


def test_run_structured_llm_call_falls_back_after_api_failure(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)
    monkeypatch.delenv("PHARMA_OS_ENABLE_LIVE_AGENTS", raising=False)
    monkeypatch.delenv("PHARMA_OS_DISABLE_AGENT_FALLBACKS", raising=False)

    def fail_call(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("pharma_os.agent_runtime._call_openai_structured_output", fail_call)

    result = run_structured_llm_call(
        agent_name="fixture_direct_agent",
        instructions="Return FixtureOutput.",
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        run_id="RUN",
        input_summary="Fixture direct input.",
        config=AgentRuntimeConfig(model="test-model", disabled=False),
        offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
    )

    assert isinstance(result.output, FixtureOutput)
    assert result.trace.provenance == "pharma_os.agent_runtime.direct_openai_api_fallback"
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace_metadata["fallback"] is True
    assert result.trace_metadata["execution_mode"] == "deterministic_fallback"
    assert result.trace_metadata["error_type"] == "RuntimeError"
    assert result.trace_metadata["retry_count"] == 0


def test_run_structured_llm_call_retries_rate_limit_before_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PHARMA_OS_LLM_MAX_RETRIES", "3")
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)
    monkeypatch.delenv("PHARMA_OS_ENABLE_LIVE_AGENTS", raising=False)
    monkeypatch.delenv("PHARMA_OS_DISABLE_AGENT_FALLBACKS", raising=False)
    monkeypatch.setattr("pharma_os.agent_runtime.time.sleep", lambda _: None)
    calls = {"count": 0}

    def fail_call(**kwargs):
        calls["count"] += 1
        raise TransientRateLimitError("rate limit")

    monkeypatch.setattr("pharma_os.agent_runtime._call_openai_structured_output", fail_call)

    result = run_structured_llm_call(
        agent_name="fixture_direct_agent",
        instructions="Return FixtureOutput.",
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        run_id="RUN",
        input_summary="Fixture direct input.",
        config=AgentRuntimeConfig(model="test-model", model_route="test_route", disabled=False),
        offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
    )

    assert calls["count"] == 3
    assert result.trace_metadata["fallback"] is True
    assert result.trace_metadata["fallback_cause"] == "rate_limit"
    assert result.trace_metadata["retry_count"] == 2
    assert result.trace_metadata["retry_exhausted"] is True
    assert result.trace.model == "test-model"
    assert result.trace.model_route == "test_route"
    assert result.trace.retry_count == 2
    assert result.trace.fallback_cause == "rate_limit"


def test_run_structured_llm_call_does_not_retry_non_transient_failure(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PHARMA_OS_LLM_MAX_RETRIES", "3")
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)
    monkeypatch.delenv("PHARMA_OS_ENABLE_LIVE_AGENTS", raising=False)
    calls = {"count": 0}

    def fail_call(**kwargs):
        calls["count"] += 1
        raise RuntimeError("schema rejected")

    monkeypatch.setattr("pharma_os.agent_runtime._call_openai_structured_output", fail_call)

    result = run_structured_llm_call(
        agent_name="fixture_direct_agent",
        instructions="Return FixtureOutput.",
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        run_id="RUN",
        input_summary="Fixture direct input.",
        config=AgentRuntimeConfig(model="test-model", disabled=False),
        offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
    )

    assert calls["count"] == 1
    assert result.trace_metadata["retry_count"] == 0


def test_run_structured_llm_call_can_disable_api_failure_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PHARMA_OS_DISABLE_AGENT_FALLBACKS", "true")
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)
    monkeypatch.delenv("PHARMA_OS_ENABLE_LIVE_AGENTS", raising=False)

    def fail_call(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("pharma_os.agent_runtime._call_openai_structured_output", fail_call)

    with pytest.raises(AgentRuntimeError, match="fallbacks disabled"):
        run_structured_llm_call(
            agent_name="fixture_direct_agent",
            instructions="Return FixtureOutput.",
            payload={"prompt": "fixture"},
            output_type=FixtureOutput,
            run_id="RUN",
            input_summary="Fixture direct input.",
            config=AgentRuntimeConfig(model="test-model", disabled=False),
            offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
        )


def test_run_structured_agent_live_path_marks_live_agent(monkeypatch) -> None:
    class Response:
        final_output = {"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7}

    class SuccessRunner:
        @staticmethod
        def run_sync(*args, **kwargs):
            return Response()

    monkeypatch.setattr(
        "pharma_os.agent_runtime.load_agents_sdk",
        lambda: (object, object, SuccessRunner, object),
    )

    result = run_structured_agent(
        agent=object(),
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        agent_name="fixture_agent",
        run_id="RUN",
        input_summary="Fixture input.",
        config=AgentRuntimeConfig(model="test-model", disabled=False),
        offline_output={"output_id": "OFFLINE", "summary": "Offline summary.", "confidence": 0.5},
    )

    assert isinstance(result.output, FixtureOutput)
    assert result.output.output_id == "OUT"
    assert result.trace.execution_mode == "live_agent"
    assert result.trace_metadata["execution_mode"] == "live_agent"


def test_run_structured_agent_falls_back_after_sdk_failure(monkeypatch) -> None:
    monkeypatch.delenv("PHARMA_OS_DISABLE_AGENT_FALLBACKS", raising=False)

    class FailingRunner:
        @staticmethod
        def run_sync(*args, **kwargs):
            raise RuntimeError("context_length_exceeded")

    monkeypatch.setattr(
        "pharma_os.agent_runtime.load_agents_sdk",
        lambda: (object, object, FailingRunner, object),
    )

    result = run_structured_agent(
        agent=object(),
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        agent_name="fixture_agent",
        run_id="RUN",
        input_summary="Fixture input.",
        config=AgentRuntimeConfig(model="test-model", disabled=False),
        offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
    )

    assert isinstance(result.output, FixtureOutput)
    assert result.trace.provenance == "pharma_os.agent_runtime.openai_agents_sdk_fallback"
    assert result.trace.execution_mode == "deterministic_fallback"
    assert result.trace_metadata["fallback"] is True
    assert result.trace_metadata["execution_mode"] == "deterministic_fallback"
    assert result.trace_metadata["error_type"] == "RuntimeError"
    assert result.trace_metadata["retry_count"] == 0


def test_run_structured_agent_retries_transient_sdk_failure_before_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OS_LLM_MAX_RETRIES", "3")
    monkeypatch.delenv("PHARMA_OS_DISABLE_AGENT_FALLBACKS", raising=False)
    monkeypatch.setattr("pharma_os.agent_runtime.time.sleep", lambda _: None)
    calls = {"count": 0}

    class FailingRunner:
        @staticmethod
        def run_sync(*args, **kwargs):
            calls["count"] += 1
            raise TransientRateLimitError("rate limit")

    monkeypatch.setattr(
        "pharma_os.agent_runtime.load_agents_sdk",
        lambda: (object, object, FailingRunner, object),
    )

    result = run_structured_agent(
        agent=object(),
        payload={"prompt": "fixture"},
        output_type=FixtureOutput,
        agent_name="fixture_agent",
        run_id="RUN",
        input_summary="Fixture input.",
        config=AgentRuntimeConfig(model="test-model", model_route="sdk_route", disabled=False),
        offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
    )

    assert calls["count"] == 3
    assert result.trace_metadata["fallback"] is True
    assert result.trace_metadata["fallback_cause"] == "rate_limit"
    assert result.trace_metadata["retry_count"] == 2
    assert result.trace.model_route == "sdk_route"


def test_run_structured_agent_raises_after_transient_retries_when_fallbacks_disabled(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OS_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("PHARMA_OS_DISABLE_AGENT_FALLBACKS", "true")
    monkeypatch.setattr("pharma_os.agent_runtime.time.sleep", lambda _: None)
    calls = {"count": 0}

    class FailingRunner:
        @staticmethod
        def run_sync(*args, **kwargs):
            calls["count"] += 1
            raise TransientRateLimitError("rate limit")

    monkeypatch.setattr(
        "pharma_os.agent_runtime.load_agents_sdk",
        lambda: (object, object, FailingRunner, object),
    )

    with pytest.raises(AgentRuntimeError, match="transient error"):
        run_structured_agent(
            agent=object(),
            payload={"prompt": "fixture"},
            output_type=FixtureOutput,
            agent_name="fixture_agent",
            run_id="RUN",
            input_summary="Fixture input.",
            config=AgentRuntimeConfig(model="test-model", disabled=False),
            offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
        )

    assert calls["count"] == 2


def test_run_structured_agent_can_disable_sdk_failure_fallback(monkeypatch) -> None:
    monkeypatch.setenv("PHARMA_OS_DISABLE_AGENT_FALLBACKS", "true")

    class FailingRunner:
        @staticmethod
        def run_sync(*args, **kwargs):
            raise RuntimeError("context_length_exceeded")

    monkeypatch.setattr(
        "pharma_os.agent_runtime.load_agents_sdk",
        lambda: (object, object, FailingRunner, object),
    )

    with pytest.raises(AgentRuntimeError, match="fallbacks disabled"):
        run_structured_agent(
            agent=object(),
            payload={"prompt": "fixture"},
            output_type=FixtureOutput,
            agent_name="fixture_agent",
            run_id="RUN",
            input_summary="Fixture input.",
            config=AgentRuntimeConfig(model="test-model", disabled=False),
            offline_output={"output_id": "OUT", "summary": "Fixture summary.", "confidence": 0.7},
        )


def test_runtime_config_enables_live_agents_when_api_key_present(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("PHARMA_OS_ENABLE_LIVE_AGENTS", raising=False)
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)

    config = runtime_config_for_live_agents(disabled_provenance="test")

    assert config.disabled is False


def test_runtime_config_treats_blank_live_agent_setting_as_unset(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PHARMA_OS_ENABLE_LIVE_AGENTS", "")
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)

    config = runtime_config_for_live_agents(disabled_provenance="test")

    assert config.disabled is False


def test_runtime_config_respects_explicit_live_agent_disable(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PHARMA_OS_ENABLE_LIVE_AGENTS", "false")
    monkeypatch.delenv("PHARMA_OS_AGENTS_DISABLED", raising=False)
    monkeypatch.delenv("PHARMA_OS_OFFLINE", raising=False)

    config = runtime_config_for_live_agents(disabled_provenance="test")

    assert config.disabled is True
    assert config.provenance == "test.disabled_by_env"
