from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_tests_to_offline_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests network-independent even when a local .env has OPENAI_API_KEY."""

    monkeypatch.setenv("PHARMA_OS_AGENTS_DISABLED", "true")
