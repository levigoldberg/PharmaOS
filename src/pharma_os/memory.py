"""In-memory storage primitives for early PharmaOS development."""

from __future__ import annotations

from pharma_os.schemas import WorkflowRun


class MemoryStore:
    """Small in-process store for workflow runs."""

    def __init__(self) -> None:
        self._runs: dict[str, WorkflowRun] = {}

    def save_run(self, run: WorkflowRun) -> WorkflowRun:
        """Persist a workflow run in memory."""

        self._runs[run.run_id] = run
        return run

    def get_run(self, run_id: str) -> WorkflowRun | None:
        """Return a workflow run by id, if present."""

        return self._runs.get(run_id)
