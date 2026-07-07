"""Minimal workflow orchestration entry points."""

from __future__ import annotations

from pharma_os.memory import MemoryStore
from pharma_os.schemas import WorkflowRun, utc_now
from pharma_os.validators import validate_workflow_name


class Orchestrator:
    """Coordinates workflow runs."""

    def __init__(self, memory: MemoryStore | None = None) -> None:
        self.memory = memory or MemoryStore()

    def run(self, workflow: str) -> WorkflowRun:
        """Create and complete a minimal workflow run."""

        run = WorkflowRun(workflow=validate_workflow_name(workflow), status="running")
        run.status = "completed"
        run.updated_at = utc_now()
        return self.memory.save_run(run)
