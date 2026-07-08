"""Minimal workflow orchestration entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pharma_os.memory import MemoryStore
from pharma_os.schemas import WorkflowRun
from pharma_os.validators import validate_workflow_name


class Orchestrator:
    """Coordinates workflow runs."""

    def __init__(self, memory: MemoryStore | None = None) -> None:
        self.memory = memory or MemoryStore()

    def run(self, workflow: str) -> WorkflowRun:
        """Create and complete a minimal workflow run."""

        now = datetime.now(timezone.utc)
        run = WorkflowRun(
            run_id=str(uuid4()),
            workflow_name=validate_workflow_name(workflow),
            status="running",
            started_at=now,
            input_provenance="cli.workflow_name",
        )
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc)
        return self.memory.save_run(run)
