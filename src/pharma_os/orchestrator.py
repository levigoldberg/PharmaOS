"""Workflow orchestration entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pharma_os.memory import MemoryStore
from pharma_os.schemas import ClinicalTrialIntelligenceInput, WorkflowRun
from pharma_os.validators import validate_workflow_name
from pharma_os.workflows.trial_intelligence import run_trial_intelligence_workflow


class Orchestrator:
    """Coordinates workflow runs."""

    def __init__(self, memory: MemoryStore | None = None) -> None:
        self.memory = memory or MemoryStore()

    def run(
        self,
        workflow: str,
        input_data: ClinicalTrialIntelligenceInput | None = None,
    ) -> object:
        """Run a workflow by name."""

        workflow_name = validate_workflow_name(workflow)
        if workflow_name == "trial_intelligence":
            if input_data is None:
                raise ValueError("trial_intelligence requires ClinicalTrialIntelligenceInput")
            return run_trial_intelligence_workflow(input_data, memory=self.memory)
        return self._placeholder_run(workflow_name)

    def _placeholder_run(self, workflow_name: str) -> WorkflowRun:
        now = datetime.now(timezone.utc)
        run = WorkflowRun(
            run_id=str(uuid4()),
            workflow_name=workflow_name,
            status="completed",
            started_at=now,
            completed_at=datetime.now(timezone.utc),
            input_provenance="cli.workflow_name",
            validation_status="not_run",
        )
        return self.memory.save_run(run)
