"""Workflow orchestration entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pharma_os.control_tower import run_control_tower_agent, validate_execution_plan
from pharma_os.memory import MemoryStore
from pharma_os.registry import WorkflowRegistry
from pharma_os.schemas import (
    AgentOutput,
    ClinicalOutcomePredictionInput,
    ClinicalTrialIntelligenceInput,
    DueDiligenceInput,
    OrchestrationRequest,
    OrchestrationRunRecord,
    ProtocolDesignInput,
    WorkflowRun,
)
from pharma_os.validators import validate_workflow_name
from pharma_os.workflows.clinical_outcome_prediction import run_clinical_outcome_prediction_workflow
from pharma_os.workflows.due_diligence import run_due_diligence_workflow
from pharma_os.workflows.protocol_design import run_protocol_design_workflow
from pharma_os.workflows.trial_intelligence import run_trial_intelligence_workflow


_WORKFLOW_RUNNERS = {
    "trial_intelligence": run_trial_intelligence_workflow,
    "clinical_outcome_prediction": run_clinical_outcome_prediction_workflow,
    "due_diligence": run_due_diligence_workflow,
    "protocol_design": run_protocol_design_workflow,
}

_INPUT_TYPES = {
    "trial_intelligence": ClinicalTrialIntelligenceInput,
    "clinical_outcome_prediction": ClinicalOutcomePredictionInput,
    "due_diligence": DueDiligenceInput,
    "protocol_design": ProtocolDesignInput,
}


class Orchestrator:
    """Coordinates workflow runs and planning-only Control Tower requests."""

    def __init__(self, memory: MemoryStore | None = None, registry: WorkflowRegistry | None = None) -> None:
        self.memory = memory or MemoryStore()
        self.registry = registry or WorkflowRegistry.default()

    def run(
        self,
        workflow: str,
        input_data: ClinicalTrialIntelligenceInput | DueDiligenceInput | ClinicalOutcomePredictionInput | ProtocolDesignInput | None = None,
    ) -> object:
        """Run a workflow by name."""

        workflow_name = validate_workflow_name(workflow)
        if workflow_name == "trial_intelligence":
            return self._run_registered_workflow(workflow_name, input_data)
        capability = self.registry.get(workflow_name)
        if capability is None:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        if not capability.executable:
            raise ValueError(f"Workflow capability is not executable: {workflow_name}")
        return self._run_registered_workflow(workflow_name, input_data)

    def plan(self, request: OrchestrationRequest) -> OrchestrationRunRecord:
        """Build and persist a planning-only Control Tower execution plan."""

        run_id = f"control-tower-{uuid4()}"
        started_at = datetime.now(timezone.utc)
        snapshot = self.memory.build_scientific_state_snapshot(request, registry=self.registry)
        agent_result = run_control_tower_agent(
            run_id=run_id,
            request=request,
            snapshot=snapshot,
            registry=self.registry,
        )
        validation_results = validate_execution_plan(
            run_id=run_id,
            plan=agent_result.output,
            snapshot=snapshot,
            registry=self.registry,
        )
        failed = any(result.status == "failed" for result in validation_results)
        validation_status = "failed" if failed else "needs_human_review" if agent_result.output.blocked else "passed"
        plan = agent_result.output.model_copy(update={"validation_status": validation_status})
        completed_at = datetime.now(timezone.utc)
        run = WorkflowRun(
            run_id=run_id,
            workflow_name="control_tower",
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
            input_provenance="control_tower.plan",
            validation_status=validation_status,
            gate_reason="; ".join(plan.block_reasons) if plan.block_reasons else None,
            metadata={"objective": request.objective[:500], "nct_id": request.nct_id},
        )
        self.memory.save_run(
            run,
            input_payload=request,
            output_payload=plan,
            trace_metadata=agent_result.trace_metadata,
        )
        self.memory.save_agent_trace(agent_result.trace)
        self.memory.save_agent_output(
            AgentOutput(
                output_id=plan.output_id,
                agent_name="ControlTowerAgent",
                run_id=run_id,
                provenance=agent_result.trace.provenance,
                confidence=plan.confidence,
                validation_status=validation_status,
                gate_reason=run.gate_reason,
            ),
            payload=plan,
        )
        self.memory.save_validation_results(run_id, validation_results)
        return OrchestrationRunRecord(
            run_id=run_id,
            request=request,
            snapshot=snapshot,
            plan=plan,
            validation_results=validation_results,
            trace=agent_result.trace,
        )

    def _run_registered_workflow(
        self,
        workflow_name: str,
        input_data: ClinicalTrialIntelligenceInput | DueDiligenceInput | ClinicalOutcomePredictionInput | ProtocolDesignInput | None,
    ) -> object:
        runner = _WORKFLOW_RUNNERS.get(workflow_name)
        input_type = _INPUT_TYPES.get(workflow_name)
        if runner is None or input_type is None:
            raise ValueError(f"Unknown workflow: {workflow_name}")
        if input_data is None:
            raise ValueError(f"{workflow_name} requires {input_type.__name__}")
        if not isinstance(input_data, input_type):
            raise ValueError(f"{workflow_name} requires {input_type.__name__}")
        return runner(input_data, memory=self.memory)
