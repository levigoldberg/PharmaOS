"""Workflow orchestration entry points."""

from __future__ import annotations

from pharma_os.memory import MemoryStore
from pharma_os.schemas import ClinicalOutcomePredictionInput, ClinicalTrialIntelligenceInput, DueDiligenceInput, ProtocolDesignInput
from pharma_os.validators import validate_workflow_name
from pharma_os.workflows.clinical_outcome_prediction import run_clinical_outcome_prediction_workflow
from pharma_os.workflows.due_diligence import run_due_diligence_workflow
from pharma_os.workflows.protocol_design import run_protocol_design_workflow
from pharma_os.workflows.trial_intelligence import run_trial_intelligence_workflow


class Orchestrator:
    """Coordinates workflow runs."""

    def __init__(self, memory: MemoryStore | None = None) -> None:
        self.memory = memory or MemoryStore()

    def run(
        self,
        workflow: str,
        input_data: ClinicalTrialIntelligenceInput | DueDiligenceInput | ClinicalOutcomePredictionInput | ProtocolDesignInput | None = None,
    ) -> object:
        """Run a workflow by name."""

        workflow_name = validate_workflow_name(workflow)
        if workflow_name == "trial_intelligence":
            if input_data is None:
                raise ValueError("trial_intelligence requires ClinicalTrialIntelligenceInput")
            if not isinstance(input_data, ClinicalTrialIntelligenceInput):
                raise ValueError("trial_intelligence requires ClinicalTrialIntelligenceInput")
            return run_trial_intelligence_workflow(input_data, memory=self.memory)
        if workflow_name == "due_diligence":
            if input_data is None:
                raise ValueError("due_diligence requires DueDiligenceInput")
            if not isinstance(input_data, DueDiligenceInput):
                raise ValueError("due_diligence requires DueDiligenceInput")
            return run_due_diligence_workflow(input_data, memory=self.memory)
        if workflow_name == "clinical_outcome_prediction":
            if input_data is None:
                raise ValueError("clinical_outcome_prediction requires ClinicalOutcomePredictionInput")
            if not isinstance(input_data, ClinicalOutcomePredictionInput):
                raise ValueError("clinical_outcome_prediction requires ClinicalOutcomePredictionInput")
            return run_clinical_outcome_prediction_workflow(input_data, memory=self.memory)
        if workflow_name == "protocol_design":
            if input_data is None:
                raise ValueError("protocol_design requires ProtocolDesignInput")
            if not isinstance(input_data, ProtocolDesignInput):
                raise ValueError("protocol_design requires ProtocolDesignInput")
            return run_protocol_design_workflow(input_data, memory=self.memory)
        raise ValueError(f"Unknown workflow: {workflow_name}")
