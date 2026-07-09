"""Workflow capability registry for planning and orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from pharma_os.schemas import ModuleCapability, WorkflowSpec


@dataclass(frozen=True)
class WorkflowRegistry:
    """Read-only registry of executable workflows and future capabilities."""

    _capabilities: dict[str, ModuleCapability]

    @classmethod
    def default(cls) -> "WorkflowRegistry":
        """Return the default PharmaOS capability registry."""

        capabilities: list[ModuleCapability] = [
            WorkflowSpec(
                name="clinical_outcome_prediction",
                workflow_name="clinical_outcome_prediction",
                lifecycle_stage="clinical_development",
                implementation_status="implemented",
                accepted_inputs=("ClinicalOutcomePredictionInput", "nct_id"),
                required_artifacts=("clinical_trial_record",),
                produced_artifacts=("clinical_outcome_prediction_output", "clinical_risk_context"),
                dependencies=(),
                executable=True,
                missing_connectors=(),
                human_gate_policy="Human review is required when validation flags high-risk clinical assumptions or missing evidence.",
                description="Agent 3 clinical outcome prediction and clinical risk context for one NCT.",
                input_schema="ClinicalOutcomePredictionInput",
                output_schema="ClinicalOutcomePredictionOutput",
                implementation_path="pharma_os.workflows.clinical_outcome_prediction.run_clinical_outcome_prediction_workflow",
            ),
            WorkflowSpec(
                name="due_diligence",
                workflow_name="due_diligence",
                lifecycle_stage="clinical_development",
                implementation_status="implemented",
                accepted_inputs=("DueDiligenceInput", "nct_id", "reviewed_commercial_assumptions"),
                required_artifacts=("clinical_outcome_prediction_output", "clinical_risk_context"),
                produced_artifacts=("due_diligence_output", "asset_memo", "commercial_model", "rnpv"),
                dependencies=("clinical_outcome_prediction",),
                executable=True,
                missing_connectors=(),
                human_gate_policy="Human review is mandatory for diligence conclusions, commercial assumptions, and investment-sensitive outputs.",
                description="Agent 4 clinical-stage due diligence using Agent 3 handoff plus safety, IP, pricing, commercial, and rNPV context.",
                input_schema="DueDiligenceInput",
                output_schema="DueDiligenceOutput",
                implementation_path="pharma_os.workflows.due_diligence.run_due_diligence_workflow",
            ),
            WorkflowSpec(
                name="protocol_design",
                workflow_name="protocol_design",
                lifecycle_stage="clinical_development",
                implementation_status="implemented",
                accepted_inputs=("ProtocolDesignInput", "nct_id", "reviewed_commercial_assumptions"),
                required_artifacts=("clinical_outcome_prediction_output", "due_diligence_output"),
                produced_artifacts=("protocol_design_output", "protocol_design_brief", "next_study_intent"),
                dependencies=("clinical_outcome_prediction", "due_diligence"),
                executable=True,
                missing_connectors=(),
                human_gate_policy="Human clinical, statistical, and regulatory review is mandatory before protocol use.",
                description="Agent 5 next-study protocol design planning using Agent 3/4 handoffs and deterministic analog benchmarking.",
                input_schema="ProtocolDesignInput",
                output_schema="ProtocolDesignOutput",
                implementation_path="pharma_os.workflows.protocol_design.run_protocol_design_workflow",
            ),
            ModuleCapability(
                name="discovery",
                lifecycle_stage="discovery",
                implementation_status="skeleton",
                accepted_inputs=("target", "disease", "asset_hypothesis"),
                required_artifacts=("target_hypothesis", "biology_evidence"),
                produced_artifacts=("discovery_prioritization",),
                dependencies=(),
                executable=False,
                missing_connectors=("target_knowledge_graph", "omics_evidence_store", "assay_inventory"),
                human_gate_policy="Scientific review required before target or asset nomination.",
                description="Architectural skeleton for target and discovery prioritization.",
            ),
            ModuleCapability(
                name="tox_pkpd_safety",
                lifecycle_stage="preclinical",
                implementation_status="skeleton",
                accepted_inputs=("asset_name", "species", "dose", "exposure"),
                required_artifacts=("preclinical_safety_package", "pkpd_package"),
                produced_artifacts=("tox_pkpd_safety_assessment",),
                dependencies=("discovery",),
                executable=False,
                missing_connectors=("nonclinical_study_repository", "pkpd_model_store", "toxicology_ontology"),
                human_gate_policy="Toxicologist and clinical pharmacology review required before dose or safety conclusions.",
                description="Architectural skeleton for tox, PK/PD, and translational safety planning.",
            ),
            ModuleCapability(
                name="enrollment_feasibility",
                lifecycle_stage="clinical_operations",
                implementation_status="skeleton",
                accepted_inputs=("nct_id", "indication", "population", "countries"),
                required_artifacts=("protocol_design_brief", "site_landscape", "patient_population_data"),
                produced_artifacts=("enrollment_feasibility_plan",),
                dependencies=("protocol_design",),
                executable=False,
                missing_connectors=("site_performance_database", "claims_or_registry_population_data", "country_startup_timelines"),
                human_gate_policy="Clinical operations review required before country, site, and enrollment commitments.",
                description="Architectural skeleton for enrollment feasibility and trial operations planning.",
            ),
            ModuleCapability(
                name="trial_execution",
                lifecycle_stage="clinical_operations",
                implementation_status="skeleton",
                accepted_inputs=("protocol_id", "sites", "enrollment_plan"),
                required_artifacts=("final_protocol", "enrollment_feasibility_plan", "study_startup_package"),
                produced_artifacts=("trial_execution_control_plan",),
                dependencies=("enrollment_feasibility",),
                executable=False,
                missing_connectors=("ctms", "edc", "site_activation_tracker", "risk_based_monitoring_system"),
                human_gate_policy="Clinical operations and quality review required before execution actions.",
                description="Architectural skeleton for live trial execution control.",
            ),
            ModuleCapability(
                name="manufacturing_biofactory",
                lifecycle_stage="manufacturing",
                implementation_status="skeleton",
                accepted_inputs=("asset_name", "process_stage", "demand_forecast"),
                required_artifacts=("cmc_package", "demand_forecast", "quality_release_data"),
                produced_artifacts=("manufacturing_control_plan",),
                dependencies=("due_diligence",),
                executable=False,
                missing_connectors=("mes", "lms", "batch_record_system", "supply_chain_planning"),
                human_gate_policy="CMC, quality, and supply review required before manufacturing decisions.",
                description="Architectural skeleton for manufacturing and biofactory orchestration.",
            ),
            ModuleCapability(
                name="launch_pv",
                lifecycle_stage="launch_postmarketing",
                implementation_status="skeleton",
                accepted_inputs=("asset_name", "label", "market"),
                required_artifacts=("approved_label", "launch_plan", "safety_management_plan"),
                produced_artifacts=("launch_pv_control_plan",),
                dependencies=("regulatory_quality_audit",),
                executable=False,
                missing_connectors=("pv_database", "medical_information_system", "commercial_launch_tracker"),
                human_gate_policy="Medical, safety, legal, and commercial review required before launch or PV actions.",
                description="Architectural skeleton for launch readiness and pharmacovigilance control.",
            ),
            ModuleCapability(
                name="regulatory_quality_audit",
                lifecycle_stage="quality_regulatory",
                implementation_status="skeleton",
                accepted_inputs=("submission_package", "quality_system_records", "protocol_or_report"),
                required_artifacts=("regulatory_strategy", "quality_evidence_package"),
                produced_artifacts=("regulatory_quality_audit_report",),
                dependencies=("protocol_design",),
                executable=False,
                missing_connectors=("regulatory_document_store", "qms", "submission_tracker"),
                human_gate_policy="Regulatory and quality review required for all audit findings.",
                description="Architectural skeleton for regulatory and quality audit planning.",
            ),
        ]
        return cls({capability.name: capability for capability in capabilities})

    def get(self, name: str) -> ModuleCapability | None:
        """Return a capability by name."""

        return self._capabilities.get(name)

    def require(self, name: str) -> ModuleCapability:
        """Return a capability or raise a clear error."""

        capability = self.get(name)
        if capability is None:
            raise KeyError(f"Unknown workflow capability: {name}")
        return capability

    def capabilities(self) -> tuple[ModuleCapability, ...]:
        """Return all registered capabilities."""

        return tuple(self._capabilities[name] for name in sorted(self._capabilities))

    def executable_workflows(self) -> tuple[WorkflowSpec, ...]:
        """Return implemented executable workflow specs."""

        return tuple(
            capability
            for capability in self.capabilities()
            if isinstance(capability, WorkflowSpec) and capability.executable
        )

    def names(self) -> tuple[str, ...]:
        """Return registered capability names."""

        return tuple(sorted(self._capabilities))
