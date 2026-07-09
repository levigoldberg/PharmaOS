"""SQLite-backed Scientific Memory for PharmaOS."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from pharma_os.registry import WorkflowRegistry
from pharma_os.schemas import (
    AgentOutput,
    AgentRunTrace,
    ArtifactStatus,
    ConfidenceFlag,
    EvidenceClaim,
    FinalReport,
    HumanGate,
    ModuleCapability,
    OrchestrationRequest,
    ScientificStateSnapshot,
    SourceMetadata,
    ValidationResult,
    WorkflowSpec,
    WorkflowRun,
)


DEFAULT_DB_PATH = ".pharma_os/scientific_memory.sqlite"


@dataclass(frozen=True)
class RunBundle:
    """All persisted artifacts for one workflow run."""

    run: WorkflowRun | None
    input_json: dict[str, Any] | list[Any] | None
    output_json: dict[str, Any] | list[Any] | None
    trace_metadata_json: dict[str, Any]
    sources: tuple[SourceMetadata, ...]
    claims: tuple[EvidenceClaim, ...]
    agent_outputs: tuple[AgentOutput, ...]
    agent_traces: tuple[AgentRunTrace, ...]
    validation_results: tuple[ValidationResult, ...]
    confidence_flags: tuple[ConfidenceFlag, ...]
    human_gates: tuple[HumanGate, ...]
    reports: tuple[FinalReport, ...]


class MemoryStore:
    """SQLite Scientific Memory store."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.db_path))
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()

    def save_run(
        self,
        run: WorkflowRun,
        *,
        input_payload: BaseModel | dict[str, Any] | None = None,
        output_payload: BaseModel | dict[str, Any] | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> WorkflowRun:
        """Persist a workflow run."""

        self._connection.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, workflow_name, status, started_at, completed_at, input_provenance,
                source_ids_json, validation_status, gate_reason, metadata_json,
                input_json, output_json, trace_metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.workflow_name,
                run.status,
                _dt(run.started_at),
                _dt(run.completed_at),
                run.input_provenance,
                _json(run.source_ids),
                run.validation_status,
                run.gate_reason,
                _json(run.metadata),
                _json_model(input_payload),
                _json_model(output_payload),
                _json(trace_metadata or {}),
            ),
        )
        self._connection.commit()
        return run

    def get_run(self, run_id: str) -> WorkflowRun | None:
        """Return a workflow run by ID."""

        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return WorkflowRun(
            run_id=row["run_id"],
            workflow_name=row["workflow_name"],
            status=row["status"],
            started_at=_parse_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
            input_provenance=row["input_provenance"],
            source_ids=tuple(json.loads(row["source_ids_json"] or "[]")),
            validation_status=row["validation_status"],
            gate_reason=row["gate_reason"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def get_latest_workflow_output(
        self,
        *,
        workflow_name: str,
        nct_id: str,
    ) -> tuple[WorkflowRun, dict[str, Any]] | None:
        """Return the latest completed non-failed workflow output for an NCT ID."""

        rows = self._connection.execute(
            """
            SELECT * FROM runs
            WHERE workflow_name = ?
              AND status = 'completed'
              AND validation_status != 'failed'
              AND output_json IS NOT NULL
            ORDER BY COALESCE(completed_at, started_at) DESC, started_at DESC
            """,
            (workflow_name,),
        ).fetchall()
        normalized_nct = nct_id.strip().upper()
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}") or {}
            input_payload = json.loads(row["input_json"] or "{}") or {}
            output_payload = json.loads(row["output_json"] or "{}") or {}
            candidate_nct = (
                metadata.get("nct_id")
                or input_payload.get("nct_id")
                or (output_payload.get("input") or {}).get("nct_id")
            )
            if str(candidate_nct or "").strip().upper() != normalized_nct:
                continue
            return (
                WorkflowRun(
                    run_id=row["run_id"],
                    workflow_name=row["workflow_name"],
                    status=row["status"],
                    started_at=_parse_dt(row["started_at"]),
                    completed_at=_parse_dt(row["completed_at"]),
                    input_provenance=row["input_provenance"],
                    source_ids=tuple(json.loads(row["source_ids_json"] or "[]")),
                    validation_status=row["validation_status"],
                    gate_reason=row["gate_reason"],
                    metadata=metadata,
                ),
                output_payload,
            )
        return None

    def build_scientific_state_snapshot(
        self,
        request: OrchestrationRequest,
        *,
        registry: WorkflowRegistry | None = None,
    ) -> ScientificStateSnapshot:
        """Build a memory-derived state snapshot for Control Tower planning."""

        effective_registry = registry or WorkflowRegistry.default()
        capabilities = effective_registry.capabilities()
        artifacts = tuple(
            artifact
            for capability in capabilities
            if isinstance(capability, WorkflowSpec)
            for artifact in self.assess_artifact_reuse(request=request, capability=capability)
        )
        present_artifact_types = {artifact.artifact_type for artifact in artifacts}
        required_artifacts = {
            artifact_type
            for capability in capabilities
            for artifact_type in capability.required_artifacts
            if not _artifact_satisfied_by_request(artifact_type, request)
        }
        missing_artifacts = tuple(sorted(required_artifacts - present_artifact_types))
        open_gates = tuple(
            gate
            for artifact in artifacts
            for gate in artifact.open_gates
            if gate.decision in {"needs_human_review", "blocked", "rejected"}
        )
        return ScientificStateSnapshot(
            snapshot_id=f"snapshot-{uuid4()}",
            request=request,
            artifacts=artifacts,
            capabilities=capabilities,
            open_gates=open_gates,
            missing_artifacts=missing_artifacts,
            notes=tuple(
                dict.fromkeys(
                    reason
                    for artifact in artifacts
                    for reason in artifact.reasons
                    if artifact.compatibility != "compatible"
                )
            ),
        )

    def assess_artifact_reuse(
        self,
        *,
        request: OrchestrationRequest,
        capability: WorkflowSpec,
    ) -> tuple[ArtifactStatus, ...]:
        """Assess reusable artifacts for one workflow capability."""

        rows = self._candidate_run_rows(workflow_name=capability.workflow_name, request=request)
        artifacts: list[ArtifactStatus] = []
        for row in rows:
            input_payload = _json_loads(row["input_json"]) or {}
            output_payload = _json_loads(row["output_json"]) or {}
            completed_at = _parse_dt(row["completed_at"]) or _parse_dt(row["started_at"])
            run = WorkflowRun(
                run_id=row["run_id"],
                workflow_name=row["workflow_name"],
                status=row["status"],
                started_at=_parse_dt(row["started_at"]) or datetime.now(timezone.utc),
                completed_at=completed_at,
                input_provenance=row["input_provenance"],
                source_ids=tuple(json.loads(row["source_ids_json"] or "[]")),
                validation_status=row["validation_status"],
                gate_reason=row["gate_reason"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            gates = self._human_gates_for_run(run.run_id)
            freshness = self._run_freshness(run.run_id, completed_at=completed_at)
            compatibility, reasons = _artifact_compatibility(
                request=request,
                capability=capability,
                run=run,
                input_payload=input_payload,
                output_payload=output_payload,
                gates=gates,
                freshness=freshness,
            )
            for artifact_type in capability.produced_artifacts:
                artifacts.append(
                    ArtifactStatus(
                        artifact_type=artifact_type,
                        producer_workflow=capability.workflow_name,
                        run_id=run.run_id,
                        output_id=_extract_output_id(output_payload),
                        validation_status=run.validation_status,
                        confidence=_extract_confidence(output_payload),
                        freshness=freshness,
                        compatibility=compatibility,
                        open_gates=gates,
                        upstream_references=_extract_upstream_references(output_payload),
                        input_fingerprint=_input_fingerprint(input_payload),
                        completed_at=completed_at,
                        reasons=reasons,
                    )
                )
        return tuple(artifacts)

    def save_sources(self, run_id: str, sources: tuple[SourceMetadata, ...]) -> None:
        """Persist source metadata for a run."""

        self._connection.executemany(
            """
            INSERT OR REPLACE INTO sources (
                source_id, run_id, title, url, authors_json, published_at,
                retrieved_at, provenance, source_type, version, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    source.source_id,
                    run_id,
                    source.title,
                    str(source.url) if source.url else None,
                    _json(source.authors),
                    _dt(source.published_at),
                    _dt(source.retrieved_at),
                    source.provenance,
                    source.source_type,
                    source.version,
                    _json_model(source),
                )
                for source in sources
            ],
        )
        self._connection.commit()

    def save_claims(self, run_id: str, claims: tuple[EvidenceClaim, ...]) -> None:
        """Persist evidence claims for a run."""

        self._connection.executemany(
            """
            INSERT OR REPLACE INTO claims (
                claim_id, run_id, claim_text, source_ids_json, provenance,
                confidence, confidence_level, qualifiers_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    claim.claim_id,
                    run_id,
                    claim.claim_text,
                    _json(claim.source_ids),
                    claim.provenance,
                    claim.confidence,
                    claim.confidence_level,
                    _json(claim.qualifiers),
                )
                for claim in claims
            ],
        )
        self._connection.commit()

    def save_agent_output(self, output: AgentOutput, payload: BaseModel | dict[str, Any] | None = None) -> None:
        """Persist an agent output envelope."""

        self._connection.execute(
            """
            INSERT OR REPLACE INTO agent_outputs (
                output_id, run_id, agent_name, provenance, claims_json, sources_json,
                confidence, validation_status, gate_reason, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                output.output_id,
                output.run_id,
                output.agent_name,
                output.provenance,
                _json_model(output.claims),
                _json_model(output.sources),
                output.confidence,
                output.validation_status,
                output.gate_reason,
                _json_model(payload or output),
            ),
        )
        self._connection.commit()

    def save_agent_trace(self, trace: AgentRunTrace) -> None:
        """Persist a safe, user-readable agent run trace."""

        self._connection.execute(
            """
            INSERT OR REPLACE INTO agent_traces (
                trace_id, run_id, agent_name, input_summary, output_id, output_type,
                output_summary, source_ids_json, confidence, rationale_summary,
                started_at, completed_at, provenance, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                trace.run_id,
                trace.agent_name,
                trace.input_summary,
                trace.output_id,
                trace.output_type,
                trace.output_summary,
                _json(trace.source_ids),
                trace.confidence,
                trace.rationale_summary,
                _dt(trace.started_at),
                _dt(trace.completed_at),
                trace.provenance,
                _json_model(trace),
            ),
        )
        self._connection.commit()

    def save_agent_traces(self, traces: tuple[AgentRunTrace, ...]) -> None:
        """Persist multiple safe agent traces."""

        for trace in traces:
            self.save_agent_trace(trace)

    def save_validation_results(
        self, run_id: str, results: tuple[ValidationResult, ...]
    ) -> None:
        """Persist validation results for a run."""

        self._connection.executemany(
            """
            INSERT OR REPLACE INTO validation_results (
                validation_id, run_id, target_id, status, validator, message,
                confidence, source_ids_json, gate_reason, provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    result.validation_id,
                    run_id,
                    result.target_id,
                    result.status,
                    result.validator,
                    result.message,
                    result.confidence,
                    _json(result.source_ids),
                    result.gate_reason,
                    result.provenance,
                )
                for result in results
            ],
        )
        self._connection.commit()

    def save_confidence_flags(self, run_id: str, flags: tuple[ConfidenceFlag, ...]) -> None:
        """Persist confidence flags for a run."""

        self._connection.executemany(
            """
            INSERT OR REPLACE INTO confidence_flags (
                flag_id, run_id, target_id, reason, severity, confidence,
                source_ids_json, provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    flag.flag_id,
                    run_id,
                    flag.target_id,
                    flag.reason,
                    flag.severity,
                    flag.confidence,
                    _json(flag.source_ids),
                    flag.provenance,
                )
                for flag in flags
            ],
        )
        self._connection.commit()

    def save_human_gate(self, run_id: str, gate: HumanGate | None) -> None:
        """Persist a human gate when one is required."""

        if gate is None:
            return
        self._connection.execute(
            """
            INSERT OR REPLACE INTO human_gates (
                gate_id, run_id, decision, gate_reason, required_roles_json,
                reviewer, reviewed_at, source_ids_json, provenance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gate.gate_id,
                run_id,
                gate.decision,
                gate.gate_reason,
                _json(gate.required_roles),
                gate.reviewer,
                _dt(gate.reviewed_at),
                _json(gate.source_ids),
                gate.provenance,
            ),
        )
        self._connection.commit()

    def save_report(self, report: FinalReport) -> FinalReport:
        """Persist a final report."""

        self._connection.execute(
            """
            INSERT OR REPLACE INTO reports (
                report_id, run_id, title, summary, claims_json, sources_json,
                validation_results_json, confidence_flags_json, human_gate_json,
                confidence, validation_status, provenance, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.report_id,
                report.run_id,
                report.title,
                report.summary,
                _json_model(report.claims),
                _json_model(report.sources),
                _json_model(report.validation_results),
                _json_model(report.confidence_flags),
                _json_model(report.human_gate),
                report.confidence,
                report.validation_status,
                report.provenance,
                _json_model(report),
            ),
        )
        self._connection.commit()
        return report

    def get_run_bundle(self, run_id: str) -> RunBundle:
        """Return all persisted objects for a run."""

        run_row = self._connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return RunBundle(
            run=self.get_run(run_id),
            input_json=_json_loads(run_row["input_json"]) if run_row is not None else None,
            output_json=_json_loads(run_row["output_json"]) if run_row is not None else None,
            trace_metadata_json=_json_loads(run_row["trace_metadata_json"]) if run_row is not None else {},
            sources=tuple(
                SourceMetadata.model_validate_json(row["payload_json"])
                for row in self._rows("sources", run_id)
            ),
            claims=tuple(
                EvidenceClaim(
                    claim_id=row["claim_id"],
                    claim_text=row["claim_text"],
                    source_ids=tuple(json.loads(row["source_ids_json"] or "[]")),
                    provenance=row["provenance"],
                    confidence=row["confidence"],
                    confidence_level=row["confidence_level"],
                    qualifiers=tuple(json.loads(row["qualifiers_json"] or "[]")),
                )
                for row in self._rows("claims", run_id)
            ),
            agent_outputs=tuple(
                AgentOutput(
                    output_id=row["output_id"],
                    run_id=row["run_id"],
                    agent_name=row["agent_name"],
                    provenance=row["provenance"],
                    claims=tuple(
                        EvidenceClaim.model_validate_json(json.dumps(item))
                        for item in json.loads(row["claims_json"] or "[]")
                    ),
                    sources=tuple(
                        SourceMetadata.model_validate_json(json.dumps(item))
                        for item in json.loads(row["sources_json"] or "[]")
                    ),
                    confidence=row["confidence"],
                    validation_status=row["validation_status"],
                    gate_reason=row["gate_reason"],
                )
                for row in self._rows("agent_outputs", run_id)
            ),
            agent_traces=tuple(
                AgentRunTrace.model_validate_json(row["payload_json"])
                for row in self._rows("agent_traces", run_id)
            ),
            validation_results=tuple(
                ValidationResult(
                    validation_id=row["validation_id"],
                    target_id=row["target_id"],
                    status=row["status"],
                    validator=row["validator"],
                    message=row["message"],
                    confidence=row["confidence"],
                    source_ids=tuple(json.loads(row["source_ids_json"] or "[]")),
                    gate_reason=row["gate_reason"],
                    provenance=row["provenance"],
                )
                for row in self._rows("validation_results", run_id)
            ),
            confidence_flags=tuple(
                ConfidenceFlag(
                    flag_id=row["flag_id"],
                    target_id=row["target_id"],
                    reason=row["reason"],
                    severity=row["severity"],
                    confidence=row["confidence"],
                    source_ids=tuple(json.loads(row["source_ids_json"] or "[]")),
                    provenance=row["provenance"],
                )
                for row in self._rows("confidence_flags", run_id)
            ),
            human_gates=tuple(
                HumanGate(
                    gate_id=row["gate_id"],
                    decision=row["decision"],
                    gate_reason=row["gate_reason"],
                    required_roles=tuple(json.loads(row["required_roles_json"] or "[]")),
                    reviewer=row["reviewer"],
                    reviewed_at=_parse_dt(row["reviewed_at"]),
                    source_ids=tuple(json.loads(row["source_ids_json"] or "[]")),
                    provenance=row["provenance"],
                )
                for row in self._rows("human_gates", run_id)
            ),
            reports=tuple(
                FinalReport.model_validate_json(row["payload_json"])
                for row in self._rows("reports", run_id)
            ),
        )

    def _rows(self, table: str, run_id: str) -> list[sqlite3.Row]:
        return list(self._connection.execute(f"SELECT * FROM {table} WHERE run_id = ?", (run_id,)))

    def _candidate_run_rows(self, *, workflow_name: str, request: OrchestrationRequest) -> list[sqlite3.Row]:
        rows = self._connection.execute(
            """
            SELECT * FROM runs
            WHERE workflow_name = ?
              AND output_json IS NOT NULL
            ORDER BY COALESCE(completed_at, started_at) DESC, started_at DESC
            """,
            (workflow_name,),
        ).fetchall()
        if not request.nct_id:
            return list(rows)
        normalized_nct = request.nct_id.strip().upper()
        matches = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}") or {}
            input_payload = _json_loads(row["input_json"]) or {}
            output_payload = _json_loads(row["output_json"]) or {}
            if _payload_nct(metadata, input_payload, output_payload) == normalized_nct:
                matches.append(row)
        return matches

    def _human_gates_for_run(self, run_id: str) -> tuple[HumanGate, ...]:
        return tuple(
            HumanGate(
                gate_id=row["gate_id"],
                decision=row["decision"],
                gate_reason=row["gate_reason"],
                required_roles=tuple(json.loads(row["required_roles_json"] or "[]")),
                reviewer=row["reviewer"],
                reviewed_at=_parse_dt(row["reviewed_at"]),
                source_ids=tuple(json.loads(row["source_ids_json"] or "[]")),
                provenance=row["provenance"],
            )
            for row in self._rows("human_gates", run_id)
        )

    def _run_freshness(self, run_id: str, *, completed_at: datetime | None) -> str:
        source_rows = self._rows("sources", run_id)
        reference_dates = [
            _parse_dt(row["retrieved_at"])
            for row in source_rows
            if _parse_dt(row["retrieved_at"]) is not None
        ]
        if not reference_dates and completed_at is not None:
            reference_dates = [completed_at]
        if not reference_dates:
            return "unknown"
        newest = max(reference_dates)
        if newest.tzinfo is None:
            newest = newest.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - newest).days
        return "stale" if age_days > 180 else "fresh"

    def _init_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                workflow_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                input_provenance TEXT NOT NULL,
                source_ids_json TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                gate_reason TEXT,
                metadata_json TEXT NOT NULL,
                input_json TEXT,
                output_json TEXT,
                trace_metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                source_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                title TEXT,
                url TEXT,
                authors_json TEXT NOT NULL,
                published_at TEXT,
                retrieved_at TEXT NOT NULL,
                provenance TEXT NOT NULL,
                source_type TEXT,
                version TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                source_ids_json TEXT NOT NULL,
                provenance TEXT NOT NULL,
                confidence REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                qualifiers_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_outputs (
                output_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                provenance TEXT NOT NULL,
                claims_json TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                validation_status TEXT NOT NULL,
                gate_reason TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_traces (
                trace_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                input_summary TEXT,
                output_id TEXT,
                output_type TEXT,
                output_summary TEXT,
                source_ids_json TEXT NOT NULL,
                confidence REAL,
                rationale_summary TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                provenance TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS validation_results (
                validation_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL,
                validator TEXT NOT NULL,
                message TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_ids_json TEXT NOT NULL,
                gate_reason TEXT,
                provenance TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS confidence_flags (
                flag_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL,
                source_ids_json TEXT NOT NULL,
                provenance TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS human_gates (
                gate_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                gate_reason TEXT NOT NULL,
                required_roles_json TEXT NOT NULL,
                reviewer TEXT,
                reviewed_at TEXT,
                source_ids_json TEXT NOT NULL,
                provenance TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                claims_json TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                validation_results_json TEXT NOT NULL,
                confidence_flags_json TEXT NOT NULL,
                human_gate_json TEXT,
                confidence REAL NOT NULL,
                validation_status TEXT NOT NULL,
                provenance TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            """
        )
        self._connection.commit()


def _dt(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _parse_dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: str | None) -> Any:
    if not value:
        return None
    return json.loads(value)


def _json_model(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, tuple):
        return json.dumps([_to_jsonable(item) for item in value], ensure_ascii=False)
    if isinstance(value, list):
        return json.dumps([_to_jsonable(item) for item in value], ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(_to_jsonable(value), ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _payload_nct(metadata: dict[str, Any], input_payload: Any, output_payload: Any) -> str | None:
    candidate = None
    if isinstance(metadata, dict):
        candidate = metadata.get("nct_id")
    if candidate is None and isinstance(input_payload, dict):
        candidate = input_payload.get("nct_id") or (input_payload.get("cli_input") or {}).get("nct_id")
    if candidate is None and isinstance(output_payload, dict):
        candidate = (
            (output_payload.get("input") or {}).get("nct_id")
            or (output_payload.get("target_trial") or {}).get("nct_id")
            or (output_payload.get("trial_identity") or {}).get("nct_id")
        )
    if candidate is None:
        return None
    return str(candidate).strip().upper()


def _artifact_compatibility(
    *,
    request: OrchestrationRequest,
    capability: WorkflowSpec,
    run: WorkflowRun,
    input_payload: Any,
    output_payload: Any,
    gates: tuple[HumanGate, ...],
    freshness: str,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    forced = {item.strip() for item in request.force_refresh}
    if capability.name in forced or capability.workflow_name in forced or set(capability.produced_artifacts) & forced:
        reasons.append("request explicitly forced refresh for this capability or artifact")
    if run.status != "completed":
        reasons.append(f"run status is {run.status}, not completed")
    if run.validation_status == "failed":
        reasons.append("run validation failed")
    if request.nct_id:
        observed_nct = _payload_nct(run.metadata, input_payload, output_payload)
        if observed_nct != request.nct_id.strip().upper():
            reasons.append("artifact NCT identifier does not match request")
    assumption_mismatches = _assumption_mismatches(request.assumptions, input_payload)
    reasons.extend(assumption_mismatches)
    blocked_gates = [gate for gate in gates if gate.decision in {"blocked", "rejected"}]
    if blocked_gates:
        reasons.append("artifact has blocking or rejected human gate")
    if freshness == "stale":
        reasons.append("artifact sources or run timestamp are stale")
    if not output_payload:
        reasons.append("artifact output payload is missing")

    if reasons:
        return "incompatible", tuple(dict.fromkeys(reasons))
    if gates:
        return "compatible", ("artifact carries open human-review gates",)
    return "compatible", ()


def _assumption_mismatches(request_assumptions: dict[str, Any], input_payload: Any) -> tuple[str, ...]:
    if not request_assumptions:
        return ()
    observed = _flatten_input_values(input_payload)
    mismatches = []
    for key, value in request_assumptions.items():
        observed_value = observed.get(key)
        if observed_value is None:
            mismatches.append(f"requested assumption {key} is absent from artifact input")
            continue
        if observed_value != value:
            mismatches.append(f"requested assumption {key} differs from artifact input")
    return tuple(mismatches)


def _flatten_input_values(input_payload: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not isinstance(input_payload, dict):
        return values
    candidates = [input_payload]
    for key in ("cli_input", "expanded_pipeline_input", "input"):
        item = input_payload.get(key)
        if isinstance(item, dict):
            candidates.append(item)
    for candidate in candidates:
        for key, value in candidate.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[key] = value
    return values


def _extract_output_id(output_payload: Any) -> str | None:
    if not isinstance(output_payload, dict):
        return None
    return str(output_payload.get("output_id") or output_payload.get("brief_id") or "") or None


def _extract_confidence(output_payload: Any) -> float | None:
    if not isinstance(output_payload, dict):
        return None
    value = output_payload.get("confidence")
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _extract_upstream_references(output_payload: Any) -> tuple[str, ...]:
    if not isinstance(output_payload, dict):
        return ()
    references = []
    for key in ("agent3_handoff", "agent4_handoff"):
        handoff = output_payload.get(key)
        if isinstance(handoff, dict):
            references.extend(str(value) for ref_key, value in handoff.items() if ref_key.endswith("_output_id") and value)
            references.extend(str(value) for ref_key, value in handoff.items() if ref_key.endswith("_run_id") and value)
    return tuple(dict.fromkeys(references))


def _input_fingerprint(input_payload: Any) -> str:
    return json.dumps(_to_jsonable(input_payload), ensure_ascii=False, sort_keys=True)


def _artifact_satisfied_by_request(artifact_type: str, request: OrchestrationRequest) -> bool:
    if artifact_type == "clinical_trial_record" and request.nct_id:
        return True
    identifiers = set(request.identifiers)
    if artifact_type in identifiers:
        return True
    return False
