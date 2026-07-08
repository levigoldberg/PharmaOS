"""SQLite-backed Scientific Memory for PharmaOS."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pharma_os.schemas import (
    AgentOutput,
    ConfidenceFlag,
    EvidenceClaim,
    FinalReport,
    HumanGate,
    SourceMetadata,
    ValidationResult,
    WorkflowRun,
)


DEFAULT_DB_PATH = ".pharma_os/scientific_memory.sqlite"


@dataclass(frozen=True)
class RunBundle:
    """All persisted artifacts for one workflow run."""

    run: WorkflowRun | None
    sources: tuple[SourceMetadata, ...]
    claims: tuple[EvidenceClaim, ...]
    agent_outputs: tuple[AgentOutput, ...]
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

        return RunBundle(
            run=self.get_run(run_id),
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
