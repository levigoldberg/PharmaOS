"""Simple HTML viewer for PharmaOS Scientific Memory runs."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pharma_os.memory import MemoryStore


def build_run_html(run_id: str, *, memory: MemoryStore | None = None) -> str:
    """Build a readable HTML view for one persisted run."""

    store = memory or MemoryStore()
    bundle = store.get_run_bundle(run_id)
    title = f"PharmaOS Run {run_id}"
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{escape(title)}</title>",
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.4;margin:24px;max-width:1200px}"
        "table{border-collapse:collapse;width:100%;margin:12px 0}"
        "th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}"
        "th{background:#f5f5f5}"
        "pre{background:#f7f7f7;border:1px solid #ddd;padding:12px;overflow:auto}"
        "details{margin:12px 0}"
        ".muted{color:#666}"
        "</style></head><body>",
        f"<h1>{escape(title)}</h1>",
    ]
    if bundle.run is None:
        parts.append("<p class='muted'>No persisted run was found.</p>")
        parts.append("</body></html>")
        return "\n".join(parts)

    run = bundle.run
    parts.extend(
        [
            "<h2>Run Metadata</h2>",
            _kv_table(
                {
                    "run_id": run.run_id,
                    "workflow_name": run.workflow_name,
                    "status": run.status,
                    "started_at": run.started_at,
                    "completed_at": run.completed_at,
                    "validation_status": run.validation_status,
                    "gate_reason": run.gate_reason,
                    "input_provenance": run.input_provenance,
                }
            ),
            _json_details("Input JSON", bundle.input_json),
            _json_details("Output JSON", bundle.output_json),
            _json_details("Trace Metadata", bundle.trace_metadata_json),
            _table_section("Agent Outputs", bundle.agent_outputs, ("output_id", "agent_name", "confidence", "validation_status", "gate_reason")),
            _table_section("Sources", bundle.sources, ("source_id", "title", "source_type", "provenance", "url")),
            _table_section("Claims", bundle.claims, ("claim_id", "claim_text", "source_ids", "confidence", "confidence_level")),
            _table_section("Validation Results", bundle.validation_results, ("validation_id", "target_id", "status", "validator", "message")),
            _table_section("Confidence Flags", bundle.confidence_flags, ("flag_id", "target_id", "severity", "reason", "confidence")),
            _table_section("Human Gates", bundle.human_gates, ("gate_id", "decision", "gate_reason", "required_roles", "reviewer")),
            _table_section("Agent Traces", bundle.agent_traces, ("trace_id", "agent_name", "output_id", "output_type", "confidence", "rationale_summary")),
            _json_details("Raw Bundle JSON", _bundle_json(bundle)),
            "</body></html>",
        ]
    )
    return "\n".join(parts)


def write_run_html(run_id: str, output_html: str | Path, *, memory: MemoryStore | None = None) -> Path:
    """Write a run HTML view and return its path."""

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_run_html(run_id, memory=memory), encoding="utf-8")
    return output_path


def _table_section(title: str, rows: tuple[Any, ...], fields: tuple[str, ...]) -> str:
    if not rows:
        return f"<h2>{escape(title)}</h2><p class='muted'>None.</p>"
    head = "".join(f"<th>{escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(_display(getattr(row, field, None)))}</td>" for field in fields) + "</tr>")
    return f"<h2>{escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _kv_table(values: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(_display(value))}</td></tr>"
        for key, value in values.items()
    )
    return f"<table><tbody>{rows}</tbody></table>"


def _json_details(title: str, value: Any) -> str:
    return (
        f"<details open><summary>{escape(title)}</summary>"
        f"<pre>{escape(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, default=str))}</pre>"
        "</details>"
    )


def _bundle_json(bundle: Any) -> dict[str, Any]:
    return {
        "run": _jsonable(bundle.run),
        "input_json": bundle.input_json,
        "output_json": bundle.output_json,
        "trace_metadata_json": bundle.trace_metadata_json,
        "agent_outputs": _jsonable(bundle.agent_outputs),
        "agent_traces": _jsonable(bundle.agent_traces),
        "sources": _jsonable(bundle.sources),
        "claims": _jsonable(bundle.claims),
        "validation_results": _jsonable(bundle.validation_results),
        "confidence_flags": _jsonable(bundle.confidence_flags),
        "human_gates": _jsonable(bundle.human_gates),
        "reports": _jsonable(bundle.reports),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return str(value)
