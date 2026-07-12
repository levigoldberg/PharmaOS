"""Canonical human-review flag normalization.

The workflow schemas expose several lower-level uncertainty signals.  This
module projects them into one small, ranked `ConfidenceFlag` surface for memory,
reports, and human-readable outputs.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from pharma_os.schemas import ConfidenceFlag, HumanGate, ValidationResult


REVIEW_SEVERITIES = ("critical", "high", "medium")
VISIBLE_REVIEW_FLAG_LIMIT = 5
_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def canonical_review_flags(
    *,
    run_id: str,
    validation_results: Iterable[ValidationResult] = (),
    risk_flags: Iterable[Any] = (),
    human_gate: HumanGate | None = None,
    confidence_flags: Iterable[ConfidenceFlag] = (),
    limit: int | None = None,
) -> tuple[ConfidenceFlag, ...]:
    """Return deduped, ranked flags that genuinely need human attention."""

    flags: list[ConfidenceFlag] = [*_flags_from_validations(run_id, validation_results)]
    flags.extend(_flags_from_risks(run_id, risk_flags))
    flags.extend(_flags_from_existing(confidence_flags))
    if human_gate is not None and human_gate.decision in {"needs_human_review", "blocked"}:
        flags.append(
            ConfidenceFlag(
                flag_id=f"flag-{run_id}-human-gate",
                target_id=human_gate.gate_id,
                reason=human_gate.gate_reason,
                severity="high" if human_gate.decision == "blocked" else "medium",
                confidence=0.6 if human_gate.decision == "blocked" else 0.75,
                source_ids=human_gate.source_ids,
                provenance="pharma_os.validators.generate_confidence_flags.human_gate",
            )
        )

    visible = [flag for flag in flags if flag.severity in REVIEW_SEVERITIES]
    deduped: dict[str, ConfidenceFlag] = {}
    for flag in sorted(visible, key=lambda item: (_RANK.get(item.severity, 9), item.reason.casefold())):
        key = _fingerprint(flag.reason)
        current = deduped.get(key)
        if current is None or _RANK.get(flag.severity, 9) < _RANK.get(current.severity, 9):
            deduped[key] = flag
    ranked = tuple(sorted(deduped.values(), key=lambda item: (_RANK.get(item.severity, 9), item.reason.casefold())))
    if limit is not None:
        return ranked[:limit]
    return ranked


def top_review_flags_from_payload(payload: dict[str, Any], *, limit: int = VISIBLE_REVIEW_FLAG_LIMIT) -> tuple[dict[str, Any], ...]:
    """Return canonical visible review flags from a serialized workflow payload."""

    raw_flags: list[dict[str, Any]] = []
    raw_flags.extend(_payload_flags(payload.get("confidence_flags"), default_provenance="payload.confidence_flags"))
    raw_flags.extend(_payload_flags(payload.get("red_flags"), default_provenance="payload.red_flags"))
    raw_flags.extend(_payload_flags(payload.get("missing_data_flags"), default_provenance="payload.missing_data_flags"))

    gate = payload.get("human_gate")
    if isinstance(gate, dict) and gate.get("decision") in {"needs_human_review", "blocked"} and gate.get("gate_reason"):
        raw_flags.append(
            {
                "flag_id": gate.get("gate_id") or "human-gate",
                "target_id": gate.get("gate_id") or "human-gate",
                "severity": "high" if gate.get("decision") == "blocked" else "medium",
                "confidence": 0.6 if gate.get("decision") == "blocked" else 0.75,
                "reason": gate.get("gate_reason"),
                "source_ids": tuple(gate.get("source_ids") or ()),
                "provenance": gate.get("provenance") or "payload.human_gate",
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for flag in sorted(raw_flags, key=lambda item: (_RANK.get(str(item.get("severity")), 9), str(item.get("reason") or "").casefold())):
        severity = str(flag.get("severity") or "medium")
        reason = str(flag.get("reason") or "").strip()
        if severity not in REVIEW_SEVERITIES or not reason:
            continue
        key = _fingerprint(reason)
        current = deduped.get(key)
        if current is None or _RANK.get(severity, 9) < _RANK.get(str(current.get("severity")), 9):
            deduped[key] = {**flag, "severity": severity, "reason": reason, "confidence": flag.get("confidence") or _confidence_for_severity(severity)}
    return tuple(sorted(deduped.values(), key=lambda item: (_RANK.get(str(item.get("severity")), 9), str(item.get("reason")).casefold())))[:limit]


def review_flag_summary(flags: Iterable[ConfidenceFlag]) -> str:
    """Compact human-facing summary of canonical review flags."""

    items = tuple(flags)
    if not items:
        return "No human-review flags are open."
    counts = {severity: sum(1 for flag in items if flag.severity == severity) for severity in REVIEW_SEVERITIES}
    pieces = [f"{count} {severity}" for severity, count in counts.items() if count]
    return f"{len(items)} human-review flags are open ({', '.join(pieces)})."


def _flags_from_validations(run_id: str, validation_results: Iterable[ValidationResult]) -> list[ConfidenceFlag]:
    flags: list[ConfidenceFlag] = []
    for result in validation_results:
        if result.status not in {"failed", "warning", "needs_human_review"}:
            continue
        reason = result.gate_reason or result.message
        severity = "high" if result.status == "failed" else "medium"
        flags.append(
            ConfidenceFlag(
                flag_id=f"flag-{run_id}-{_slug(result.validation_id)}",
                target_id=result.target_id,
                reason=reason,
                severity=severity,
                confidence=max(0.0, min(result.confidence, 1.0)),
                source_ids=result.source_ids,
                provenance="pharma_os.validators.generate_confidence_flags.validation",
            )
        )
    return flags


def _flags_from_risks(run_id: str, risk_flags: Iterable[Any]) -> list[ConfidenceFlag]:
    flags: list[ConfidenceFlag] = []
    for index, risk in enumerate(risk_flags, start=1):
        severity = str(getattr(risk, "severity", "medium"))
        if severity not in REVIEW_SEVERITIES:
            continue
        reason = getattr(risk, "description", None) or getattr(risk, "reason", str(risk))
        source_ids = tuple(getattr(risk, "source_ids", ()))
        flags.append(
            ConfidenceFlag(
                flag_id=f"flag-{run_id}-risk-{index}",
                target_id=getattr(risk, "risk_id", None) or getattr(risk, "flag_id", f"risk-{index}"),
                reason=str(reason),
                severity=severity,
                confidence=0.4 if severity == "critical" else 0.65 if severity == "high" else 0.8,
                source_ids=source_ids,
                provenance="pharma_os.validators.generate_confidence_flags.risk",
            )
        )
    return flags


def _flags_from_existing(confidence_flags: Iterable[ConfidenceFlag]) -> list[ConfidenceFlag]:
    return [flag for flag in confidence_flags if flag.severity in REVIEW_SEVERITIES]


def _payload_flags(value: Any, *, default_provenance: str) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    if not isinstance(value, (list, tuple)):
        return flags
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        reason = item.get("reason") or item.get("description") or item.get("message")
        if not reason:
            continue
        flags.append(
            {
                "flag_id": item.get("flag_id") or f"payload-flag-{index}",
                "target_id": item.get("target_id") or item.get("risk_id") or item.get("section") or item.get("category") or "workflow-output",
                "severity": item.get("severity") or "medium",
                "confidence": item.get("confidence") or _confidence_for_severity(str(item.get("severity") or "medium")),
                "category": item.get("category") or item.get("section"),
                "reason": reason,
                "source_ids": tuple(item.get("source_ids") or ()),
                "provenance": item.get("provenance") or default_provenance,
            }
        )
    return flags


def _fingerprint(reason: str) -> str:
    text = reason.casefold()
    text = re.sub(r"\bnct\d{8}\b", "nct", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())[:220]


def _confidence_for_severity(severity: str) -> float:
    if severity == "critical":
        return 0.4
    if severity == "high":
        return 0.65
    if severity == "medium":
        return 0.8
    return 0.9


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")[:80] or "target"
