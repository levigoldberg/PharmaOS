"""Human-readable HTML viewer for PharmaOS Scientific Memory runs."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pharma_os.due_diligence_report import build_due_diligence_report_payload
from pharma_os.memory import DEFAULT_DB_PATH, MemoryStore
from pharma_os.review_flags import top_review_flags_from_payload


def build_run_html(run_id: str, *, memory: MemoryStore | None = None) -> str:
    """Build a readable HTML view for one persisted run."""

    store = memory or MemoryStore()
    bundle = store.get_run_bundle(run_id)
    title = f"PharmaOS Run {run_id}"
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{escape(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        "<main>",
        f"<h1>{escape(title)}</h1>",
    ]
    if bundle.run is None:
        parts.append("<p class='muted'>No persisted run was found.</p>")
        parts.append("</main></body></html>")
        return "\n".join(parts)

    run = bundle.run
    parts.extend(
        [
            _section(
                "Run Metadata",
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
            ),
            _workflow_report_section(bundle.output_json, run.workflow_name),
            _execution_mode_summary_section(bundle),
            _human_readable_summary_section(bundle.output_json),
            _section("Source-Backed Claims", _claims_cards(bundle.claims)),
            _json_details("Input JSON", bundle.input_json),
            _json_details("Output JSON", bundle.output_json),
            _json_details("Trace Metadata", bundle.trace_metadata_json),
            _table_section("Agent Outputs", bundle.agent_outputs, ("output_id", "agent_name", "execution_mode", "confidence", "validation_status", "gate_reason")),
            _table_section("Sources", bundle.sources, ("source_id", "title", "source_type", "provenance", "url")),
            _table_section("Claims", bundle.claims, ("claim_id", "claim_text", "source_ids", "confidence", "confidence_level")),
            _table_section("Validation Results", bundle.validation_results, ("validation_id", "target_id", "status", "validator", "message")),
            _table_section("Confidence Flags", bundle.confidence_flags, ("flag_id", "target_id", "severity", "reason", "confidence")),
            _table_section("Human Gates", bundle.human_gates, ("gate_id", "decision", "gate_reason", "required_roles", "reviewer")),
            _table_section("Agent Traces", bundle.agent_traces, ("trace_id", "agent_name", "model_route", "model", "execution_mode", "retry_count", "fallback_cause", "output_id", "output_type", "confidence", "rationale_summary")),
            _json_details("Raw Bundle JSON", _bundle_json(bundle)),
            "</main></body></html>",
        ]
    )
    return "\n".join(parts)


def write_run_html(run_id: str, output_html: str | Path, *, memory: MemoryStore | None = None) -> Path:
    """Write a run HTML view and return its path."""

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_run_html(run_id, memory=memory), encoding="utf-8")
    return output_path


def build_nct_report_html(nct_id: str, *, memory: MemoryStore | None = None) -> str:
    """Build a curated cumulative report for one NCT ID from Scientific Memory."""

    store = memory or MemoryStore()
    normalized_nct = nct_id.strip().upper()
    workflow_outputs = _latest_nct_workflow_outputs(normalized_nct, store)
    identity = _nct_report_identity(workflow_outputs, normalized_nct)
    title = f"{identity.get('asset_name') or normalized_nct} Development Brief"
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{escape(title)}</title>",
        f"<style>{_CSS}{_CUMULATIVE_CSS}</style></head><body>",
        "<main class='cumulative-report'>",
        _nct_report_hero(identity, workflow_outputs),
    ]
    if not workflow_outputs:
        parts.append(_section("No Persisted Workflow Outputs", _paragraphs([f"No completed Agent 3, Agent 4, or Agent 5 outputs were found for {normalized_nct}."])))
        parts.append("</main></body></html>")
        return "\n".join(parts)

    parts.extend(
        [
            _section("Executive Readout", _nct_executive_readout(workflow_outputs)),
            _section("Decision Metrics", _nct_kpis(workflow_outputs)),
            _section("Lifecycle Summary", _lifecycle_summary_table(workflow_outputs)),
        ]
    )
    for workflow_name in ("clinical_outcome_prediction", "due_diligence", "protocol_design"):
        item = workflow_outputs.get(workflow_name)
        if item is None:
            continue
        _, output = item
        if workflow_name == "clinical_outcome_prediction":
            parts.append(_cumulative_agent3_section(output))
        elif workflow_name == "due_diligence":
            parts.append(_cumulative_agent4_section(output))
        elif workflow_name == "protocol_design":
            parts.append(_cumulative_agent5_section(output))
    parts.append(_section("Source And Confidence Context", _source_confidence_context(workflow_outputs)))
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_nct_report(
    nct_id: str,
    output_html: str | Path | None = None,
    *,
    memory: MemoryStore | None = None,
) -> Path:
    """Write the cumulative NCT report and return its path."""

    store = memory or MemoryStore()
    output_path = Path(output_html) if output_html is not None else _default_nct_report_path(nct_id, store)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_nct_report_html(nct_id, memory=store), encoding="utf-8")
    return output_path


def write_nct_report_if_persistent(nct_id: str | None, *, memory: MemoryStore) -> Path | None:
    """Write the cumulative report for file-backed Scientific Memory stores."""

    if not nct_id or str(getattr(memory, "db_path", "")) == ":memory:":
        return None
    return write_nct_report(nct_id, memory=memory)


def _latest_nct_workflow_outputs(nct_id: str, store: MemoryStore) -> dict[str, tuple[Any, dict[str, Any]]]:
    outputs: dict[str, tuple[Any, dict[str, Any]]] = {}
    for workflow_name in ("clinical_outcome_prediction", "due_diligence", "protocol_design"):
        latest = store.get_latest_workflow_output(workflow_name=workflow_name, nct_id=nct_id)
        if latest is None:
            continue
        run, payload = latest
        if isinstance(payload, dict):
            outputs[workflow_name] = (run, payload)
    return outputs


def _default_nct_report_path(nct_id: str, store: MemoryStore) -> Path:
    filename = f"{_safe_filename(nct_id.strip().upper())}.html"
    db_path = getattr(store, "db_path", Path(DEFAULT_DB_PATH))
    db_path = Path(db_path)
    if str(db_path) != ":memory:" and db_path.parent.name == ".pharma_os":
        return db_path.parent.parent / "reports" / filename
    return Path("reports") / filename


def _safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe or "report"


def _nct_report_identity(outputs: dict[str, tuple[Any, dict[str, Any]]], nct_id: str) -> dict[str, Any]:
    identity: dict[str, Any] = {"nct_id": nct_id}
    for _, output in outputs.values():
        trial = _dict(output.get("target_trial") or output.get("trial") or output.get("trial_identity"))
        asset = _dict(output.get("asset_identity"))
        if trial:
            identity.setdefault("title", trial.get("brief_title") or trial.get("official_title"))
            identity.setdefault("status", _pretty_status(trial.get("overall_status")))
            identity.setdefault("phase", _pretty_phase_list(_list(trial.get("phases"))))
            identity.setdefault("conditions", ", ".join(_list(trial.get("conditions"))))
            identity.setdefault("sponsor", trial.get("sponsor") or _nested(trial, "lead_sponsor", "name"))
            identity.setdefault("enrollment", trial.get("enrollment_count"))
        if asset:
            identity.setdefault("asset_name", asset.get("asset_name"))
            identity.setdefault("indication", asset.get("normalized_indication"))
            identity.setdefault("modality", asset.get("modality"))
            identity.setdefault("sponsor", asset.get("sponsor"))
    identity.setdefault("indication", identity.get("conditions"))
    return identity


def _nct_report_hero(identity: dict[str, Any], outputs: dict[str, tuple[Any, dict[str, Any]]]) -> str:
    latest_completed = [
        getattr(run, "completed_at", None)
        for run, _ in outputs.values()
        if getattr(run, "completed_at", None) is not None
    ]
    subtitle = identity.get("title") or "Cumulative PharmaOS scientific development brief."
    return _hero(
        "PharmaOS Cumulative Report",
        identity.get("asset_name") or identity.get("nct_id"),
        subtitle,
        [
            ("NCT", identity.get("nct_id")),
            ("Indication", identity.get("indication")),
            ("Phase", identity.get("phase")),
            ("Status", identity.get("status")),
            ("Sponsor", identity.get("sponsor")),
            ("Updated", max(latest_completed).date().isoformat() if latest_completed else None),
        ],
    )


def _nct_executive_readout(outputs: dict[str, tuple[Any, dict[str, Any]]]) -> str:
    takeaways: list[str] = []
    agent3 = _output(outputs, "clinical_outcome_prediction")
    agent4 = _output(outputs, "due_diligence")
    agent5 = _output(outputs, "protocol_design")
    if agent3:
        endpoint = _dict(agent3.get("endpoint_risk_assessment"))
        enrollment = _dict(agent3.get("enrollment_duration_risk"))
        pos = _dict(agent3.get("historical_pos_estimate"))
        takeaways.append(
            _join_sentence_parts(
                [
                    f"Clinical risk is framed as endpoint risk {_display(endpoint.get('risk_level')).lower()}" if endpoint.get("risk_level") else None,
                    f"enrollment risk {_display(enrollment.get('risk_level')).lower()}" if enrollment.get("risk_level") else None,
                    f"historical PoS {_percent(pos.get('probability_of_success'))}" if pos.get("probability_of_success") is not None else None,
                ]
            )
        )
    if agent4:
        memo = _dict(agent4.get("asset_memo"))
        commercial = _dict(agent4.get("commercial_model"))
        rnpv = _dict(agent4.get("rnpv"))
        memo_summary = _first_sentences(memo.get("summary"), 2)
        takeaways.append(memo_summary)
        if (
            commercial.get("calculable") is False
            or rnpv.get("calculable") is False
        ) and "non-calculable" not in memo_summary.casefold():
            takeaways.append("Commercial sizing or rNPV remains non-calculable under the persisted evidence and assumptions.")
    if agent5:
        brief = _dict(agent5.get("protocol_design_brief"))
        synopsis = _section_body(brief.get("executive_synopsis"))
        takeaways.append(_first_sentences(synopsis, 2))
    return _bullets(_dedupe_text(takeaways, limit=6))


def _nct_kpis(outputs: dict[str, tuple[Any, dict[str, Any]]]) -> str:
    agent3 = _output(outputs, "clinical_outcome_prediction")
    agent4 = _output(outputs, "due_diligence")
    agent5 = _output(outputs, "protocol_design")
    endpoint = _dict(agent3.get("endpoint_risk_assessment")) if agent3 else {}
    enrollment = _dict(agent3.get("enrollment_duration_risk")) if agent3 else {}
    pos = _dict((agent4 or {}).get("pos")) or _dict((agent3 or {}).get("historical_pos_estimate"))
    approval = _dict((agent3 or {}).get("approval_likelihood_proxy"))
    commercial = _dict((agent4 or {}).get("commercial_model"))
    rnpv = _dict((agent4 or {}).get("rnpv"))
    benchmark = _dict((agent5 or {}).get("analog_benchmark_bundle"))
    return _kpi_grid(
        [
            ("Historical PoS", _percent(pos.get("probability_of_success")), _pretty_lookup(pos.get("lookup_key")) or pos.get("current_phase")),
            ("Approval Proxy", _percent(approval.get("probability")), approval.get("basis")),
            ("Endpoint Risk", endpoint.get("risk_level") or "NA", "Agent 3 endpoint assessment."),
            ("Enrollment Risk", enrollment.get("risk_level") or "NA", "Agent 3 operational timing assessment."),
            ("Peak Net Sales", _money_or_not_calculable(commercial.get("peak_net_sales")), "Deterministic commercial model."),
            ("Base rNPV", _money_or_not_calculable(rnpv.get("rnpv")), "Deterministic rNPV."),
            ("LOE", rnpv.get("loe_year") or _nested(agent4 or {}, "patent_loe_review", "estimated_loe_year") or "NA", "Loss-of-exclusivity assumption or estimate."),
            ("Analog Confidence", _percent(benchmark.get("confidence")), "Agent 5 analog benchmark confidence."),
        ]
    )


def _lifecycle_summary_table(outputs: dict[str, tuple[Any, dict[str, Any]]]) -> str:
    rows = []
    labels = {
        "clinical_outcome_prediction": "Agent 3 - Clinical Outcome Prediction",
        "due_diligence": "Agent 4 - Due Diligence",
        "protocol_design": "Agent 5 - Protocol Design",
    }
    for workflow_name in ("clinical_outcome_prediction", "due_diligence", "protocol_design"):
        item = outputs.get(workflow_name)
        if item is None:
            rows.append({"workflow": labels[workflow_name], "status": "not available", "run": "", "updated": "", "confidence": "", "gate": ""})
            continue
        run, output = item
        gate = _dict(output.get("human_gate"))
        rows.append(
            {
                "workflow": labels[workflow_name],
                "status": _pretty_status(output.get("validation_status") or getattr(run, "validation_status", None)),
                "run": getattr(run, "run_id", output.get("run_id")),
                "updated": _date(getattr(run, "completed_at", None)),
                "confidence": _percent(output.get("confidence")),
                "gate": _pretty_status(gate.get("decision") or ""),
            }
        )
    return _dict_table(rows, ("workflow", "status", "updated", "confidence", "gate", "run"))


def _cumulative_agent3_section(output: dict[str, Any]) -> str:
    trial = _dict(output.get("trial_identity"))
    asset = _dict(output.get("asset_identity"))
    endpoint = _dict(output.get("endpoint_risk_assessment"))
    enrollment = _dict(output.get("enrollment_duration_risk"))
    comparator = _dict(output.get("comparator_benchmarking"))
    pos = _dict(output.get("historical_pos_estimate"))
    approval = _dict(output.get("approval_likelihood_proxy"))
    failure = _dict(output.get("failure_mode_classification"))
    safety = _dict(output.get("safety_context"))
    findings = _dedupe_text(
        [
            _risk_sentence("Endpoint", endpoint),
            _risk_sentence("Enrollment", enrollment),
            _first_sentences(comparator.get("benchmark_summary"), 2, max_chars=340),
            safety.get("summary") or _first_missing_reason(safety),
        ],
        limit=5,
    )
    risks = _top_risk_items([
        *_list(failure.get("likely_failure_modes")),
        *_list(output.get("missing_data_flags")),
    ], limit=5)
    review_flags = top_review_flags_from_payload(output, limit=5)
    body = "".join(
        [
            _compact_workflow_header(
                "Agent 3 - Clinical Outcome Prediction",
                _join_sentence_parts(
                    [
                        f"{asset.get('asset_name') or trial.get('nct_id')} has endpoint risk {_display(endpoint.get('risk_level')).lower()}" if endpoint.get("risk_level") else None,
                        f"enrollment risk {_display(enrollment.get('risk_level')).lower()}" if enrollment.get("risk_level") else None,
                        f"historical PoS {_percent(pos.get('probability_of_success'))}" if pos.get("probability_of_success") is not None else None,
                    ]
                ),
                output,
            ),
            _workflow_body(
                _cards(
                    [
                        ("Key Findings", _bullets(findings)),
                        ("Major Risks And Uncertainties", _bullets(risks)),
                    ]
                ),
                _kv_table(
                    {
                        "nct_id": trial.get("nct_id") or _nested(output, "input", "nct_id"),
                        "phase": _pretty_phase_list(_list(trial.get("phases"))),
                        "status": _pretty_status(trial.get("overall_status")),
                        "asset": asset.get("asset_name"),
                        "indication": asset.get("normalized_indication") or ", ".join(_list(trial.get("conditions"))),
                        "historical_pos": _percent(pos.get("probability_of_success")),
                        "approval_proxy": _percent(approval.get("probability")),
                        "matched_comparator_trials": comparator.get("matched_public_trials_count"),
                    }
                ),
            ),
            _details("Review Context", _flag_table(list(review_flags)) + _source_context_line(output)),
        ]
    )
    return _section("Agent 3 - Clinical Outcome Prediction", body)


def _cumulative_agent4_section(output: dict[str, Any]) -> str:
    trial = _dict(output.get("target_trial"))
    asset = _dict(output.get("asset_identity"))
    memo = _dict(output.get("asset_memo"))
    clinical = _dict(output.get("clinical_risk_summary"))
    evidence = _dict(output.get("clinical_evidence"))
    landscape = _dict(output.get("competitive_landscape"))
    safety = _dict(output.get("safety_label_summary"))
    patent = _dict(output.get("patent_loe_review"))
    pricing = _dict(output.get("pricing"))
    commercial = _dict(output.get("commercial_model"))
    rnpv = _dict(output.get("rnpv"))
    red_flags = _top_risk_items(_list(output.get("red_flags")), limit=5)
    missing = _top_risk_items(_list(output.get("missing_data_flags")), limit=4)
    review_flags = top_review_flags_from_payload(output, limit=5)
    findings = _dedupe_text(
        [
            _first_sentences(memo.get("summary"), 2),
            _clinical_risk_sentence(clinical),
            _ctgov_evidence_sentence(trial, evidence),
            _first_sentences(landscape.get("benchmark_summary"), 2, max_chars=340),
            patent.get("review_summary"),
            safety.get("warnings_summary"),
        ],
        limit=6,
    )
    body = "".join(
        [
            _compact_workflow_header("Agent 4 - Clinical-Stage Due Diligence", _first_sentences(memo.get("summary"), 2), output),
            _workflow_body(
                _cards(
                    [
                        ("Decision-Relevant Findings", _bullets(findings)),
                        ("Major Risks And Uncertainties", _bullets(_dedupe_risk_texts([*red_flags, *missing], limit=5))),
                    ]
                ),
                _kv_table(
                    {
                        "asset": asset.get("asset_name"),
                        "indication": asset.get("normalized_indication"),
                        "clinical_pos": _percent(_nested(output, "pos", "probability_of_success") or clinical.get("historical_pos")),
                        "pricing_analog": _brief_pricing_analog(pricing.get("matched_product")),
                        "annual_wac": _money(pricing.get("annual_wac")),
                        "peak_net_sales": _money_or_not_calculable(commercial.get("peak_net_sales")),
                        "base_rnpv": _money_or_not_calculable(rnpv.get("rnpv")),
                        "loe_year": rnpv.get("loe_year") or patent.get("estimated_loe_year"),
                        "commercial_calculable": commercial.get("calculable"),
                        "rnpv_calculable": rnpv.get("calculable"),
                    }
                ),
            ),
            _details("Review Flags", _flag_table(list(review_flags)) + _source_context_line(output)),
        ]
    )
    return _section("Agent 4 - Due Diligence", body)


def _cumulative_agent5_section(output: dict[str, Any]) -> str:
    brief = _dict(output.get("protocol_design_brief"))
    intent = _dict(output.get("next_study_intent") or brief.get("next_study_intent"))
    benchmark = _dict(output.get("analog_benchmark_bundle"))
    reviewer = _dict(brief.get("reviewer_critique"))
    synopsis = _section_body(brief.get("executive_synopsis"))
    findings = _dedupe_text(
        [
            _first_sentences(_section_body(brief.get("strategic_rationale")), 2, max_chars=340),
            _protocol_benchmark_sentence(benchmark),
            _section_body(brief.get("endpoint_strategy")),
            _section_body(brief.get("operational_feasibility_risks")),
        ],
        limit=5,
    )
    decisions = [
        _protocol_design_decision(item)
        for item in _list(output.get("analog_derived_design_decisions"))
        if isinstance(item, dict) and item.get("proposed_value")
    ][:5]
    limitations = _dedupe_risk_texts(
        [
            *_list(benchmark.get("limitations")),
            *_list(reviewer.get("limitations")),
            *_top_risk_items(_list(output.get("missing_data_flags")), limit=3),
        ],
        limit=5,
    )
    review_flags = top_review_flags_from_payload(output, limit=5)
    body = "".join(
        [
            _compact_workflow_header("Agent 5 - Protocol Design Brief", _protocol_takeaway(intent, synopsis), output),
            _workflow_body(
                _cards(
                    [
                        ("Protocol Strategy Findings", _bullets(findings)),
                        ("Risks And Limitations", _bullets(limitations)),
                    ]
                ),
                _kv_table(
                    {
                        "proposed_next_stage": _brief_stage(intent.get("proposed_next_stage")),
                        "study_role": _brief_stage(intent.get("study_role")),
                        "key_clinical_question": intent.get("key_clinical_question"),
                        "selected_analogs": len(_list(benchmark.get("selected_analog_ids"))),
                        "follow_on_trials": len(_list(output.get("follow_on_trials"))),
                        "median_enrollment": _summary_value(benchmark.get("enrollment"), "median"),
                        "median_duration": _summary_value(benchmark.get("planned_duration_months"), "median"),
                        "benchmark_confidence": _percent(benchmark.get("confidence")),
                    }
                ),
            ),
            _cards([("Analog-Derived Design Signals", _bullets(decisions or ["No analog-derived design signals were emitted."]))]),
            _details(
                "Review Flags",
                _flag_table(list(review_flags)) + _source_context_line(output),
            ),
        ]
    )
    return _section("Agent 5 - Protocol Design", body)


def _compact_workflow_header(title: str, takeaway: Any, output: dict[str, Any]) -> str:
    status = output.get("validation_status")
    review_flags = top_review_flags_from_payload(output, limit=5)
    chips = [
        ("Validation", status),
        ("Confidence", _percent(output.get("confidence"))),
        ("Review Flags", _review_flag_badge(review_flags)),
        ("Run", output.get("run_id")),
    ]
    return (
        "<div class='workflow-head'>"
        f"<div><p class='eyebrow'>{escape(title)}</p>{_paragraphs([takeaway])}</div>"
        f"<div class='status-strip'>{''.join(_status_pill(label, value) for label, value in chips if _display(value))}</div>"
        "</div>"
    )


def _workflow_body(narrative: str, metrics: str) -> str:
    return f"<div class='workflow-body'><div class='workflow-narrative'>{narrative}</div><aside class='metrics-panel'>{metrics}</aside></div>"


def _status_pill(label: str, value: Any) -> str:
    text = _display(value)
    shown = _pretty_status(text) if label == "Validation" else text
    status_class = "status-warn" if any(token in text.casefold() for token in ("review", "warning", "failed", "blocked", "non-calculable")) else "status-ok"
    return f"<span class='status-pill {status_class}'><strong>{escape(label)}</strong>{escape(shown)}</span>"


def _review_flag_badge(flags: tuple[dict[str, Any], ...]) -> str:
    if not flags:
        return "None"
    top = flags[0]
    if len(flags) == 1:
        return f"{_display(top.get('severity')).title()}: {_truncate(_display(top.get('reason')), 72)}"
    counts = {}
    for flag in flags:
        severity = _display(flag.get("severity")).casefold() or "medium"
        counts[severity] = counts.get(severity, 0) + 1
    summary = ", ".join(f"{count} {severity}" for severity, count in counts.items())
    return f"{len(flags)} flags: {summary}"


def _first_review_reason(flags: tuple[dict[str, Any], ...]) -> str:
    if not flags:
        return ""
    top = flags[0]
    return f"{_display(top.get('severity')).title()}: {_truncate(_display(top.get('reason')), 120)}"


def _source_confidence_context(outputs: dict[str, tuple[Any, dict[str, Any]]]) -> str:
    rows = []
    labels = {
        "clinical_outcome_prediction": "Agent 3 - Clinical Outcome Prediction",
        "due_diligence": "Agent 4 - Due Diligence",
        "protocol_design": "Agent 5 - Protocol Design",
    }
    for workflow_name, (run, output) in outputs.items():
        summary = _dict(output.get("execution_mode_summary"))
        review_flags = top_review_flags_from_payload(output, limit=5)
        rows.append(
            {
                "workflow": labels.get(workflow_name, _label(workflow_name)),
                "sources": len(_list(output.get("sources"))),
                "claims": len(_list(output.get("claims"))),
                "review_flags": len(review_flags),
                "top_review_reason": _first_review_reason(review_flags),
                "validation": _pretty_status(output.get("validation_status") or getattr(run, "validation_status", None)),
                "execution": summary.get("summary") or "",
            }
        )
    return (
        _paragraphs(["This report is generated only from persisted typed workflow outputs in Scientific Memory. It intentionally omits raw JSON, full trace logs, low-value flags, and repetitive generated sections."])
        + _dict_table(rows, ("workflow", "sources", "claims", "review_flags", "top_review_reason", "validation", "execution"))
    )


def _source_context_line(output: dict[str, Any]) -> str:
    review_flags = top_review_flags_from_payload(output, limit=5)
    pieces = [
        f"{len(_list(output.get('sources')))} persisted sources",
        f"{len(_list(output.get('claims')))} source-backed claims",
        f"{len(review_flags)} review flags surfaced",
    ]
    return f"<p class='sources'>{escape('; '.join(pieces))}</p>"


def _details(title: str, body: str) -> str:
    return f"<details><summary>{escape(title)}</summary>{body}</details>"


def _output(outputs: dict[str, tuple[Any, dict[str, Any]]], workflow_name: str) -> dict[str, Any] | None:
    item = outputs.get(workflow_name)
    return item[1] if item else None


def _section_body(value: Any) -> str:
    item = _dict(value)
    if item:
        return _display(item.get("body") or item.get("summary") or item.get("rationale"))
    return _display(value)


def _risk_sentence(label: str, section: dict[str, Any]) -> str:
    if not section:
        return ""
    top_gap = _top_risk_items(_list(section.get("missing_data_flags")), limit=1)
    return _join_sentence_parts(
        [
            f"{label} risk: {section.get('risk_level')}",
            _first_sentences(section.get("rationale"), 1, max_chars=260),
            f"Key uncertainty: {top_gap[0].split(': ', 1)[-1]}" if top_gap else None,
        ]
    )


def _clinical_risk_sentence(clinical: dict[str, Any]) -> str:
    return _join_sentence_parts(
        [
            f"Endpoint risk {clinical.get('endpoint_risk_level')}" if clinical.get("endpoint_risk_level") else None,
            f"enrollment risk {clinical.get('enrollment_duration_risk_level')}" if clinical.get("enrollment_duration_risk_level") else None,
            f"historical PoS {_percent(clinical.get('historical_pos'))}" if clinical.get("historical_pos") is not None else None,
        ]
    )


def _ctgov_evidence_sentence(trial: dict[str, Any], evidence: dict[str, Any]) -> str:
    if not trial and not evidence:
        return ""
    phase = _pretty_phase_list(_list(trial.get("phases")))
    status = _pretty_status(trial.get("overall_status"))
    primary_count = len(_list(trial.get("primary_endpoints")))
    pubmed_count = evidence.get("pubmed_article_count")
    trial_context = " ".join(part for part in (phase, status) if part)
    pieces = [
        f"CT.gov confirms {trial_context} target-trial evidence" if trial_context else "CT.gov target-trial evidence is available",
        f"{primary_count} primary endpoint{'s' if primary_count != 1 else ''}" if primary_count else None,
        f"PubMed metadata count {pubmed_count}" if pubmed_count is not None else None,
    ]
    return _join_sentence_parts(pieces)


def _brief_pricing_analog(value: Any) -> str:
    text = _display(value)
    if not text:
        return ""
    for marker in (" (pricing analog:", " - "):
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    return _truncate(text, 120)


def _protocol_takeaway(intent: dict[str, Any], synopsis: Any) -> str:
    indication = intent.get("indication")
    stage = _brief_stage(intent.get("proposed_next_stage"))
    role = _brief_stage(intent.get("study_role"))
    pieces = [
        f"Draft next-study strategy brief for {indication}" if indication else _first_sentences(synopsis, 1, max_chars=180),
        f"proposed next stage: {stage}" if stage else None,
        f"role: {role}" if role else None,
        "human review is required before protocol use",
    ]
    return _join_sentence_parts(pieces)


def _protocol_benchmark_sentence(benchmark: dict[str, Any]) -> str:
    selected_count = len(_list(benchmark.get("selected_analog_ids")))
    enrollment = _summary_value(benchmark.get("enrollment"), "median")
    duration = _summary_value(benchmark.get("planned_duration_months"), "median")
    comparator = ""
    categories = _list(benchmark.get("comparator_categories"))
    if categories and isinstance(categories[0], dict):
        comparator = _brief_stage(categories[0].get("label"))
    pieces = [
        f"Benchmarking is based on {selected_count} selected analog CT.gov trial{'s' if selected_count != 1 else ''}",
        f"median enrollment {enrollment}" if enrollment != "NA" else None,
        f"median planned duration {duration}" if duration != "NA" else None,
        f"comparator pattern {comparator}" if comparator else None,
    ]
    return _join_sentence_parts(pieces)


def _protocol_design_decision(item: dict[str, Any]) -> str:
    field = _label(item.get("field_name"))
    value = _brief_stage(item.get("proposed_value"))
    method = _brief_stage(item.get("derivation_method"))
    return f"{field}: {value} ({method})" if method else f"{field}: {value}"


def _brief_stage(value: Any) -> str:
    text = _pretty_embedded_enums(_display(value)).replace("_", " ").replace("Phase 2", "Phase II")
    text = text.replace("study.)", "study)").replace("'.", "'").strip()
    return _truncate(text, 180)


def _join_sentence_parts(parts: list[Any]) -> str:
    text = "; ".join(_display(part).strip(" .;") for part in parts if _display(part).strip())
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _first_sentences(value: Any, count: int = 1, *, max_chars: int = 520) -> str:
    text = " ".join(_display(value).split())
    if not text:
        return ""
    sentences: list[str] = []
    for sentence in text.replace("? ", "?. ").replace("! ", "!. ").split(". "):
        clean = sentence.strip()
        if not clean:
            continue
        if clean[-1] not in ".!?":
            clean += "."
        sentences.append(clean)
        if len(sentences) >= count:
            break
    result = " ".join(sentences) or text
    return _pretty_embedded_enums(_truncate(result, max_chars))


def _dedupe_text(values: list[Any], *, limit: int) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        text = " ".join(_display(value).split())
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        results.append(_truncate(text, 420))
        if len(results) >= limit:
            break
    return results


def _top_risk_items(items: list[Any], *, limit: int) -> list[str]:
    rows = [item for item in items if isinstance(item, dict)]
    rows.sort(key=lambda item: (_severity_rank(item.get("severity")), _display(item.get("reason") or item.get("rationale"))))
    texts = []
    seen_fingerprints: set[str] = set()
    for item in rows:
        reason = item.get("reason") or item.get("rationale") or item.get("message")
        severity = item.get("severity")
        fingerprint = _risk_fingerprint(reason)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        if reason:
            texts.append(f"{severity}: {reason}" if severity else reason)
    return _dedupe_text(texts, limit=limit)


def _dedupe_risk_texts(values: list[Any], *, limit: int) -> list[str]:
    seen_fingerprints: set[str] = set()
    results: list[str] = []
    for value in values:
        text = " ".join(_display(value).split())
        if not text:
            continue
        text = _humanized_risk_text(text)
        fingerprint = _risk_fingerprint(text)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        results.append(_truncate(text, 320))
        if len(results) >= limit:
            break
    return results


def _humanized_risk_text(text: str) -> str:
    if "commercial-population-measure-missing" in text:
        return text.replace("commercial-population-measure-missing", "the population measure required for commercial sizing is missing")
    if any(token in text for token in ("market_pubmed_no_articles", "pubmed_market_query_failed", "PubMedError")):
        severity = text.split(":", 1)[0] if ":" in text else "medium"
        return f"{severity}: Commercial sizing could not retrieve adequate PubMed market epidemiology evidence."
    return _pretty_embedded_enums(text)


def _risk_fingerprint(value: Any) -> str:
    text = _display(value).casefold()
    if "benchmark stability" in text or "fewer than five selected analog" in text:
        return "benchmark-stability"
    if "pubmed market epidemiology" in text or "market_pubmed" in text or "pubmed_market_query_failed" in text:
        return "market-pubmed-gap"
    if "commercial" in text and "calculable" in text:
        return "commercial-not-calculable"
    if "rnpv" in text and "calculable" in text:
        return "rnpv-not-calculable"
    if "selected_population_measure" in text or "commercial-population" in text:
        return "commercial-population-gap"
    if "rxnorm" in text:
        return "rxnorm-missing"
    if "modality" in text:
        return "modality-missing"
    if "label" in text or "openfda" in text:
        return "label-missing"
    return text


def _severity_rank(value: Any) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(_display(value).casefold(), 4)


def _first_missing_reason(section: dict[str, Any]) -> str:
    for item in _list(section.get("missing_data_flags")):
        if isinstance(item, dict) and item.get("reason"):
            return item["reason"]
    return ""


def _review_questions_from_output(output: dict[str, Any], risks: list[str], *, limit: int) -> list[str]:
    gate = _dict(output.get("human_gate"))
    questions = []
    if gate.get("gate_reason"):
        questions.append(gate["gate_reason"])
    questions.extend(risks)
    return _dedupe_text(questions, limit=limit)


def _date(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "date"):
        return value.date().isoformat()
    text = str(value)
    return text[:10]


def _pretty_phase_list(values: list[Any]) -> str:
    phases = [_pretty_phase(value) for value in values if _display(value)]
    return ", ".join(phases)


def _pretty_phase(value: Any) -> str:
    text = _display(value).replace("_", " ").strip()
    compact = text.replace(" ", "").upper()
    mapping = {
        "PHASE1": "Phase I",
        "PHASE2": "Phase II",
        "PHASE3": "Phase III",
        "PHASE4": "Phase IV",
        "EARLYPHASE1": "Early Phase I",
    }
    return mapping.get(compact, text.title())


def _pretty_status(value: Any) -> str:
    text = _display(value).replace("_", " ").strip()
    if not text:
        return ""
    return text.title().replace(" And ", " and ").replace(" Of ", " of ")


def _pretty_lookup(value: Any) -> str:
    return _display(value).replace("|", " / ")


def _pretty_embedded_enums(value: Any) -> str:
    text = _display(value)
    replacements = {
        "PHASE1": "Phase I",
        "PHASE2": "Phase II",
        "PHASE3": "Phase III",
        "PHASE4": "Phase IV",
        "ACTIVE_NOT_RECRUITING": "Active Not Recruiting",
        "NOT_YET_RECRUITING": "Not Yet Recruiting",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", " ") if text.isupper() and "_" in text else text


def _money_or_not_calculable(value: Any) -> str:
    return _money(value) if _float(value) is not None else "Not calculable"


def _workflow_report_section(output_json: Any, workflow_name: str) -> str:
    if not isinstance(output_json, dict):
        return ""
    if workflow_name in {"control_tower", "control_tower_orchestration"} or output_json.get("report", {}).get("parent_run_id"):
        return _control_tower_report(output_json)
    if workflow_name == "due_diligence" or "asset_memo" in output_json:
        return _due_diligence_report(output_json)
    if workflow_name == "clinical_outcome_prediction" or "endpoint_risk_assessment" in output_json:
        return _clinical_outcome_prediction_report(output_json)
    if workflow_name == "protocol_design" or "protocol_design_brief" in output_json:
        return _protocol_design_report(output_json)
    return ""


def _execution_mode_summary_section(bundle: Any) -> str:
    summary = None
    if isinstance(bundle.output_json, dict):
        summary = bundle.output_json.get("execution_mode_summary")
        report = _dict(bundle.output_json.get("report"))
        summary = summary or report.get("execution_mode_summary")
    if not summary and bundle.reports:
        latest = bundle.reports[-1]
        summary = latest.execution_mode_summary.model_dump(mode="json")
    runtime_table = _runtime_trace_table(bundle.agent_traces)
    if not isinstance(summary, dict):
        return _section("AI Execution Mode", runtime_table) if runtime_table else ""
    return _section(
        "AI Execution Mode",
        _two_col(
            _kv_table(
                {
                    "summary": summary.get("summary"),
                    "requested_reasoning_steps": summary.get("requested_reasoning_steps"),
                    "live_ai_calls_completed": summary.get("live_ai_calls_completed"),
                    "deterministic_fallbacks_used": summary.get("deterministic_fallbacks_used"),
                }
            ),
            _kv_table(
                {
                    "live_agent_calls_completed": summary.get("live_agent_calls_completed"),
                    "direct_llm_calls_completed": summary.get("direct_llm_calls_completed"),
                    "reused_artifacts_used": summary.get("reused_artifacts_used"),
                }
            ),
        )
        + runtime_table,
    )


def _control_tower_report(output: dict[str, Any]) -> str:
    report = _dict(output.get("report"))
    request = _dict(output.get("request"))
    plans = _list(output.get("plans") or ([output.get("plan")] if output.get("plan") else []))
    steps = _list(output.get("step_results"))
    replans = _list(output.get("replans"))
    snapshots = _list(output.get("snapshots"))
    validation_results = _list(output.get("validation_results"))
    final_snapshot = _dict(output.get("final_snapshot"))
    unavailable = _list(report.get("unavailable_modules"))
    unresolved = _list(report.get("unresolved_gates"))
    return "".join(
        [
            _hero(
                "Control Tower Orchestration",
                request.get("objective") or report.get("objective") or "Control Tower objective",
                report.get("final_state_summary") or "Memory-aware orchestration report.",
                [
                    ("Parent Run", output.get("run_id")),
                    ("NCT", request.get("nct_id")),
                    ("Plans", len(plans)),
                    ("Steps", len(steps)),
                    ("Unresolved Gates", len(unresolved)),
                ],
            ),
            _control_tower_executive_audit(report, request, steps, plans, validation_results, output),
            _kpi_grid(
                [
                    ("Executed", sum(1 for step in steps if isinstance(step, dict) and step.get("status") in {"executed", "refreshed"}), "Workflow steps executed or refreshed."),
                    ("Reused", sum(1 for step in steps if isinstance(step, dict) and step.get("status") == "reused"), "Artifacts reused from Scientific Memory."),
                    ("Skipped", sum(1 for step in steps if isinstance(step, dict) and step.get("status") == "skipped"), "Steps explicitly skipped by plan."),
                    ("Blocked", sum(1 for step in steps if isinstance(step, dict) and step.get("status") == "blocked"), "Steps blocked by gates or unavailable capabilities."),
                    ("Replans", len(replans), "Material state changes that triggered replanning."),
                    ("Final Artifacts", len(_list(final_snapshot.get("artifacts"))), "Artifacts visible in final ScientificStateSnapshot."),
                ]
            ),
            _section(
                "Objective And State",
                _two_col(
                    _kv_table(
                        {
                            "objective": request.get("objective"),
                            "nct_id": request.get("nct_id"),
                            "asset_name": request.get("asset_name"),
                            "indication": request.get("indication"),
                            "force_refresh": ", ".join(_list(request.get("force_refresh"))),
                        }
                    ),
                    _kv_table(
                        {
                            "initial_state": report.get("initial_state_summary"),
                            "final_state": report.get("final_state_summary"),
                            "initial_snapshot": report.get("initial_snapshot_id"),
                            "final_snapshot": report.get("final_snapshot_id"),
                        }
                    ),
                ),
            ),
            _section(
                "Decision Evidence",
                _cards(
                    [
                        ("Pending Decision", _paragraphs([report.get("pending_decision_summary") or "No pending decision was recorded."])),
                        ("Evidence Requirements", _bullets(_list(report.get("evidence_requirement_summaries")) or ["None recorded."])),
                        ("Critical Gaps", _bullets(_list(report.get("critical_evidence_gaps")) or ["None recorded."])),
                        ("Unresolved Or Contradictory Claims", _bullets([*_list(report.get("unresolved_claims")), *_list(report.get("contradictory_claims"))] or ["None recorded."])),
                    ]
                ),
            ),
            _section("Plans", _control_tower_plan_tables(plans)),
            _section("Step Results", _control_tower_step_table(steps)),
            _section(
                "Replans And Blocks",
                _cards(
                    [
                        ("Replans", _bullets(_list(report.get("replan_summaries")) or [item.get("reason") for item in replans if isinstance(item, dict)] or ["None."])),
                        ("Unresolved Gates", _bullets(unresolved or ["None."])),
                        ("Unavailable Modules", _bullets(unavailable or ["None."])),
                    ]
                ),
            ),
            _section("Final State", _control_tower_artifact_table(_list(final_snapshot.get("artifacts")))),
            _section("Planning Snapshots", _control_tower_snapshot_table(snapshots)),
        ]
    )


def _control_tower_executive_audit(
    report: dict[str, Any],
    request: dict[str, Any],
    steps: list[Any],
    plans: list[Any],
    validation_results: list[Any],
    output: dict[str, Any],
) -> str:
    identifiers = _dict(request.get("identifiers"))
    action_taken = _control_tower_action_taken(steps, validation_results)
    executed_steps = [step for step in steps if isinstance(step, dict) and step.get("status") in {"executed", "refreshed"}]
    reused_steps = [step for step in steps if isinstance(step, dict) and step.get("status") == "reused"]
    first_step = next((step for step in steps if isinstance(step, dict)), None)
    child_runs = [step.get("child_run_id") for step in executed_steps if step.get("child_run_id")]
    outputs = [
        step.get("output_id") or step.get("reused_output_id")
        for step in steps
        if isinstance(step, dict) and (step.get("output_id") or step.get("reused_output_id"))
    ]
    why = _control_tower_why(first_step, plans, validation_results)
    attention = _control_tower_attention_items(steps, validation_results, report)
    return "".join(
        [
            _section(
                "What Happened",
                _two_col(
                    _kv_table(
                        {
                            "requested_goal": request.get("objective") or report.get("objective"),
                            "inferred_capability": identifiers.get("target_capability"),
                            "execution_intent": identifiers.get("execution_intent"),
                            "nct_id": request.get("nct_id"),
                            "action_taken": action_taken,
                        }
                    ),
                    _kv_table(
                        {
                            "workflow_executed": "yes" if executed_steps else "no",
                            "memory_reused": "yes" if reused_steps else "no",
                            "child_run_ids": ", ".join(str(item) for item in child_runs),
                            "output_ids": ", ".join(str(item) for item in outputs),
                            "parent_run_id": output.get("run_id"),
                        }
                    ),
                ),
            ),
            _section("Why", _paragraphs(why)),
            _section("Human Attention Needed", _bullets(attention or ["No run-specific attention items were identified."])),
        ]
    )


def _control_tower_action_taken(steps: list[Any], validation_results: list[Any]) -> str:
    if any(isinstance(step, dict) and step.get("status") in {"executed", "refreshed"} for step in steps):
        actions = [
            f"{step.get('capability_name')} {step.get('status')}"
            for step in steps
            if isinstance(step, dict) and step.get("status") in {"executed", "refreshed"}
        ]
        return "; ".join(actions)
    if any(isinstance(step, dict) and step.get("status") == "reused" for step in steps):
        actions = [
            f"{step.get('capability_name')} reused existing artifact {step.get('reused_output_id') or step.get('reused_run_id')}"
            for step in steps
            if isinstance(step, dict) and step.get("status") == "reused"
        ]
        return "; ".join(actions)
    if any(isinstance(step, dict) and step.get("status") in {"blocked", "failed"} for step in steps):
        step = next(step for step in steps if isinstance(step, dict) and step.get("status") in {"blocked", "failed"})
        return f"No workflow executed; {step.get('capability_name')} {step.get('status')}."
    if _failed_validation_messages(validation_results):
        return "No workflow executed; Control Tower plan validation failed."
    return "No workflow step was executed or reused."


def _control_tower_why(first_step: Any, plans: list[Any], validation_results: list[Any]) -> list[str]:
    failed = _failed_validation_messages(validation_results)
    if failed:
        return [f"Plan validation failed: {item}" for item in failed[:3]]
    if isinstance(first_step, dict) and first_step.get("rationale"):
        return [first_step.get("rationale")]
    latest_plan = next((plan for plan in reversed(plans) if isinstance(plan, dict)), None)
    if latest_plan and latest_plan.get("objective_interpretation"):
        return [latest_plan.get("objective_interpretation")]
    return ["The Control Tower did not emit a planner rationale."]


def _control_tower_attention_items(steps: list[Any], validation_results: list[Any], report: dict[str, Any]) -> list[str]:
    items = _failed_validation_messages(validation_results)
    items.extend(str(item) for item in _list(report.get("fallback_summaries")) if item)
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("status") in {"blocked", "failed"} and step.get("rationale"):
            items.append(str(step.get("rationale")))
        for gate in _list(step.get("gates")):
            if isinstance(gate, dict) and gate.get("gate_reason"):
                items.append(str(gate.get("gate_reason")))
    items.extend(str(item) for item in _list(report.get("unavailable_modules")) if item)
    return list(dict.fromkeys(items))[:8]


def _failed_validation_messages(validation_results: list[Any]) -> list[str]:
    return [
        str(item.get("message") or item.get("gate_reason"))
        for item in validation_results
        if isinstance(item, dict) and item.get("status") == "failed" and (item.get("message") or item.get("gate_reason"))
    ]


def _control_tower_plan_tables(plans: list[Any]) -> str:
    rows = []
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        rows.append(
            {
                "plan": plan.get("output_id"),
                "snapshot": plan.get("snapshot_id"),
                "status": plan.get("validation_status"),
                "blocked": plan.get("blocked"),
                "steps": ", ".join(
                    f"{step.get('capability_name')}:{step.get('action')}"
                    for step in _list(plan.get("steps"))
                    if isinstance(step, dict)
                ),
            }
        )
    return _dict_table(rows, ("plan", "snapshot", "status", "blocked", "steps"))


def _control_tower_step_table(steps: list[Any]) -> str:
    rows = [
        {
            "capability": step.get("capability_name"),
            "action": step.get("action"),
            "status": step.get("status"),
            "child_run": step.get("child_run_id"),
            "reused_run": step.get("reused_run_id"),
            "output": step.get("output_id") or step.get("reused_output_id"),
            "validation": step.get("validation_status"),
            "mode": step.get("execution_mode"),
            "state_changed": step.get("state_changed"),
            "rationale": step.get("rationale"),
        }
        for step in steps
        if isinstance(step, dict)
    ]
    return _dict_table(rows, ("capability", "action", "status", "mode", "child_run", "reused_run", "output", "validation", "state_changed", "rationale"))


def _control_tower_artifact_table(artifacts: list[Any]) -> str:
    rows = [
        {
            "artifact": artifact.get("artifact_type"),
            "producer": artifact.get("producer_workflow"),
            "run": artifact.get("run_id"),
            "output": artifact.get("output_id"),
            "validation": artifact.get("validation_status"),
            "compatibility": artifact.get("compatibility"),
            "freshness": artifact.get("freshness"),
            "confidence": _percent(artifact.get("confidence")),
        }
        for artifact in artifacts
        if isinstance(artifact, dict)
    ]
    return _dict_table(rows, ("artifact", "producer", "run", "output", "validation", "compatibility", "freshness", "confidence"))


def _control_tower_snapshot_table(snapshots: list[Any]) -> str:
    rows = [
        {
            "snapshot": snapshot.get("snapshot_id"),
            "artifacts": len(_list(snapshot.get("artifacts"))),
            "open_gates": len(_list(snapshot.get("open_gates"))),
            "missing_artifacts": ", ".join(_list(snapshot.get("missing_artifacts"))),
        }
        for snapshot in snapshots
        if isinstance(snapshot, dict)
    ]
    return _dict_table(rows, ("snapshot", "artifacts", "open_gates", "missing_artifacts"))


def _clinical_outcome_prediction_report(output: dict[str, Any]) -> str:
    trial = _dict(output.get("trial_identity"))
    asset = _dict(output.get("asset_identity"))
    endpoint = _dict(output.get("endpoint_risk_assessment"))
    enrollment = _dict(output.get("enrollment_duration_risk"))
    comparator = _dict(output.get("comparator_benchmarking"))
    pos = _dict(output.get("historical_pos_estimate"))
    approval = _dict(output.get("approval_likelihood_proxy"))
    failure = _dict(output.get("failure_mode_classification"))
    safety = _dict(output.get("safety_context"))
    source_availability = _dict(output.get("source_availability"))
    missing_flags = _list(output.get("missing_data_flags"))
    input_payload = _dict(output.get("input"))

    return "".join(
        [
            _hero(
                "Clinical Outcome Prediction",
                trial.get("brief_title") or trial.get("official_title") or output.get("run_id"),
                endpoint.get("rationale") or "Clinical risk and outcome-readiness assessment.",
                [
                    ("Run", output.get("run_id")),
                    ("NCT", trial.get("nct_id") or input_payload.get("nct_id")),
                    ("Asset", asset.get("asset_name")),
                    ("Indication", asset.get("normalized_indication") or ", ".join(_list(trial.get("conditions")))),
                    ("Validation", output.get("validation_status")),
                ],
            ),
            _kpi_grid(
                [
                    ("Approval Proxy", _percent(approval.get("probability")), approval.get("basis") or "Source-derived approval likelihood proxy."),
                    ("Historical PoS", _percent(pos.get("probability_of_success")), _display(pos.get("lookup_key") or pos.get("current_phase"))),
                    ("Endpoint Risk", endpoint.get("risk_level"), "Risk level from endpoint and design assessment."),
                    ("Enrollment Risk", enrollment.get("risk_level"), enrollment.get("rationale")),
                    ("Comparators", comparator.get("matched_public_trials_count"), comparator.get("landscape_summary")),
                    ("Missing Flags", len(missing_flags), "Data limitations that should be reviewed."),
                ]
            ),
            _section(
                "What It Assessed",
                _two_col(
                    _kv_table(
                        {
                            "nct_id": trial.get("nct_id") or input_payload.get("nct_id"),
                            "phase": ", ".join(_list(trial.get("phases"))),
                            "status": trial.get("overall_status"),
                            "sponsor": trial.get("sponsor"),
                            "conditions": ", ".join(_list(trial.get("conditions"))),
                        }
                    ),
                    _kv_table(
                        {
                            "asset_name": asset.get("asset_name"),
                            "modality": asset.get("modality"),
                            "therapeutic_area": asset.get("therapeutic_area"),
                            "intervention_type": asset.get("intervention_type"),
                            "asset_confidence": _percent(asset.get("confidence")),
                        }
                    ),
                ),
            ),
            _section(
                "Risk Rationale",
                _cards(
                    [
                        ("Endpoint Risk", _paragraphs([endpoint.get("rationale")]) + _bullets(_list(endpoint.get("risk_factors")))),
                        ("Enrollment Risk", _paragraphs([enrollment.get("rationale")])),
                        ("Safety Context", _paragraphs([safety.get("summary")]) + _flag_table(_list(safety.get("missing_data_flags")))),
                        ("Source Availability", _source_availability_table(_list(source_availability.get("flags")))),
                    ]
                ),
            ),
            _section("Failure Modes", _failure_mode_table(_list(failure.get("likely_failure_modes")))),
            _section(
                "Comparator Benchmarking",
                _paragraphs(
                    [
                        comparator.get("benchmark_summary"),
                        comparator.get("status_summary"),
                        comparator.get("phase_summary"),
                        comparator.get("endpoint_summary"),
                        comparator.get("population_summary"),
                    ]
                )
                + _dict_table(
                    [{"comparator_trial_id": item} for item in _list(comparator.get("comparator_trial_ids"))],
                    ("comparator_trial_id",),
                ),
            ),
            _section("Missing Data And Review Flags", _flag_table(missing_flags)),
        ]
    )


def _failure_mode_table(modes: list[Any]) -> str:
    rows = [
        {
            "category": item.get("category"),
            "severity": item.get("severity"),
            "rationale": item.get("rationale"),
            "sources": ", ".join(_list(item.get("source_ids"))),
        }
        for item in modes
        if isinstance(item, dict)
    ]
    return _dict_table(rows, ("category", "severity", "rationale", "sources"))


def _source_availability_table(flags: list[Any]) -> str:
    rows = [
        {
            "source": item.get("source_name"),
            "status": item.get("status"),
            "type": item.get("source_type"),
            "reason": item.get("reason"),
        }
        for item in flags
        if isinstance(item, dict)
    ]
    return _dict_table(rows, ("source", "status", "type", "reason"))


def _due_diligence_report(output: dict[str, Any]) -> str:
    asset = _dict(output.get("asset_identity"))
    target = _dict(output.get("target_trial") or output.get("trial"))
    memo = _dict(output.get("asset_memo"))
    pricing = _dict(output.get("pricing"))
    commercial = _dict(output.get("commercial_model"))
    rnpv = _dict(output.get("rnpv"))
    pos = _dict(output.get("pos"))
    clinical = _dict(output.get("clinical_risk_summary"))
    evidence = _dict(output.get("clinical_evidence"))
    landscape = _dict(output.get("competitive_landscape"))
    safety = _dict(output.get("safety_label_summary"))
    patent = _dict(output.get("patent_loe_review"))
    red_flags = _list(output.get("red_flags"))
    missing_flags = _list(output.get("missing_data_flags"))
    review_flags = top_review_flags_from_payload(output, limit=5)
    investment = _dict(output.get("investment_report")) or build_due_diligence_report_payload(output)

    title = memo.get("title") or f"Clinical-Stage Due Diligence Memo: {_display(asset.get('asset_name') or target.get('nct_id'))}"
    review_points = [item.get("reason") for item in red_flags[:6] if isinstance(item, dict)]
    review_points.extend(_list(memo.get("review_questions"))[:4])

    return "".join(
        [
            _hero(
                "Clinical-Stage Due Diligence Memo",
                title,
                memo.get("summary") or "Draft diligence artifact for human review.",
                [
                    ("Asset", asset.get("asset_name") or target.get("nct_id")),
                    ("Indication", asset.get("normalized_indication") or ", ".join(_list(target.get("conditions")))),
                    ("NCT", target.get("nct_id")),
                    ("Validation", output.get("validation_status")),
                ],
            ),
            _kpi_grid(
                [
                    ("Base rNPV", _money(rnpv.get("rnpv")), "Risk adjusted value after PoS, tax, margin, discounting, and development cost."),
                    ("Peak Net Sales", _money(commercial.get("peak_net_sales")), "Last deterministic launch-ramp year in the commercial forecast."),
                    ("PoS", _percent(pos.get("probability_of_success")), _display(pos.get("source_label") or "Workbook-derived probability of success.")),
                    ("Annual WAC", _money(pricing.get("annual_wac")), "Annualized from local WAC row plus openFDA dosing when available."),
                    ("LOE", _display(rnpv.get("loe_year") or patent.get("estimated_loe_year")), patent.get("review_summary")),
                    ("Confidence", _percent(output.get("confidence")), "Workflow-level confidence before human review."),
                ]
            ),
            _section("Panoptic-Style Investment Snapshot", _investment_snapshot_section(investment)),
            _section(
                "Investment Snapshot",
                _two_col(
                    _kv_table(
                        {
                            "asset_name": asset.get("asset_name"),
                            "sponsor": asset.get("sponsor") or _nested(target, "lead_sponsor", "name"),
                            "indication": asset.get("normalized_indication") or ", ".join(_list(target.get("conditions"))),
                            "therapeutic_area": asset.get("therapeutic_area"),
                            "phase": ", ".join(_list(target.get("phases"))),
                            "status": target.get("overall_status"),
                            "trial_title": target.get("brief_title") or target.get("official_title"),
                        }
                    ),
                    _bullets(review_points or ["No top review points were emitted."]),
                ),
            ),
            _section(
                "Evidence And Risk Context",
                _cards(
                    [
                        ("Clinical Risk", _kv_table({"endpoint_risk": clinical.get("endpoint_risk_level"), "enrollment_duration_risk": clinical.get("enrollment_duration_risk_level"), "historical_pos": _percent(clinical.get("historical_pos")), "approval_proxy": _percent(clinical.get("approval_likelihood_proxy"))})),
                        ("Clinical Evidence", _paragraphs([evidence.get("ctgov_summary"), f"PubMed query: {_display(evidence.get('pubmed_query'))}", f"PubMed articles: {_display(evidence.get('pubmed_article_count'))}"])),
                        ("Competitive Landscape", _paragraphs([landscape.get("benchmark_summary"), landscape.get("status_summary"), landscape.get("endpoint_summary")])),
                        ("Safety And IP", _paragraphs([f"openFDA label available: {_display(safety.get('label_available'))}", safety.get("warnings_summary"), patent.get("review_summary")])),
                    ]
                ),
            ),
            _section("Pricing Source Logic", _pricing_section(pricing)),
            _section("Deterministic Commercial Calculations", _commercial_section(commercial)),
            _section("Market Conversion Assumptions", _dict_table(_list(investment.get("market_conversion_assumptions")), ("conversion", "step", "base_fraction", "resulting_patients", "source"))),
            _section("Deterministic rNPV Calculation", _rnpv_section(rnpv, commercial)),
            _section("Sensitivity Summary", _dict_table(_list(investment.get("sensitivity_summary")), ("variable", "low_input", "low_case_rnpv", "base_case_rnpv", "high_input", "high_case_rnpv"))),
            _section(
                "Forecast Charts",
                _chart_specs_section(_list(investment.get("chart_specs"))) or _charts_section(commercial, rnpv),
            ),
            _section(
                "Human Review Surface",
                _cards(
                    [
                        ("Top Review Flags", _flag_table(list(review_flags))),
                        ("Memo Review Questions", _bullets(_list(memo.get("review_questions")) or ["None emitted."])),
                    ]
                ),
            ),
        ]
    )


def _protocol_design_report(output: dict[str, Any]) -> str:
    target = _dict(output.get("target_trial"))
    brief = _dict(output.get("protocol_design_brief"))
    intent = _dict(output.get("next_study_intent") or brief.get("next_study_intent"))
    benchmark = _dict(output.get("analog_benchmark_bundle"))
    follow_on_benchmark = _dict(output.get("follow_on_benchmark_bundle"))
    candidates = _list(output.get("analog_candidates"))
    selected_ids = set(_list(benchmark.get("selected_analog_ids")))
    selected_candidates = [item for item in candidates if _nested(item, "trial", "nct_id") in selected_ids]
    reviewer = _dict(brief.get("reviewer_critique"))

    return "".join(
        [
            _hero(
                "Agent 5 Protocol Design Brief",
                brief.get("title") or f"Draft Protocol Design Brief for {target.get('nct_id')}",
                brief.get("executive_synopsis", {}).get("body") if isinstance(brief.get("executive_synopsis"), dict) else "Draft strategy artifact for human review.",
                [
                    ("Target", target.get("nct_id")),
                    ("Status", target.get("overall_status")),
                    ("Phase", ", ".join(_list(target.get("phases")))),
                    ("Proposed Next Study", intent.get("proposed_next_stage")),
                    ("Human Review", _display(brief.get("requires_human_review"))),
                ],
            ),
            _kpi_grid(
                [
                    ("Next Study", intent.get("proposed_next_stage"), _display(intent.get("study_role"))),
                    ("Selected Analogs", len(_list(benchmark.get("selected_analog_ids"))), "CT.gov analog trials selected by Agent 5."),
                    ("Follow-on Trials", len(_list(output.get("follow_on_trials"))), "Lineage-confirmed follow-on trials found for selected analogs."),
                    ("Benchmark Confidence", _percent(benchmark.get("confidence")), "Confidence after deterministic analog coverage checks."),
                    ("Median Enrollment", _summary_value(benchmark.get("enrollment"), "median"), "Selected analog participant median."),
                    ("Median Duration", _summary_value(benchmark.get("planned_duration_months"), "median"), "Selected analog planned duration median."),
                ]
            ),
            _section("Next Study Intent", _next_study_intent_section(intent)),
            _section(
                "Target Trial",
                _kv_table(
                    {
                        "nct_id": target.get("nct_id"),
                        "title": target.get("brief_title") or target.get("official_title"),
                        "conditions": ", ".join(_list(target.get("conditions"))),
                        "interventions": ", ".join(_intervention_names(target)),
                        "sponsor": _nested(target, "lead_sponsor", "name"),
                        "enrollment": target.get("enrollment_count"),
                        "primary_completion": target.get("primary_completion_date"),
                    }
                ),
            ),
            _section("Direct Analog Benchmark", _analog_benchmark_section(benchmark)),
            _section("Analog Search Plan", _search_plan_section(_dict(benchmark.get("search_plan")))),
            _section("Selected Analog Trials", _selected_analogs_table(selected_candidates)),
            _section("Follow-on Lineage", _follow_on_lineage_section(output, follow_on_benchmark)),
            _section("Protocol Brief Sections", _protocol_sections(brief)),
            _section(
                "Review And Limitations",
                _cards(
                    [
                        ("Human Review Questions", _bullets(_list(brief.get("human_review_questions")) or ["None emitted."])),
                        ("Reviewer Missing Elements", _bullets(_list(reviewer.get("missing_elements")) or ["None emitted."])),
                        ("Statistical Questions", _bullets(_list(reviewer.get("statistical_questions")) or ["None emitted."])),
                        ("Regulatory Questions", _bullets(_list(reviewer.get("regulatory_questions")) or ["None emitted."])),
                    ]
                ),
            ),
        ]
    )


def _next_study_intent_section(intent: dict[str, Any]) -> str:
    if not intent:
        return "<p class='muted'>No next-study intent was emitted.</p>"
    missing_flags = [
        item.get("reason") if isinstance(item, dict) else item
        for item in _list(intent.get("missing_data_flags"))
    ]
    return "".join(
        [
            _kv_table(
                {
                    "evidence_anchor_nct_id": intent.get("evidence_anchor_nct_id"),
                    "current_development_stage": intent.get("current_development_stage"),
                    "proposed_next_stage": intent.get("proposed_next_stage"),
                    "study_role": intent.get("study_role"),
                    "development_objective": intent.get("development_objective"),
                    "key_clinical_question": intent.get("key_clinical_question"),
                    "indication": intent.get("indication"),
                    "target_population_context": intent.get("target_population_context"),
                    "regimen_context": intent.get("regimen_context"),
                    "confidence": _percent(intent.get("confidence")),
                    "requires_human_review": intent.get("requires_human_review"),
                }
            ),
            _cards(
                [
                    ("Rationale", _paragraphs([intent.get("rationale")])),
                    ("Alternatives Considered", _bullets(_list(intent.get("alternatives_considered")) or ["None emitted."])),
                    ("Missing Data Flags", _bullets(missing_flags or ["None emitted."])),
                ]
            ),
        ]
    )


def _pricing_section(pricing: dict[str, Any]) -> str:
    details = _dict(pricing.get("annualization_details"))
    base = _kv_table(
        {
            "matched_product": pricing.get("matched_product"),
            "wac_value": _money(pricing.get("wac_value")),
            "wac_unit_basis": pricing.get("wac_unit_basis"),
            "annual_wac": _money(pricing.get("annual_wac")),
            "confidence": _percent(pricing.get("confidence")),
            "source_ids": ", ".join(_list(pricing.get("source_ids"))),
        }
    )
    detail_table = _kv_table(details) if details else "<p class='muted'>No annualization details were available.</p>"
    formula = details.get("formula") or "annual_wac = WAC package price annualized by sourced dosing and package units."
    return _two_col(
        base + _paragraphs([pricing.get("dosing_summary")]),
        f"<p class='formula'>{escape(_display(formula))}</p>{detail_table}",
    )


def _commercial_section(commercial: dict[str, Any]) -> str:
    rows = _list(commercial.get("revenue_forecast"))
    assumptions = _list(commercial.get("assumptions"))
    ledger = _list(commercial.get("assumption_ledger"))
    population = _dict(commercial.get("selected_population_measure"))
    funnel = _dict(commercial.get("patient_funnel"))
    input_summary = _dict(commercial.get("commercial_input_bundle_summary"))
    denominator = _dict(input_summary.get("us_population_denominator"))
    query_diagnostics = _list(input_summary.get("market_query_diagnostics"))[:12]
    review_questions = _list(commercial.get("human_review_questions"))
    confidence_flags = _list(commercial.get("confidence_flags"))
    forecast_rows = []
    annual_patients = _float(commercial.get("annual_patients"))
    peak_penetration = _float(commercial.get("peak_penetration"))
    for row in rows:
        if not isinstance(row, dict):
            continue
        actual_penetration = None
        ramp_to_peak = None
        treated = _float(row.get("treated_patients"))
        if treated is not None and annual_patients:
            actual_penetration = treated / annual_patients
        if actual_penetration is not None and peak_penetration:
            ramp_to_peak = actual_penetration / peak_penetration
        forecast_rows.append(
            {
                "year": row.get("year"),
                "ramp_to_peak": _percent(ramp_to_peak),
                "actual_penetration": _percent(actual_penetration),
                "treated_patients": _patients(row.get("treated_patients")),
                "net_price": _money(row.get("net_price")),
                "net_revenue": _money(row.get("net_revenue")),
            }
        )
    ledger_rows = [
        {
            "assumption": item.get("assumption_name"),
            "base": _display(item.get("base") if item.get("base") is not None else item.get("value")),
            "source": item.get("source_type"),
            "review": _display(item.get("human_review_required")),
            "rationale": item.get("rationale"),
        }
        for item in ledger
        if isinstance(item, dict)
    ]
    return "".join(
        [
            _paragraphs(
                [
                    "Market sizing uses source-backed or reviewed population evidence plus AI-selected/defaulted funnel assumptions, then deterministic Python revenue math.",
                    "Deterministic formula: net_price = annual_wac * (1 - gross_to_net); treated_patients = commercially_addressable_patients * peak_penetration * launch_ramp_year; net_revenue = treated_patients * net_price.",
                    f"Calculable: {_display(commercial.get('calculable'))}. Peak net sales: {_money(commercial.get('peak_net_sales'))}.",
                    "If non-calculable, unresolved commercial inputs are listed below for human review.",
                ]
            ),
            _mini_heading("Commercial Market Sizing"),
            _two_col(
                _kv_table(
                    {
                        "selected_market_archetype": commercial.get("selected_market_archetype"),
                        "market_basis": commercial.get("market_basis"),
                        "population_value": _patients(population.get("value")),
                        "population_source": population.get("source_type"),
                        "population_review_required": population.get("human_review_required"),
                        "pricing_source": input_summary.get("pricing_source"),
                    }
                ),
                _kv_table(
                    {
                        "starting_population": _patients(funnel.get("starting_population")),
                        "diagnosed_patients": _patients(funnel.get("diagnosed_patients")),
                        "treated_or_managed": _patients(funnel.get("treated_or_managed_patients")),
                        "eligible_patients": _patients(funnel.get("eligible_patients")),
                        "commercially_addressable": _patients(funnel.get("commercially_addressable_patients")),
                    }
                ),
            ),
            _mini_heading("Market Evidence Diagnostics"),
            _two_col(
                _kv_table(
                    {
                        "us_population_total": _patients(denominator.get("total_us_population")),
                        "adult_population": _patients(denominator.get("adult_population")),
                        "pediatric_population": _patients(denominator.get("pediatric_population")),
                        "population_source": denominator.get("source_id"),
                        "population_source_type": denominator.get("source_type"),
                        "population_source_year": denominator.get("source_year"),
                        "population_review_required": denominator.get("human_review_required"),
                    }
                ),
                _dict_table(
                    [
                        {
                            "query": item.get("query"),
                            "status": item.get("status"),
                            "articles": item.get("article_count"),
                            "error": item.get("error_type") or item.get("error"),
                        }
                        for item in query_diagnostics
                        if isinstance(item, dict)
                    ],
                    ("query", "status", "articles", "error"),
                )
                if query_diagnostics
                else "<p class='muted'>No market-query diagnostics were available.</p>",
            ),
            _mini_heading("Commercial Assumption Ledger"),
            _dict_table(ledger_rows, ("assumption", "base", "source", "review", "rationale")),
            _mini_heading("Commercial Human Review"),
            _cards(
                [
                    ("Review Questions", _bullets(review_questions or ["None emitted."])),
                    ("Confidence And Missing Inputs", _bullets(confidence_flags or _list(input_summary.get("missing_inputs")) or ["None emitted."])),
                ]
            ),
            _mini_heading("Commercial Assumptions"),
            _assumptions_table(assumptions),
            _mini_heading("Revenue Forecast"),
            _dict_table(forecast_rows, ("year", "ramp_to_peak", "actual_penetration", "treated_patients", "net_price", "net_revenue")),
        ]
    )


def _rnpv_section(rnpv: dict[str, Any], commercial: dict[str, Any]) -> str:
    rows = _list(commercial.get("revenue_forecast"))
    launch_year = _int(rnpv.get("launch_year"))
    valuation_year = _assumption_value(rnpv, "valuation_year")
    tax_rate = _float(_assumption_value(rnpv, "tax_rate")) or 0.0
    operating_margin = _float(rnpv.get("operating_margin")) or 0.0
    pos = _float(rnpv.get("probability_of_success")) or 0.0
    discount = _float(rnpv.get("discount_rate")) or 0.0
    loe_year = _int(rnpv.get("loe_year"))
    valuation_year_int = _int(valuation_year) or 0
    cash_flow_rows = []
    for row in rows:
        if not isinstance(row, dict) or launch_year is None:
            continue
        calendar_year = launch_year + int(row.get("year") or 0) - 1
        if loe_year is not None and calendar_year > loe_year:
            continue
        years = max(0, calendar_year - valuation_year_int)
        net_revenue = _float(row.get("net_revenue")) or 0.0
        cash_flow = net_revenue * operating_margin * (1 - tax_rate) * pos
        discounted = cash_flow / ((1 + discount) ** years) if discount > -1 else cash_flow
        cash_flow_rows.append(
            {
                "calendar_year": calendar_year,
                "net_revenue": _money(net_revenue),
                "risk_adjusted_cash_flow": _money(cash_flow),
                "discount_years": years,
                "discounted_cash_flow": _money(discounted),
            }
        )
    return "".join(
        [
            _kpi_grid(
                [
                    ("rNPV", _money(rnpv.get("rnpv")), "Base deterministic value."),
                    ("Launch", rnpv.get("launch_year"), "Launch year assumption."),
                    ("LOE", rnpv.get("loe_year"), "Loss-of-exclusivity cutoff."),
                    ("Discount", _percent(rnpv.get("discount_rate")), "Annual discount rate."),
                ]
            ),
            _paragraphs(
                [
                    "Deterministic formula: rNPV = -development_cost + sum(net_revenue * operating_margin * (1 - tax_rate) * PoS / (1 + discount_rate)^years_since_valuation) through LOE.",
                    f"Development cost: {_money(rnpv.get('development_cost'))}; operating margin: {_percent(rnpv.get('operating_margin'))}; PoS: {_percent(rnpv.get('probability_of_success'))}.",
                ]
            ),
            _mini_heading("rNPV Assumptions"),
            _assumptions_table(_list(rnpv.get("assumptions"))),
            _mini_heading("Risk-Adjusted Cash Flow"),
            _dict_table(cash_flow_rows, ("calendar_year", "net_revenue", "risk_adjusted_cash_flow", "discount_years", "discounted_cash_flow")),
        ]
    )


def _charts_section(commercial: dict[str, Any], rnpv: dict[str, Any]) -> str:
    rows = _list(commercial.get("revenue_forecast"))
    if not rows:
        return "<p class='muted'>No commercial forecast was available for charts.</p>"
    return _two_col(_revenue_svg(rows), _rnpv_bar_svg(rnpv, commercial))


def _investment_snapshot_section(investment: dict[str, Any]) -> str:
    snapshot = _dict(investment.get("investment_snapshot"))
    flags = _list(investment.get("top_confidence_flags"))
    return _two_col(
        _kv_table(snapshot),
        _cards(
            [
                (
                    "Top Confidence Flags",
                    _bullets(
                        [
                            f"{item.get('severity')}: {item.get('message')}"
                            for item in flags
                            if isinstance(item, dict)
                        ]
                        or ["None emitted."]
                    ),
                )
            ]
        ),
    )


def _chart_specs_section(charts: list[Any]) -> str:
    if not charts:
        return ""
    blocks = []
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        blocks.append(
            "<div class='chart-spec'>"
            f"<h3>{escape(_display(chart.get('title')))}</h3>"
            f"{_render_investment_chart(chart)}"
            f"<p>{escape(_display(chart.get('notes')))}</p>"
            "</div>"
        )
    return "".join(blocks)


def _render_investment_chart(chart: dict[str, Any]) -> str:
    chart_id = chart.get("chart_id")
    if chart_id == "revenue_forecast":
        return _investment_revenue_line_chart(chart)
    if chart_id == "patient_funnel":
        return _investment_patient_funnel_chart(chart)
    if chart_id == "rnpv_sensitivity":
        return _investment_rnpv_tornado_chart(chart)
    return _investment_bar_chart(chart)


def _investment_revenue_line_chart(chart: dict[str, Any]) -> str:
    data = _list(chart.get("data"))
    series = (
        ("revenue", "Revenue", "#2563eb"),
        ("risk_adjusted_revenue", "Risk-adjusted", "#c2410c"),
    )
    values = [
        value
        for row in data
        for key, _, _ in series
        if isinstance(row, dict) and (value := _float(row.get(key))) is not None
    ]
    if not values:
        return "<p class='muted'>Not available.</p>"
    width, height = 820, 340
    left, right, top, bottom = 78, 28, 30, 58
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = _nice_max(max(values))
    steps = max(1, len(data) - 1)
    grid = []
    for index in range(5):
        value = max_value * index / 4
        y = top + plot_h - (value / max_value) * plot_h
        grid.append(
            f"<line class='grid' x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' />"
            f"<text class='axis' x='{left - 12}' y='{y + 4:.1f}' text-anchor='end'>{escape(_money_m(value))}</text>"
        )
    x_labels = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        x = left + (index / steps) * plot_w
        x_labels.append(f"<text class='axis' x='{x:.1f}' y='{height - 22}' text-anchor='middle'>{escape(_forecast_year_label(row))}</text>")
    paths = []
    final_labels = []
    legends = []
    for key, label, color in series:
        coords = []
        last_coord = None
        last_value = None
        for index, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            value = _float(row.get(key))
            if value is None:
                continue
            x = left + (index / steps) * plot_w
            y = top + plot_h - (value / max_value) * plot_h
            coords.append(f"{x:.1f},{y:.1f}")
            last_coord = (x, y)
            last_value = value
        if coords:
            paths.append(f"<polyline class='revenue-line' fill='none' stroke='{color}' points='{' '.join(coords)}' />")
            for coord in coords:
                x, y = coord.split(",")
                paths.append(f"<circle cx='{x}' cy='{y}' r='3.5' fill='{color}' />")
            legends.append(f"<span><i style='background:{color}'></i>{escape(label)}</span>")
        if last_coord and last_value is not None:
            final_labels.append(
                f"<text class='data-label' x='{last_coord[0] - 8:.1f}' y='{last_coord[1] - 8:.1f}' text-anchor='end'>{escape(label)}: {escape(_money_m(last_value))}</text>"
            )
    return (
        "<div class='chart-wrap'>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(_display(chart.get('title')))}'>"
        + "".join(grid)
        + f"<line class='axis-line' x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' />"
        + f"<line class='axis-line' x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' />"
        + "".join(paths)
        + "".join(final_labels)
        + "".join(x_labels)
        + "</svg>"
        f"<div class='legend'>{''.join(legends)}</div>"
        "</div>"
    )


def _investment_patient_funnel_chart(chart: dict[str, Any]) -> str:
    data = _list(chart.get("data"))
    values = [_float(row.get("patients")) for row in data if isinstance(row, dict)]
    values = [value for value in values if value is not None]
    if not values:
        return "<p class='muted'>Not available.</p>"
    width = 820
    row_h = 48
    top = 28
    left = 210
    right = 150
    height = top + row_h * len(data) + 30
    plot_w = width - left - right
    max_value = max(values) or 1.0
    rows = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        value = _float(row.get("patients"))
        if value is None:
            continue
        y = top + index * row_h
        bar_w = max(3, (value / max_value) * plot_w)
        conversion = _float(row.get("conversion_from_prior"))
        conversion_text = "" if conversion is None else f"{conversion:.0%} from prior"
        rows.append(f"<text class='funnel-label' x='{left - 12}' y='{y + 25}' text-anchor='end'>{escape(_display(row.get('population_step')))}</text>")
        rows.append(f"<rect class='funnel-bar' x='{left}' y='{y}' width='{bar_w:.1f}' height='30' rx='8' />")
        rows.append(f"<text class='data-label' x='{left + bar_w + 10:.1f}' y='{y + 20}'>{escape(_patients(value))}</text>")
        if conversion_text:
            rows.append(f"<text class='axis' x='{left + bar_w + 10:.1f}' y='{y + 37}'>{escape(conversion_text)}</text>")
    return (
        "<div class='chart-wrap'>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(_display(chart.get('title')))}'>"
        + "".join(rows)
        + "</svg>"
        "</div>"
    )


def _investment_rnpv_tornado_chart(chart: dict[str, Any]) -> str:
    rows = []
    for row in _list(chart.get("data")):
        if not isinstance(row, dict):
            continue
        low = _float(row.get("low_case_rnpv"))
        base = _float(row.get("base_case_rnpv"))
        high = _float(row.get("high_case_rnpv"))
        if low is None or base is None or high is None:
            continue
        rows.append(
            {
                "variable": row.get("variable"),
                "low": low,
                "base": base,
                "high": high,
                "swing": max(abs(low - base), abs(high - base)),
            }
        )
    if not rows:
        return "<p class='muted'>Not available.</p>"
    rows.sort(key=lambda row: row["swing"], reverse=True)
    base_value = rows[0]["base"]
    min_value = min(min(row["low"], row["high"], row["base"]) for row in rows)
    max_value = max(max(row["low"], row["high"], row["base"]) for row in rows)
    if min_value == max_value:
        min_value -= 1
        max_value += 1
    width = 820
    row_h = 52
    top = 42
    left = 190
    right = 82
    height = top + row_h * len(rows) + 42
    plot_w = width - left - right

    def x_for(value: float) -> float:
        return left + ((value - min_value) / (max_value - min_value)) * plot_w

    base_x = x_for(base_value)
    svg_rows = [
        f"<line class='grid emphasis' x1='{base_x:.1f}' y1='{top - 18}' x2='{base_x:.1f}' y2='{height - 34}' />",
        f"<text class='axis' x='{base_x:.1f}' y='{top - 24}' text-anchor='middle'>Base {escape(_money_m(base_value))}</text>",
    ]
    for index, row in enumerate(rows):
        y = top + index * row_h
        low_x = x_for(row["low"])
        high_x = x_for(row["high"])
        low_left = min(low_x, base_x)
        high_left = min(high_x, base_x)
        low_class = "tornado-better" if row["low"] > row["base"] else "tornado-worse"
        high_class = "tornado-better" if row["high"] > row["base"] else "tornado-worse"
        svg_rows.append(f"<text class='funnel-label' x='{left - 14}' y='{y + 23}' text-anchor='end'>{escape(_display(row.get('variable')))}</text>")
        svg_rows.append(f"<rect class='{low_class}' x='{low_left:.1f}' y='{y}' width='{abs(base_x - low_x):.1f}' height='20' rx='5' />")
        svg_rows.append(f"<rect class='{high_class}' x='{high_left:.1f}' y='{y + 22}' width='{abs(high_x - base_x):.1f}' height='20' rx='5' />")
        svg_rows.append(f"<text class='axis' x='{low_x:.1f}' y='{y - 4}' text-anchor='middle'>Low {escape(_money_m(row['low']))}</text>")
        svg_rows.append(f"<text class='axis' x='{high_x:.1f}' y='{y + 56}' text-anchor='middle'>High {escape(_money_m(row['high']))}</text>")
    return (
        "<div class='chart-wrap'>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(_display(chart.get('title')))}'>"
        + "".join(svg_rows)
        + "</svg>"
        "<div class='legend'><span><i style='background:#dc2626'></i>Worse rNPV outcome</span><span><i style='background:#16a34a'></i>Better rNPV outcome</span></div>"
        "</div>"
    )


def _investment_bar_chart(chart: dict[str, Any]) -> str:
    data = _list(chart.get("data"))
    y_keys = _list(chart.get("y_keys"))
    values = [
        value
        for row in data
        for key in y_keys
        if isinstance(row, dict) and (value := _float(row.get(key))) is not None
    ]
    if not values:
        return "<p class='muted'>Not available.</p>"
    width, height, left, top, bottom = 720, 300, 70, 24, 92
    plot_w, plot_h = width - left - 24, height - top - bottom
    max_value = max(abs(value) for value in values) or 1.0
    group_w = plot_w / max(1, len(data))
    bar_w = max(8, group_w / max(1, len(y_keys)) * 0.68)
    colors = ("#1f6f78", "#9a4f18", "#4e5f2f")
    bars = []
    labels = []
    zero_y = top + plot_h if min(values) >= 0 else top + plot_h / 2
    for group_index, row in enumerate(data):
        if not isinstance(row, dict):
            continue
        group_x = left + group_index * group_w + group_w * 0.15
        for key_index, key in enumerate(y_keys):
            value = _float(row.get(key))
            if value is None:
                continue
            x = group_x + key_index * (bar_w + 3)
            bar_h = min(plot_h, abs(value) / max_value * plot_h)
            y = zero_y - bar_h if value >= 0 else zero_y
            bars.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' fill='{colors[key_index % len(colors)]}' />")
        label = row.get(chart.get("x_key"))
        labels.append(f"<text x='{left + group_index * group_w + group_w / 2:.1f}' y='252' text-anchor='middle'>{escape(_truncate(str(label), 18))}</text>")
    legends = "".join(f"<span><i style='background:{colors[index % len(colors)]}'></i>{escape(str(key).replace('_', ' ').title())}</span>" for index, key in enumerate(y_keys))
    return (
        "<div class='chart-wrap'>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{escape(_display(chart.get('title')))}'>"
        f"<line x1='{left}' y1='{zero_y:.1f}' x2='{left + plot_w}' y2='{zero_y:.1f}' />"
        f"<text x='8' y='28'>{escape(_short_money(max_value))}</text>"
        + "".join(bars)
        + "".join(labels)
        + "</svg>"
        f"<div class='legend'>{legends}</div>"
        "</div>"
    )


def _analog_benchmark_section(benchmark: dict[str, Any]) -> str:
    return "".join(
        [
            _cards(
                [
                    ("Enrollment", _numeric_summary_table(_dict(benchmark.get("enrollment")))),
                    ("Planned Duration", _numeric_summary_table(_dict(benchmark.get("planned_duration_months")))),
                    ("Site Count", _numeric_summary_table(_dict(benchmark.get("site_count")))),
                    ("Benchmark Limitations", _bullets(_list(benchmark.get("limitations")) or ["None emitted."])),
                ]
            ),
            _mini_heading("Design Frequencies"),
            _two_col(
                _frequency_table("Randomization", benchmark.get("randomized_frequency")),
                _frequency_table("Blinding", benchmark.get("blinding_frequency")),
            ),
            _two_col(
                _frequency_table("Comparator Categories", benchmark.get("comparator_categories")),
                _frequency_table("Primary Endpoint Families", benchmark.get("primary_endpoint_family_frequency")),
            ),
            _two_col(
                _frequency_table("Countries", benchmark.get("country_distribution")),
                _frequency_table("Status", benchmark.get("status_distribution")),
            ),
        ]
    )


def _search_plan_section(search_plan: dict[str, Any]) -> str:
    queries = [
        {
            "query_id": query.get("query_id"),
            "condition": query.get("condition"),
            "intervention": query.get("intervention"),
            "phase": query.get("phase"),
            "analog_dimension": query.get("expected_analog_dimension"),
            "limit": query.get("limit"),
            "rationale": query.get("rationale"),
        }
        for query in _list(search_plan.get("queries"))
        if isinstance(query, dict)
    ]
    return "".join(
        [
            _paragraphs([search_plan.get("rationale_summary"), search_plan.get("guardrail_summary")]),
            _dict_table(queries, ("query_id", "condition", "intervention", "phase", "analog_dimension", "limit", "rationale")),
        ]
    )


def _selected_analogs_table(candidates: list[Any]) -> str:
    rows = []
    for candidate in candidates:
        trial = _dict(candidate.get("trial")) if isinstance(candidate, dict) else {}
        rows.append(
            {
                "nct_id": trial.get("nct_id"),
                "title": trial.get("brief_title") or trial.get("official_title"),
                "status": trial.get("overall_status"),
                "phase": ", ".join(_list(trial.get("phases"))),
                "enrollment": trial.get("enrollment_count"),
                "primary_completion": trial.get("primary_completion_date"),
                "queries": ", ".join(_list(candidate.get("query_ids"))) if isinstance(candidate, dict) else "",
            }
        )
    return _dict_table(rows, ("nct_id", "title", "status", "phase", "enrollment", "primary_completion", "queries"))


def _follow_on_lineage_section(output: dict[str, Any], follow_on_benchmark: dict[str, Any]) -> str:
    adjudications = _list(output.get("follow_on_adjudications"))
    follow_on_trials = _list(output.get("follow_on_trials"))
    adjudication_rows = [
        {
            "anchor_analog": item.get("anchor_analog_nct_id"),
            "status": item.get("status"),
            "selected_follow_ons": ", ".join(_list(item.get("selected_follow_on_nct_ids"))),
            "alternatives": ", ".join(_list(item.get("alternative_follow_on_nct_ids"))),
            "confidence": _percent(item.get("confidence")),
            "rationale": item.get("rationale"),
        }
        for item in adjudications
        if isinstance(item, dict)
    ]
    trial_rows = [
        {
            "nct_id": trial.get("nct_id"),
            "title": trial.get("brief_title") or trial.get("official_title"),
            "status": trial.get("overall_status"),
            "phase": ", ".join(_list(trial.get("phases"))),
            "sponsor": _nested(trial, "lead_sponsor", "name"),
            "start": trial.get("start_date"),
        }
        for trial in follow_on_trials
        if isinstance(trial, dict)
    ]
    if not adjudication_rows and not trial_rows:
        return _paragraphs(["No lineage-confirmed follow-on trials were emitted for the selected analog set."])
    return "".join(
        [
            _cards(
                [
                    (
                        "Follow-on Benchmark Limitations",
                        _bullets(_list(follow_on_benchmark.get("limitations")) or ["None emitted."]),
                    ),
                    (
                        "Follow-on Missing Data",
                        _flag_table(_list(follow_on_benchmark.get("missing_data_flags"))),
                    ),
                ]
            ),
            _mini_heading("Lineage Adjudications"),
            _dict_table(adjudication_rows, ("anchor_analog", "status", "selected_follow_ons", "alternatives", "confidence", "rationale")),
            _mini_heading("Fetched Follow-on Trials"),
            _dict_table(trial_rows, ("nct_id", "title", "status", "phase", "sponsor", "start")),
        ]
    )


def _protocol_sections(brief: dict[str, Any]) -> str:
    keys = (
        "executive_synopsis",
        "strategic_rationale",
        "analog_trial_benchmark_summary",
        "target_population",
        "study_design",
        "comparator_and_landscape_rationale",
        "endpoint_strategy",
        "draft_eligibility_framework",
        "draft_schedule_of_assessments_framework",
        "safety_monitoring_outline",
        "statistical_analysis_skeleton",
        "operational_feasibility_risks",
        "regulatory_standards_considerations",
    )
    cards = []
    for key in keys:
        section = _dict(brief.get(key))
        if section:
            title = section.get("title") or key.replace("_", " ").title()
            body = _paragraphs([section.get("body")])
            if section.get("source_ids"):
                body += f"<p class='sources'>Sources: {escape(', '.join(_list(section.get('source_ids'))))}</p>"
            cards.append((title, body))
    return _cards(cards)


def _human_readable_summary_section(output_json: Any) -> str:
    if not isinstance(output_json, dict):
        return _section("Human-Readable Module Summary", "<p class='muted'>None.</p>")
    summary = output_json.get("human_readable_summary")
    if not isinstance(summary, dict):
        return _section("Human-Readable Module Summary", "<p class='muted'>None.</p>")
    findings = summary.get("key_findings") if isinstance(summary.get("key_findings"), list) else []
    findings_html = "".join(
        "<li>"
        f"<strong>{escape(_display(finding.get('title') if isinstance(finding, dict) else 'Finding'))}</strong>: "
        f"{escape(_display(finding.get('detail') if isinstance(finding, dict) else finding))}"
        "</li>"
        for finding in findings
    )
    takeaways = summary.get("key_takeaways") if isinstance(summary.get("key_takeaways"), list) else []
    takeaways_html = "".join(f"<li>{escape(_display(item))}</li>" for item in takeaways)
    limitations = summary.get("limitations") if isinstance(summary.get("limitations"), list) else []
    limitations_html = "".join(f"<li>{escape(_display(item))}</li>" for item in limitations)
    return _section(
        "Human-Readable Module Summary",
        f"<h3>{escape(_display(summary.get('headline')))}</h3>"
        f"<p>{escape(_display(summary.get('plain_language_summary')))}</p>"
        f"<p><strong>Handoff:</strong> {escape(_display(summary.get('handoff_summary')))}</p>"
        f"<h4>Key Takeaways</h4><ul>{takeaways_html or '<li>None.</li>'}</ul>"
        f"<h4>Key Findings</h4><ul>{findings_html or '<li>None.</li>'}</ul>"
        f"<h4>Limitations</h4><ul>{limitations_html or '<li>None.</li>'}</ul>",
    )


def _runtime_trace_table(traces: tuple[Any, ...]) -> str:
    rows = [
        {
            "agent": getattr(trace, "agent_name", None),
            "route": getattr(trace, "model_route", None),
            "model": getattr(trace, "model", None),
            "mode": getattr(trace, "execution_mode", None),
            "retries": getattr(trace, "retry_count", 0),
            "fallback": getattr(trace, "fallback_cause", None),
        }
        for trace in traces
        if getattr(trace, "model", None) or getattr(trace, "model_route", None) or getattr(trace, "retry_count", 0)
    ]
    if not rows:
        return ""
    return _mini_heading("Runtime Routes And Retries") + _dict_table(rows, ("agent", "route", "model", "mode", "retries", "fallback"))


def _table_section(title: str, rows: tuple[Any, ...], fields: tuple[str, ...]) -> str:
    if not rows:
        return _section(title, "<p class='muted'>None.</p>")
    head = "".join(f"<th>{escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(_display(getattr(row, field, None)))}</td>" for field in fields) + "</tr>")
    return _section(title, f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>")


def _kv_table(values: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><th>{escape(_label(key))}</th><td>{escape(_display(value))}</td></tr>"
        for key, value in values.items()
    )
    return f"<table><tbody>{rows}</tbody></table>"


def _dict_table(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> str:
    if not rows:
        return "<p class='muted'>None.</p>"
    head = "".join(f"<th>{escape(_label(field))}</th>" for field in fields)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{escape(_display(row.get(field)))}</td>" for field in fields) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _label(value: Any) -> str:
    special = {
        "nct": "NCT",
        "id": "ID",
        "pos": "PoS",
        "rnpv": "rNPV",
        "wac": "WAC",
        "loe": "LOE",
        "ctgov": "CT.gov",
        "api": "API",
        "url": "URL",
    }
    return " ".join(special.get(part.casefold(), part.title()) for part in _display(value).replace("_", " ").replace("-", " ").split())


def _assumptions_table(assumptions: list[Any]) -> str:
    rows = [
        {
            "name": item.get("name"),
            "value": _display(item.get("value")),
            "unit": item.get("unit"),
            "type": item.get("assumption_type"),
            "provenance": item.get("provenance"),
        }
        for item in assumptions
        if isinstance(item, dict)
    ]
    return _dict_table(rows, ("name", "value", "unit", "type", "provenance"))


def _flag_table(flags: list[Any]) -> str:
    rows = [
        {
            "rating": item.get("rating") or item.get("severity"),
            "area": item.get("module") or item.get("category") or item.get("section") or item.get("target_id"),
            "reason": item.get("reason"),
        }
        for item in flags
        if isinstance(item, dict)
    ]
    return _dict_table(rows, ("rating", "area", "reason"))


def _numeric_summary_table(summary: dict[str, Any]) -> str:
    return _kv_table(
        {
            "observed_count": summary.get("observed_count"),
            "missing_count": summary.get("missing_count"),
            "mean": _number(summary.get("mean")),
            "median": _number(summary.get("median")),
            "minimum": _number(summary.get("minimum")),
            "maximum": _number(summary.get("maximum")),
            "iqr": _number(summary.get("iqr")),
            "unit": summary.get("unit"),
        }
    )


def _frequency_table(title: str, frequencies: Any) -> str:
    rows = [
        {"label": item.get("label"), "count": item.get("count"), "frequency": _percent(item.get("frequency"))}
        for item in _list(frequencies)
        if isinstance(item, dict)
    ]
    return f"<h4>{escape(title)}</h4>{_dict_table(rows, ('label', 'count', 'frequency'))}"


def _claims_cards(claims: tuple[Any, ...]) -> str:
    if not claims:
        return "<p class='muted'>None.</p>"
    cards = []
    for claim in claims[:10]:
        text = getattr(claim, "claim_text", None)
        confidence = getattr(claim, "confidence_level", None) or getattr(claim, "confidence", None)
        cards.append((_display(confidence), f"<p>{escape(_display(text))}</p><p class='sources'>{escape(_display(getattr(claim, 'source_ids', None)))}</p>"))
    return _cards(cards)


def _json_details(title: str, value: Any) -> str:
    return (
        f"<details><summary>{escape(title)}</summary>"
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


def _section(title: str, body: str) -> str:
    return f"<section><h2>{escape(title)}</h2>{body}</section>"


def _mini_heading(title: str) -> str:
    return f"<h3>{escape(title)}</h3>"


def _hero(eyebrow: str, title: Any, subtitle: Any, facts: list[tuple[str, Any]]) -> str:
    chips = "".join(f"<span><strong>{escape(label)}</strong>{escape(_display(value))}</span>" for label, value in facts if _display(value))
    return (
        "<section class='hero'>"
        f"<p class='eyebrow'>{escape(eyebrow)}</p>"
        f"<h2>{escape(_display(title))}</h2>"
        f"<p>{escape(_display(subtitle))}</p>"
        f"<div class='chips'>{chips}</div>"
        "</section>"
    )


def _kpi_grid(items: list[tuple[str, Any, Any]]) -> str:
    cards = []
    for label, value, detail in items:
        cards.append(
            "<div class='kpi'>"
            f"<span>{escape(label)}</span>"
            f"<strong>{escape(_display(value)) or 'NA'}</strong>"
            f"<p>{escape(_display(detail))}</p>"
            "</div>"
        )
    return f"<div class='kpis'>{''.join(cards)}</div>"


def _cards(items: list[tuple[Any, str]]) -> str:
    if not items:
        return "<p class='muted'>None.</p>"
    return "<div class='cards'>" + "".join(f"<article><h3>{escape(_display(title))}</h3>{body}</article>" for title, body in items) + "</div>"


def _two_col(left: str, right: str) -> str:
    return f"<div class='two-col'><div>{left}</div><div>{right}</div></div>"


def _paragraphs(values: list[Any]) -> str:
    html = "".join(f"<p>{escape(_display(value))}</p>" for value in values if _display(value))
    return html or "<p class='muted'>None.</p>"


def _bullets(values: list[Any]) -> str:
    return "<ul>" + "".join(f"<li>{escape(_display(value))}</li>" for value in values if _display(value)) + "</ul>"


def _revenue_svg(rows: list[Any]) -> str:
    points = []
    values = [_float(row.get("net_revenue")) or 0.0 for row in rows if isinstance(row, dict)]
    if not values:
        return "<p class='muted'>No revenue values available.</p>"
    maximum = max(values) or 1.0
    width, height = 520, 260
    left, top, graph_w, graph_h = 54, 24, 430, 180
    for index, value in enumerate(values):
        x = left + (graph_w * index / max(1, len(values) - 1))
        y = top + graph_h - (graph_h * value / maximum)
        points.append(f"{x:.1f},{y:.1f}")
    circles = "".join(f"<circle cx='{point.split(',')[0]}' cy='{point.split(',')[1]}' r='4'/>" for point in points)
    labels = "".join(
        f"<text x='{left + (graph_w * index / max(1, len(values) - 1)):.1f}' y='238'>{escape(str(row.get('year')))}</text>"
        for index, row in enumerate(rows)
        if isinstance(row, dict)
    )
    return (
        "<div class='chart'><h3>Commercial Forecast</h3>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Revenue forecast line chart'>"
        f"<line x1='{left}' y1='{top + graph_h}' x2='{left + graph_w}' y2='{top + graph_h}'/>"
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + graph_h}'/>"
        f"<polyline points='{' '.join(points)}'/>"
        f"{circles}{labels}"
        f"<text x='{left}' y='18'>Peak {_money(maximum)}</text>"
        "</svg></div>"
    )


def _rnpv_bar_svg(rnpv: dict[str, Any], commercial: dict[str, Any]) -> str:
    peak = _float(commercial.get("peak_net_sales")) or 0.0
    value = _float(rnpv.get("rnpv")) or 0.0
    development_cost = _float(rnpv.get("development_cost")) or 0.0
    items = [("Peak sales", peak), ("rNPV", value), ("Dev cost", -development_cost)]
    max_abs = max(abs(item[1]) for item in items) or 1.0
    bars = []
    for index, (label, amount) in enumerate(items):
        y = 40 + index * 58
        bar_w = 190 * abs(amount) / max_abs
        x = 250 if amount >= 0 else 250 - bar_w
        klass = "pos" if amount >= 0 else "neg"
        bars.append(
            f"<text x='20' y='{y + 18}'>{escape(label)}</text>"
            f"<rect class='{klass}' x='{x:.1f}' y='{y}' width='{bar_w:.1f}' height='28'/>"
            f"<text x='455' y='{y + 18}'>{escape(_money(amount))}</text>"
        )
    return (
        "<div class='chart'><h3>Value Summary</h3>"
        "<svg viewBox='0 0 520 230' role='img' aria-label='rNPV value summary bar chart'>"
        "<line x1='250' y1='24' x2='250' y2='205'/>"
        f"{''.join(bars)}"
        "</svg></div>"
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _intervention_names(target: dict[str, Any]) -> list[str]:
    names = []
    for item in _list(target.get("interventions")):
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def _assumption_value(section: dict[str, Any], name: str) -> Any:
    for item in _list(section.get("assumptions")):
        if isinstance(item, dict) and item.get("name") == name:
            return item.get("value")
    return None


def _summary_value(summary: Any, key: str) -> str:
    item = _dict(summary)
    value = item.get(key)
    unit = item.get("unit")
    return f"{_number(value)} {unit or ''}".strip() if value is not None else "NA"


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None


def _money(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "NA"
    sign = "-" if number < 0 else ""
    number = abs(number)
    if number >= 1_000_000_000:
        return f"{sign}${number / 1_000_000_000:.2f}B"
    if number >= 1_000_000:
        return f"{sign}${number / 1_000_000:.2f}M"
    if number >= 1_000:
        return f"{sign}${number / 1_000:.1f}K"
    return f"{sign}${number:,.2f}"


def _percent(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "NA"
    return f"{number * 100:.1f}%"


def _patients(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "NA"
    return f"{number:,.0f}"


def _number(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "NA"
    if abs(number) >= 100:
        return f"{number:,.0f}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def _money_m(value: float) -> str:
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{sign}${absolute / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{sign}${absolute / 1_000_000:.0f}M"
    if absolute >= 1_000:
        return f"{sign}${absolute / 1_000:.0f}K"
    return f"{sign}${absolute:.0f}"


def _short_money(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _forecast_year_label(row: dict[str, Any]) -> str:
    calendar_year = row.get("calendar_year")
    commercial_year = row.get("year")
    if calendar_year not in (None, "", "Not available") and commercial_year not in (None, "", "Not available"):
        return f"{calendar_year} (Y{commercial_year})"
    if calendar_year not in (None, "", "Not available"):
        return str(calendar_year)
    return f"Y{commercial_year}"


def _nice_max(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** (len(str(int(value))) - 1)
    scaled = value / magnitude
    if scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def _truncate(value: str, length: int) -> str:
    return value if len(value) <= length else value[: length - 1] + "..."


_CSS = """
:root{color-scheme:light;--ink:#18202a;--muted:#657282;--line:#dce3ea;--soft:#f5f7f9;--panel:#fff;--accent:#0e6f68;--warn:#a94f13}
*{box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.45;margin:0;background:#eef2f5;color:var(--ink)}
main{max-width:1280px;margin:0 auto;padding:28px}
h1{font-size:22px;margin:0 0 18px}
h2{font-size:22px;margin:0 0 14px}
h3{font-size:16px;margin:0 0 10px}
h4{font-size:14px;margin:12px 0 6px}
p{margin:0 0 10px}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:16px 0;padding:20px;box-shadow:0 1px 2px rgba(16,24,40,.04)}
.hero{background:#102a2c;color:#f8fbfb;border-color:#102a2c;padding:28px}
.hero h2{font-size:32px;line-height:1.1;margin:0 0 10px;max-width:900px}
.hero p{max-width:980px;color:#d8e5e3}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#9dd4cd!important;margin-bottom:8px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:16px}
.chips span{display:inline-flex;gap:8px;align-items:center;border:1px solid rgba(255,255,255,.22);border-radius:999px;padding:6px 10px;background:rgba(255,255,255,.06);font-size:13px}
.chips strong{color:#9dd4cd}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin:16px 0}
.kpi{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;min-height:116px}
.kpi span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.kpi strong{display:block;font-size:24px;line-height:1.1;margin:8px 0;color:#0f3433;overflow-wrap:anywhere}
.kpi p{font-size:12px;color:var(--muted);margin:0}
.two-col{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
article{border:1px solid var(--line);border-radius:8px;background:#fbfcfd;padding:14px}
table{border-collapse:collapse;width:100%;margin:10px 0;background:#fff;font-size:13px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:#f4f7f9;color:#334152;font-weight:650}
pre{background:#111827;color:#e5e7eb;border-radius:8px;padding:14px;overflow:auto;font-size:12px}
details{margin:12px 0}
summary{cursor:pointer;font-weight:650}
ul{margin:8px 0 0;padding-left:20px}
li{margin:4px 0}
.muted{color:var(--muted)}
.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#edf6f5;border:1px solid #cfe5e2;border-radius:6px;padding:10px;color:#164a47}
.sources{font-size:12px;color:var(--muted)}
.chart{border:1px solid var(--line);border-radius:8px;padding:14px;background:#fbfcfd}
.chart-spec{border:1px solid var(--line);border-radius:8px;padding:16px;background:#fbfcfd;margin:12px 0}
.chart-spec p{font-size:12px;color:var(--muted)}
.chart-wrap svg{width:100%;height:auto;display:block}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 0;font-size:12px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:5px}
.legend i{width:10px;height:10px;display:inline-block;border-radius:3px}
.grid{stroke:#dce3ea;stroke-width:1}
.grid.emphasis{stroke:#475569;stroke-width:1.2;stroke-dasharray:4 4}
.axis-line{stroke:#94a3b8;stroke-width:1}
.axis{fill:#64748b;font-size:11px}
.data-label{fill:#18202a;font-weight:650;font-size:11px}
.revenue-line{stroke-width:3.5;stroke-linecap:round;stroke-linejoin:round}
.funnel-label{fill:#18202a;font-size:12px;font-weight:650}
.funnel-bar{fill:#2563eb;opacity:.88}
.tornado-worse{fill:#dc2626;opacity:.8}
.tornado-better{fill:#16a34a;opacity:.8}
svg{width:100%;height:auto}
svg line{stroke:#9aa8b5;stroke-width:1}
svg polyline{fill:none;stroke:var(--accent);stroke-width:4;stroke-linecap:round;stroke-linejoin:round}
svg circle{fill:var(--accent)}
svg text{fill:#344255;font-size:12px}
svg rect.pos{fill:#0e6f68}
svg rect.neg{fill:#a94f13}
@media(max-width:980px){main{padding:16px}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.two-col,.cards{grid-template-columns:1fr}.hero h2{font-size:26px}}
@media(max-width:560px){.kpis{grid-template-columns:1fr}section{padding:14px}.hero{padding:20px}table{font-size:12px}}
"""


_CUMULATIVE_CSS = """
.cumulative-report{max-width:1160px;padding-bottom:72px}
.cumulative-report h1{font-size:30px}
.cumulative-report section{margin:18px 0;padding:22px}
.cumulative-report .hero{border-radius:10px;padding:30px;background:#0f2d2f}
.cumulative-report .hero h2{font-size:36px;letter-spacing:0;max-width:940px}
.cumulative-report .hero p{font-size:15px}
.cumulative-report .kpis{grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}
.cumulative-report .kpi{min-height:104px;border-radius:8px}
.workflow-head{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:18px;align-items:start;margin:-2px 0 16px;padding-bottom:14px;border-bottom:1px solid var(--line)}
.workflow-head p{margin-bottom:0}
.workflow-head .eyebrow{color:#0e6f68!important;margin-bottom:6px}
.workflow-body{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(360px,.8fr);gap:22px;align-items:start}
.workflow-body .cards{grid-template-columns:1fr;gap:12px}
.workflow-body .cards article{padding:16px 18px}
.metrics-panel table{margin-top:0}
.metrics-panel th{width:34%;min-width:150px}
.status-strip{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;max-width:430px}
.status-pill{display:inline-flex;align-items:center;gap:7px;border:1px solid #d4dde5;border-radius:999px;background:#f7fafb;color:#334152;padding:5px 9px;font-size:12px;white-space:nowrap}
.status-pill strong{color:#657282;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.status-pill.status-warn{border-color:#f3d6b6;background:#fff7ed;color:#8a3b0d}
.status-pill.status-ok{border-color:#cfe5e2;background:#edf8f6;color:#0f4a46}
.cumulative-report details{border:1px solid var(--line);border-radius:8px;background:#fbfcfd;padding:10px 12px}
.cumulative-report details[open]{padding-bottom:14px}
.cumulative-report summary{color:#334152}
.cumulative-report .sources{margin:10px 0 0}
.cumulative-report table{font-size:12.5px}
.cumulative-report .cards article{background:#fff}
@media(max-width:980px){.cumulative-report .kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.workflow-head,.workflow-body{grid-template-columns:1fr}.status-strip{justify-content:flex-start;max-width:none}}
@media(max-width:560px){.cumulative-report .kpis{grid-template-columns:1fr}.cumulative-report .hero h2{font-size:28px}.status-pill{white-space:normal}}
"""
