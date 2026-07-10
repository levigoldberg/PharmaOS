"""Command-line interface for PharmaOS."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pydantic import ValidationError
from dotenv import load_dotenv

from pharma_os.html_report import write_run_html
from pharma_os.memory import DEFAULT_DB_PATH, MemoryStore
from pharma_os.orchestrator import Orchestrator
from pharma_os.registry import WorkflowRegistry
from pharma_os.report import build_report
from pharma_os.request_understanding import RequestUnderstandingError, understand_orchestration_goal
from pharma_os.schemas import ClinicalOutcomePredictionInput, ClinicalTrialIntelligenceInput, DueDiligenceInput, OrchestrationRequest, ProtocolDesignInput


NCT_RE = re.compile(r"^NCT\d{8}$", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    """Build the PharmaOS CLI parser."""

    parser = argparse.ArgumentParser(prog="pharma_os")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow", help="Workflow name to run")
    run_parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite Scientific Memory path")
    run_parser.add_argument("--input-json", help="Workflow input JSON file")
    run_parser.add_argument("--output-json", help="Optional output JSON path")
    run_parser.add_argument("--output-html", help="Optional run HTML viewer output path")
    run_parser.add_argument("--disease", help="Disease or indication for Agent 3 trial-landscape mode")
    run_parser.add_argument("--drug", help="Optional drug/intervention for Agent 3 trial-landscape mode")
    run_parser.add_argument("--target", help="Optional target for Agent 3 trial-landscape mode")
    run_parser.add_argument("--phase", help="Optional phase for Agent 3 trial-landscape mode")
    run_parser.add_argument("--limit", type=int, default=10, help="Maximum records to retrieve")
    run_parser.add_argument("--nct-id", help="NCT ID for due_diligence or clinical_outcome_prediction")
    run_parser.add_argument("--pos-workbook-path", help="Optional PoS workbook path")
    run_parser.add_argument("--wac-data-path", help="Optional WAC workbook path for due_diligence")
    run_parser.add_argument("--annual-patients", type=float, help="Reviewed annual eligible patient assumption")
    run_parser.add_argument("--peak-penetration", type=float, help="Reviewed peak penetration assumption")
    run_parser.add_argument("--gross-to-net", type=float, help="Reviewed gross-to-net assumption")
    run_parser.add_argument("--operating-margin", type=float, help="Reviewed operating margin assumption")
    run_parser.add_argument("--discount-rate", type=float, help="Reviewed discount rate assumption")
    run_parser.add_argument("--development-cost", type=float, help="Reviewed remaining development cost assumption")
    run_parser.add_argument("--launch-year", type=int, help="Reviewed expected launch year")
    run_parser.add_argument("--loe-year", type=int, help="Reviewed expected loss-of-exclusivity year")
    run_parser.add_argument("--refresh-agent3", action="store_true", help="Force due_diligence to generate a fresh Agent 3 handoff")
    run_parser.add_argument("--refresh-agent4", action="store_true", help="Force protocol_design to generate a fresh Agent 4 handoff")
    run_parser.add_argument("--analog-top-k", type=int, default=10, help="Maximum analog trials selected for protocol_design")

    orchestrate_parser = subparsers.add_parser("orchestrate", help="Run the memory-aware Control Tower orchestration loop")
    orchestrate_parser.add_argument("--goal", help="Control Tower objective")
    orchestrate_parser.add_argument("--nct-id", help="Optional ClinicalTrials.gov NCT identifier")
    orchestrate_parser.add_argument("--asset-name", help="Optional asset name")
    orchestrate_parser.add_argument("--indication", help="Optional indication")
    orchestrate_parser.add_argument("--input-json", help="Optional OrchestrationRequest JSON file")
    orchestrate_parser.add_argument("--force-refresh", action="append", default=(), help="Capability or artifact to refresh; may be repeated")
    orchestrate_parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite Scientific Memory path")
    orchestrate_parser.add_argument("--output-json", help="Optional output JSON path")
    orchestrate_parser.add_argument("--output-html", help="Optional Control Tower HTML report output path")
    orchestrate_parser.add_argument("--pos-workbook-path", help="Optional PoS workbook path")
    orchestrate_parser.add_argument("--wac-data-path", help="Optional WAC workbook path")
    orchestrate_parser.add_argument("--annual-patients", type=float, help="Reviewed annual eligible patient assumption")
    orchestrate_parser.add_argument("--peak-penetration", type=float, help="Reviewed peak penetration assumption")
    orchestrate_parser.add_argument("--gross-to-net", type=float, help="Reviewed gross-to-net assumption")
    orchestrate_parser.add_argument("--operating-margin", type=float, help="Reviewed operating margin assumption")
    orchestrate_parser.add_argument("--discount-rate", type=float, help="Reviewed discount rate assumption")
    orchestrate_parser.add_argument("--development-cost", type=float, help="Reviewed remaining development cost assumption")
    orchestrate_parser.add_argument("--launch-year", type=int, help="Reviewed expected launch year")
    orchestrate_parser.add_argument("--loe-year", type=int, help="Reviewed expected loss-of-exclusivity year")

    report_parser = subparsers.add_parser("report", help="Generate a run report")
    report_parser.add_argument("--run-id", required=True, help="Workflow run id")
    report_parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite Scientific Memory path")
    report_parser.add_argument("--output-json", help="Optional report JSON path")
    report_parser.add_argument("--output-html", help="Optional run HTML viewer output path")

    view_parser = subparsers.add_parser("view", help="Generate a simple HTML run viewer")
    view_parser.add_argument("--run-id", required=True, help="Workflow run id")
    view_parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite Scientific Memory path")
    view_parser.add_argument("--output-html", required=True, help="HTML output path")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PharmaOS CLI."""

    load_dotenv(dotenv_path=Path(".env"))
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            store = MemoryStore(args.db_path)
            input_data = _workflow_input(args)
            result = Orchestrator(memory=store).run(args.workflow, input_data)
            payload = result.model_dump_json() if hasattr(result, "model_dump_json") else json.dumps(result)
            _write_output(args.output_json, payload)
            if args.output_html:
                run_id = getattr(result, "run_id", None)
                if not run_id:
                    raise ValueError("run output does not include run_id for HTML generation")
                write_run_html(run_id, args.output_html, memory=store)
            print(payload)
            return 0

        if args.command == "orchestrate":
            store = MemoryStore(args.db_path)
            request = _orchestration_request(args)
            result = Orchestrator(memory=store).orchestrate(request)
            payload = result.model_dump_json()
            output_json, output_html = _orchestration_output_paths(args, result.run_id)
            _write_output(output_json, payload)
            write_run_html(result.run_id, output_html, memory=store)
            print(payload)
            return 0

        if args.command == "report":
            store = MemoryStore(args.db_path)
            report = build_report(args.run_id, memory=store)
            payload = report.model_dump_json()
            _write_output(args.output_json, payload)
            if args.output_html:
                write_run_html(args.run_id, args.output_html, memory=store)
            print(payload)
            return 0
        if args.command == "view":
            store = MemoryStore(args.db_path)
            output_path = write_run_html(args.run_id, args.output_html, memory=store)
            print(json.dumps({"run_id": args.run_id, "output_html": str(output_path)}))
            return 0
    except (OSError, RuntimeError, ValueError, ValidationError, RequestUnderstandingError) as exc:
        print(f"error: {exc}")
        return 2
    return 1


def _workflow_input(
    args: argparse.Namespace,
) -> ClinicalTrialIntelligenceInput | DueDiligenceInput | ClinicalOutcomePredictionInput | ProtocolDesignInput | None:
    if args.workflow == "trial_intelligence":
        if args.input_json:
            return ClinicalTrialIntelligenceInput.model_validate_json(
                Path(args.input_json).read_text(encoding="utf-8")
            )
        if not args.disease:
            raise ValueError("trial_intelligence Agent 3 landscape mode requires --disease unless --input-json is supplied")
        return ClinicalTrialIntelligenceInput(
            disease=args.disease,
            drug=args.drug,
            target=args.target,
            phase=args.phase,
            limit=args.limit,
        )
    if args.workflow == "due_diligence":
        if args.input_json:
            return DueDiligenceInput.model_validate_json(
                Path(args.input_json).read_text(encoding="utf-8")
            )
        if not args.nct_id:
            raise ValueError("due_diligence requires --nct-id unless --input-json is supplied")
        return DueDiligenceInput(
            nct_id=args.nct_id,
            pos_workbook_path=args.pos_workbook_path,
            wac_data_path=args.wac_data_path,
            annual_patients=args.annual_patients,
            peak_penetration=args.peak_penetration,
            gross_to_net=args.gross_to_net,
            operating_margin=args.operating_margin,
            discount_rate=args.discount_rate,
            development_cost=args.development_cost,
            launch_year=args.launch_year,
            loe_year=args.loe_year,
            refresh_agent3=args.refresh_agent3,
        )
    if args.workflow == "clinical_outcome_prediction":
        if args.input_json:
            return ClinicalOutcomePredictionInput.model_validate_json(
                Path(args.input_json).read_text(encoding="utf-8")
            )
        if not args.nct_id:
            raise ValueError("clinical_outcome_prediction requires --nct-id unless --input-json is supplied")
        return ClinicalOutcomePredictionInput(
            nct_id=args.nct_id,
            pos_workbook_path=args.pos_workbook_path,
        )
    if args.workflow == "protocol_design":
        if args.input_json:
            return ProtocolDesignInput.model_validate_json(
                Path(args.input_json).read_text(encoding="utf-8")
            )
        if not args.nct_id:
            raise ValueError("protocol_design requires --nct-id unless --input-json is supplied")
        return ProtocolDesignInput(
            nct_id=args.nct_id,
            pos_workbook_path=args.pos_workbook_path,
            wac_data_path=args.wac_data_path,
            annual_patients=args.annual_patients,
            peak_penetration=args.peak_penetration,
            gross_to_net=args.gross_to_net,
            operating_margin=args.operating_margin,
            discount_rate=args.discount_rate,
            development_cost=args.development_cost,
            launch_year=args.launch_year,
            loe_year=args.loe_year,
            refresh_agent3=args.refresh_agent3,
            refresh_agent4=args.refresh_agent4,
            analog_top_k=args.analog_top_k,
        )
    if args.input_json:
        raise ValueError(f"{args.workflow} does not define an input schema")
    return None


def _orchestration_request(args: argparse.Namespace) -> OrchestrationRequest:
    if args.input_json:
        return OrchestrationRequest.model_validate_json(
            Path(args.input_json).read_text(encoding="utf-8")
        )
    if not args.goal:
        raise ValueError("orchestrate requires --goal unless --input-json is supplied")
    explicit_assumptions = {
        key: value
        for key, value in {
            "pos_workbook_path": args.pos_workbook_path,
            "wac_data_path": args.wac_data_path,
            "annual_patients": args.annual_patients,
            "peak_penetration": args.peak_penetration,
            "gross_to_net": args.gross_to_net,
            "operating_margin": args.operating_margin,
            "discount_rate": args.discount_rate,
            "development_cost": args.development_cost,
            "launch_year": args.launch_year,
            "loe_year": args.loe_year,
        }.items()
        if value is not None
    }
    explicit_nct = _normalize_nct(args.nct_id, field_name="--nct-id") if args.nct_id else None
    if _requires_ai_request_understanding(args):
        registry = WorkflowRegistry.default()
        parsed = understand_orchestration_goal(
            goal=args.goal,
            explicit_fields={
                "nct_id": explicit_nct,
                "asset_name": args.asset_name,
                "indication": args.indication,
                "assumptions": explicit_assumptions,
                "force_refresh": tuple(args.force_refresh or ()),
            },
            registry=registry,
        )
        return _request_from_understanding(
            goal=args.goal,
            parsed=parsed,
            explicit_nct=explicit_nct,
            explicit_asset_name=args.asset_name,
            explicit_indication=args.indication,
            explicit_assumptions=explicit_assumptions,
            explicit_force_refresh=tuple(args.force_refresh or ()),
            registry=registry,
        )
    return OrchestrationRequest(
        objective=args.goal,
        nct_id=explicit_nct,
        asset_name=args.asset_name,
        indication=args.indication,
        assumptions=explicit_assumptions,
        force_refresh=tuple(args.force_refresh or ()),
    )


def _requires_ai_request_understanding(args: argparse.Namespace) -> bool:
    return not any((args.nct_id, args.asset_name, args.indication))


def _request_from_understanding(
    *,
    goal: str,
    parsed: object,
    explicit_nct: str | None,
    explicit_asset_name: str | None,
    explicit_indication: str | None,
    explicit_assumptions: dict[str, object],
    explicit_force_refresh: tuple[str, ...],
    registry: WorkflowRegistry,
) -> OrchestrationRequest:
    nct_id = _normalize_nct(getattr(parsed, "nct_id", None), field_name="AI-extracted nct_id") if getattr(parsed, "nct_id", None) else None
    if explicit_nct and nct_id and explicit_nct != nct_id:
        raise ValueError(f"--nct-id {explicit_nct} conflicts with AI-extracted NCT ID {nct_id}.")
    resolved_nct = explicit_nct or nct_id
    target_capability = getattr(parsed, "target_capability", None)
    decision_type = getattr(parsed, "decision_type", None)
    capability = registry.get(target_capability) if target_capability else None
    missing = tuple(getattr(parsed, "missing_required_fields", ()) or ())
    questions = tuple(getattr(parsed, "clarifying_questions", ()) or ())
    confidence = float(getattr(parsed, "confidence", 0.0) or 0.0)
    executable_target = capability is not None and capability.executable and capability.implementation_status == "implemented"
    if executable_target and not resolved_nct:
        missing = tuple(dict.fromkeys((*missing, "nct_id")))
        questions = tuple(dict.fromkeys((*questions, "Which ClinicalTrials.gov NCT ID should PharmaOS use?")))
    if (confidence < 0.6 or missing or questions) and (capability is None or executable_target):
        details = []
        if confidence < 0.6:
            details.append(f"AI request understanding confidence is {confidence:.2f}.")
        if missing:
            details.append(f"Missing required fields: {', '.join(missing)}.")
        if questions:
            details.append("Clarifying questions: " + " ".join(questions))
        raise ValueError("Cannot safely orchestrate from the goal. " + " ".join(details))

    assumptions = {
        **(getattr(parsed, "assumptions", {}) or {}),
        **explicit_assumptions,
    }
    identifiers = {
        "request_understanding": "ai",
    }
    if target_capability:
        identifiers["target_capability"] = str(target_capability)
    skip_capabilities = tuple(getattr(parsed, "skip_capabilities", ()) or ())
    if skip_capabilities:
        identifiers["skip_capabilities"] = ",".join(str(item) for item in skip_capabilities)
    requested_outputs = tuple(getattr(parsed, "requested_outputs", ()) or ())
    if requested_outputs:
        identifiers["requested_outputs"] = ",".join(str(item) for item in requested_outputs)

    parsed_force_refresh = tuple(str(item) for item in (getattr(parsed, "force_refresh", ()) or ()))
    normalized_objective = str(getattr(parsed, "normalized_objective", None) or goal)
    return OrchestrationRequest(
        objective=normalized_objective,
        nct_id=resolved_nct,
        asset_name=explicit_asset_name or getattr(parsed, "asset_name", None),
        indication=explicit_indication or getattr(parsed, "indication", None),
        identifiers=identifiers,
        assumptions=assumptions,
        force_refresh=tuple(dict.fromkeys((*explicit_force_refresh, *parsed_force_refresh))),
        decision_type=decision_type if decision_type != "unknown" else None,
    )


def _normalize_nct(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if not NCT_RE.match(normalized):
        raise ValueError(f"{field_name} must be a valid ClinicalTrials.gov identifier like NCT12345678.")
    return normalized


def _orchestration_output_paths(args: argparse.Namespace, run_id: str) -> tuple[str, str]:
    output_json = args.output_json or str(Path("outputs") / f"control_tower_orchestration_{run_id}.json")
    output_html = args.output_html or str(Path("outputs") / f"control_tower_orchestration_{run_id}.html")
    return output_json, output_html


def _write_output(path: str | None, payload: str) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
