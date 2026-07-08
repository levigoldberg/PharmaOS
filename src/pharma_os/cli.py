"""Command-line interface for PharmaOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError
from dotenv import load_dotenv

from pharma_os.memory import DEFAULT_DB_PATH, MemoryStore
from pharma_os.orchestrator import Orchestrator
from pharma_os.report import build_report
from pharma_os.schemas import ClinicalTrialIntelligenceInput


def build_parser() -> argparse.ArgumentParser:
    """Build the PharmaOS CLI parser."""

    parser = argparse.ArgumentParser(prog="pharma_os")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow", help="Workflow name to run")
    run_parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite Scientific Memory path")
    run_parser.add_argument("--input-json", help="Workflow input JSON file")
    run_parser.add_argument("--output-json", help="Optional output JSON path")
    run_parser.add_argument("--disease", help="Disease or indication for trial_intelligence")
    run_parser.add_argument("--drug", help="Optional drug/intervention for trial_intelligence")
    run_parser.add_argument("--target", help="Optional target for trial_intelligence")
    run_parser.add_argument("--phase", help="Optional phase for trial_intelligence")
    run_parser.add_argument("--limit", type=int, default=10, help="Maximum records to retrieve")

    report_parser = subparsers.add_parser("report", help="Generate a run report")
    report_parser.add_argument("--run-id", required=True, help="Workflow run id")
    report_parser.add_argument("--db-path", default=DEFAULT_DB_PATH, help="SQLite Scientific Memory path")
    report_parser.add_argument("--output-json", help="Optional report JSON path")

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
            print(payload)
            return 0

        if args.command == "report":
            store = MemoryStore(args.db_path)
            report = build_report(args.run_id, memory=store)
            payload = report.model_dump_json()
            _write_output(args.output_json, payload)
            print(payload)
            return 0
    except (OSError, ValueError, ValidationError) as exc:
        print(f"error: {exc}")
        return 2
    return 1


def _workflow_input(args: argparse.Namespace) -> ClinicalTrialIntelligenceInput | None:
    if args.workflow != "trial_intelligence":
        return None
    if args.input_json:
        return ClinicalTrialIntelligenceInput.model_validate_json(
            Path(args.input_json).read_text(encoding="utf-8")
        )
    if not args.disease:
        raise ValueError("trial_intelligence requires --disease unless --input-json is supplied")
    return ClinicalTrialIntelligenceInput(
        disease=args.disease,
        drug=args.drug,
        target=args.target,
        phase=args.phase,
        limit=args.limit,
    )


def _write_output(path: str | None, payload: str) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload + "\n", encoding="utf-8")
