"""Command-line interface for PharmaOS."""

from __future__ import annotations

import argparse

from pharma_os.orchestrator import Orchestrator
from pharma_os.report import build_report


def build_parser() -> argparse.ArgumentParser:
    """Build the PharmaOS CLI parser."""

    parser = argparse.ArgumentParser(prog="pharma_os")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a workflow")
    run_parser.add_argument("workflow", help="Workflow name to run")

    report_parser = subparsers.add_parser("report", help="Generate a run report")
    report_parser.add_argument("--run-id", required=True, help="Workflow run id")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the PharmaOS CLI."""

    args = build_parser().parse_args(argv)

    if args.command == "run":
        run = Orchestrator().run(args.workflow)
        print(run.model_dump_json())
        return 0

    if args.command == "report":
        report = build_report(args.run_id)
        print(report.model_dump_json())
        return 0

    return 1
