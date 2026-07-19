"""`leaklens scan --config config.yaml` — the developer-facing entry point.

Thin glue over pieces already built and tested: load+validate the config (with the
ownership gate), run the enabled modules through the isolating runner, print the terminal
scorecard, and persist the JSON + HTML reports. No audit logic lives here.
"""
import leaklens._compat  # noqa: F401  — MUST be first: resource stub + CPU pin for vec2text

import argparse
import sys
from pathlib import Path

from leaklens.config import ConfigError, load_config
from leaklens.report import html_report, json_report
from leaklens.report.scorecard import print_scorecard
from leaklens.runner import run_scan

JSON_NAME = "report.json"
HTML_NAME = "report.html"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leaklens",
        description="Audit RAG infrastructure for embedding-inversion / cache leakage.")
    sub = parser.add_subparsers(dest="command")
    scan = sub.add_parser("scan", help="run the audit against a configured target")
    scan.add_argument("--config", required=True, type=Path,
                      help="path to config.yaml (see config.example.yaml)")
    return parser


def cmd_scan(args) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    findings = run_scan(cfg)
    print_scorecard(findings)

    out_dir = Path(cfg.options.output_dir)
    json_path = json_report.write_json_report(findings, out_dir / JSON_NAME)
    html_path = html_report.write_html_report(findings, out_dir / HTML_NAME)
    print(f"\nwrote {json_path}")
    print(f"wrote {html_path}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
