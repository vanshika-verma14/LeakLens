"""Persisted JSON report — the machine-readable Finding dump (FR-3).

Relies on Finding's str-mixin enums: `dataclasses.asdict` + `json.dumps` serialize verdict
and severity as plain strings with no custom encoder (guarded by test_finding.py). Each
finding is enriched with its OWASP categories so the JSON is self-describing.
"""
import dataclasses
import json
from pathlib import Path

from leaklens.report import owasp


def build_report(findings) -> dict:
    """Return the report as a plain dict: {"findings": [...]}, JSON-ready."""
    return {
        "findings": [
            {**dataclasses.asdict(f), "owasp_categories": owasp.categories_for(f)}
            for f in findings
        ]
    }


def write_json_report(findings, path) -> Path:
    """Write the JSON report to `path` (utf-8, pretty) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(build_report(findings), indent=2, ensure_ascii=False)
    path.write_text(payload, encoding="utf-8")
    return path
