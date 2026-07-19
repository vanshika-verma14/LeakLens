"""Live terminal scorecard — one aligned row per Finding.

Pure ASCII on purpose: the target is a Windows cp1252 console that mangles unicode
(learnings 2026-07-16), so use `->` not arrows and keep decorations plain. `format_*`
returns a string (testable); the CLI does the printing.
"""
from leaklens.report import owasp

_HEADER = f"{'module':12s} {'verdict':13s} {'severity':9s} {'score':>6s}  {'owasp':38s} summary"
_RULE = "-" * len(_HEADER)


def _truncate(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "~"


def format_row(finding) -> str:
    return (f"{_truncate(finding.module, 12):12s} "
            f"{_truncate(finding.verdict, 13):13s} "
            f"{_truncate(finding.severity, 9):9s} "
            f"{finding.score:6.2f}  "
            f"{_truncate(owasp.primary(finding), 38):38s} "
            f"{_truncate(finding.summary, 60)}")


def format_scorecard(findings) -> str:
    """Render all findings as an ASCII table (header + rule + one row each)."""
    lines = ["LeakLens scorecard", "=" * len(_HEADER), _HEADER, _RULE]
    if not findings:
        lines.append("(no findings)")
    else:
        lines.extend(format_row(f) for f in findings)
    return "\n".join(lines)


def print_scorecard(findings) -> None:
    print(format_scorecard(findings))
