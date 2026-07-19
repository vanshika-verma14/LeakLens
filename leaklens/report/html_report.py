"""Persisted HTML report — one self-contained file, no framework, no external assets.

Everything that comes from a Finding is passed through `html.escape` before it reaches the
page: evidence carries recovered text that can contain `<`, `&`, or markup, and this
report is meant to be opened in a browser. Inline CSS only, so the file works when moved.
"""
import html
from pathlib import Path

from leaklens.report import owasp

_VERDICT_CLASS = {"LEAK": "leak", "NO_LEAK": "clear", "INCONCLUSIVE": "inconclusive"}

_STYLE = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }
h1 { margin-bottom: .25rem; }
.finding { border: 1px solid #8884; border-radius: 8px; padding: 1rem 1.25rem;
           margin: 1rem 0; }
.badge { display: inline-block; padding: .1rem .55rem; border-radius: 999px;
         font-weight: 600; font-size: .85rem; }
.leak { background: #d33; color: #fff; }
.clear { background: #2a2; color: #fff; }
.inconclusive { background: #888; color: #fff; }
.meta { color: #8a8a8a; font-size: .9rem; margin: .35rem 0; }
.remediation { background: #8881; padding: .5rem .75rem; border-radius: 6px; }
table.evidence { border-collapse: collapse; margin-top: .5rem; font-size: .9rem; }
table.evidence td, table.evidence th { border: 1px solid #8884; padding: .25rem .5rem;
                                       text-align: left; vertical-align: top; }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _evidence_table(evidence) -> str:
    """Render a list of evidence items generically — no assumption about their keys."""
    rows = []
    for item in evidence:
        if isinstance(item, dict):
            cells = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>"
                            for k, v in item.items())
            rows.append(f"<table class='evidence'>{cells}</table>")
        else:
            rows.append(f"<p>{_esc(item)}</p>")
    return "".join(rows)


def _finding_html(finding) -> str:
    cls = _VERDICT_CLASS.get(str(finding.verdict), "inconclusive")
    cats = ", ".join(_esc(c) for c in owasp.categories_for(finding))
    parts = [
        "<section class='finding'>",
        f"<h2>{_esc(finding.module)} "
        f"<span class='badge {cls}'>{_esc(finding.verdict)}</span></h2>",
        f"<p class='meta'>severity {_esc(finding.severity)} &middot; "
        f"score {_esc(f'{finding.score:.4f}')} &middot; OWASP: {cats}</p>",
        f"<p>{_esc(finding.summary)}</p>",
    ]
    if finding.remediation:
        parts.append(f"<p class='remediation'><strong>Remediation:</strong> "
                     f"{_esc(finding.remediation)}</p>")
    if finding.evidence:
        parts.append("<details><summary>Evidence</summary>"
                     f"{_evidence_table(finding.evidence)}</details>")
    parts.append("</section>")
    return "".join(parts)


def render_html(findings) -> str:
    """Return a complete, self-contained HTML document for the findings."""
    body = "".join(_finding_html(f) for f in findings) or "<p>No findings.</p>"
    return (
        "<!doctype html>\n<html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>LeakLens report</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<h1>LeakLens report</h1>"
        "<p class='meta'>Infrastructure-layer leakage audit.</p>"
        f"{body}</body></html>\n"
    )


def write_html_report(findings, path) -> Path:
    """Write the HTML report to `path` (utf-8) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(findings), encoding="utf-8")
    return path
