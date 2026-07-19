"""Tests for the report layer — fast, no model.

Covers the OWASP mapping and that a Finding renders to each format. The escaping test is
load-bearing (evidence contains attacker-influenced recovered text), as is the mixed-list
test that proves the renderers never special-case a module.
"""
import json

from leaklens.finding import Finding, Severity, Verdict
from leaklens.report import html_report, json_report, owasp, scorecard
from leaklens.report.owasp import LLM06, LLM08, UNCATEGORIZED


def leak_finding(**over):
    kwargs = dict(
        module="inversion", verdict=Verdict.LEAK, score=0.82, severity=Severity.HIGH,
        owasp=LLM08, summary="recovered key entities at mean recall 0.82",
        evidence=[{"id": "pii-001", "recall": 1.0,
                   "original": "Email Priya Sharma", "recovered": "email priya sharma"}],
        remediation="add gaussian noise to stored embeddings",
        details={"per_category_recall": {"pii": 0.6}},
    )
    kwargs.update(over)
    return Finding(**kwargs)


def inconclusive_finding():
    return Finding(module="inversion", verdict=Verdict.INCONCLUSIVE, score=0.0,
                   severity=Severity.LOW, owasp=LLM08,
                   summary="no inverter available for 'openai/text-embedding-3-small'")


# --- owasp mapping ---

def test_inversion_maps_to_llm08_and_llm06():
    cats = owasp.categories_for(leak_finding())
    assert cats == [LLM08, LLM06]
    assert owasp.primary(leak_finding()) == LLM08


def test_unknown_module_falls_back_to_finding_owasp():
    f = leak_finding(module="cache_poison", owasp="LLM04: Data and Model Poisoning")
    assert owasp.categories_for(f) == ["LLM04: Data and Model Poisoning"]


def test_no_owasp_is_uncategorized():
    f = leak_finding(module="mystery", owasp="")
    assert owasp.categories_for(f) == [UNCATEGORIZED]


# --- scorecard ---

def test_scorecard_contains_fields_and_is_ascii():
    out = scorecard.format_scorecard([leak_finding()])
    assert "inversion" in out
    assert "LEAK" in out
    assert "0.82" in out
    assert "LLM08" in out
    assert out.isascii(), "scorecard must be pure ASCII for the cp1252 console"
    assert "→" not in out  # no unicode arrow


def test_scorecard_handles_empty():
    assert "no findings" in scorecard.format_scorecard([]).lower()


# --- json ---

def test_json_report_round_trips(tmp_path):
    path = json_report.write_json_report([leak_finding()], tmp_path / "r" / "report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    f = data["findings"][0]
    assert f["verdict"] == "LEAK"           # enum serialized as plain string
    assert f["severity"] == "HIGH"
    assert f["module"] == "inversion"
    assert f["owasp_categories"] == [LLM08, LLM06]
    assert f["details"]["per_category_recall"]["pii"] == 0.6


# --- html ---

def test_html_is_self_contained_document():
    out = html_report.render_html([leak_finding()])
    assert out.lstrip().startswith("<!doctype html")
    assert "</html>" in out
    assert "inversion" in out and "LEAK" in out
    assert "http://" not in out and "https://" not in out  # no external assets


def test_html_escapes_evidence_content():
    f = leak_finding(evidence=[{"recovered": "<b>inject</b> & <script>x</script>"}])
    out = html_report.render_html([f])
    assert "&lt;b&gt;inject&lt;/b&gt;" in out
    assert "<b>inject</b>" not in out
    assert "<script>x</script>" not in out


# --- uniformity: no module-specific special-casing ---

def test_renderers_accept_mixed_findings(tmp_path):
    findings = [leak_finding(), inconclusive_finding()]
    assert scorecard.format_scorecard(findings)
    assert "INCONCLUSIVE" in scorecard.format_scorecard(findings)
    json_report.write_json_report(findings, tmp_path / "r.json")
    assert html_report.render_html(findings).count("class='finding'") == 2
