"""Tests for the Finding contract — fast, no model.

Serialization + defaults, per the Slice-3 manifest. The str-mixin assertions
are load-bearing: the report layer serializes Findings via asdict + json.dumps
and compares verdicts against plain strings.
"""
import dataclasses
import json

from leaklens.finding import Finding, Severity, Verdict


def make_finding(**overrides):
    kwargs = dict(
        module="inversion",
        verdict=Verdict.LEAK,
        score=0.8,
        severity=Severity.HIGH,
        owasp="LLM08: Vector and Embedding Weaknesses",
        summary="recovered 8/10 key entities",
    )
    kwargs.update(overrides)
    return Finding(**kwargs)


def test_defaults():
    f = make_finding()
    assert f.evidence == []
    assert f.remediation == ""
    assert f.details == {}


def test_mutable_defaults_are_independent():
    a, b = make_finding(), make_finding()
    a.evidence.append(("original", "recovered"))
    a.details["recall"] = 0.8
    assert b.evidence == []
    assert b.details == {}


def test_verdict_members():
    assert {v.value for v in Verdict} == {"LEAK", "NO_LEAK", "INCONCLUSIVE"}
    assert Verdict.INCONCLUSIVE == "INCONCLUSIVE"


def test_severity_members():
    assert {s.value for s in Severity} == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert Severity.CRITICAL == "CRITICAL"


def test_json_serialization_round_trip():
    f = make_finding(
        evidence=[["the original text", "the recovered text"]],
        remediation="add gaussian noise sigma=0.05",
        details={"recall": 0.8, "cosine": 0.97},
    )
    payload = json.loads(json.dumps(dataclasses.asdict(f)))
    assert payload["verdict"] == "LEAK"
    assert payload["severity"] == "HIGH"
    assert payload["module"] == "inversion"
    assert payload["score"] == 0.8
    assert payload["evidence"] == [["the original text", "the recovered text"]]
    assert payload["remediation"] == "add gaussian noise sigma=0.05"
    assert payload["details"] == {"recall": 0.8, "cosine": 0.97}
