"""The one shape every module returns: a `Finding`.

The runner and report layer only ever consume this — never a module-specific
result — so one module failing or returning INCONCLUSIVE can't force
special-casing anywhere downstream. INCONCLUSIVE is a first-class verdict:
honesty over a confident-but-unsupported answer (ARCHITECTURE principles 1, 4).

`Verdict` and `Severity` mix in `str` so `dataclasses.asdict` + `json.dumps`
serialize a Finding with zero extra code — the JSON report relies on this.
"""
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    LEAK = "LEAK"
    NO_LEAK = "NO_LEAK"
    INCONCLUSIVE = "INCONCLUSIVE"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class Finding:
    module: str                       # "inversion" | "cache_poison"
    verdict: Verdict
    score: float                      # 0.0–1.0 leakage severity
    severity: Severity
    owasp: str                        # e.g. "LLM08: Vector and Embedding Weaknesses"
    summary: str                      # one line for the scorecard
    evidence: list = field(default_factory=list)   # original↔recovered pairs, numbers
    remediation: str = ""             # concrete fix (e.g. recommended noise σ)
    details: dict = field(default_factory=dict)    # raw metrics for the JSON report
