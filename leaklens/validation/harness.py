"""Run LeakLens over a set of validation targets and score the classification matrix.

The harness reads a manifest (built by `scripts/build_validation_targets.py`) describing
each target store, its baked-in defense, and its *expected* verdict. It then runs the real
inversion module against each store and checks the tool's verdict against the expectation.

Two properties make this the moat:

* **Reproducible.** The seed is threaded into sampling, so a fixed seed reproduces the same
  verdicts (NFR-3). `run_validation` is pure given (manifest, inverter, seed).
* **Honest.** The `expected` label is *derived from measured per-category recall* at build
  time (see the build script), not assumed. The harness only confirms actual == expected;
  a divergence is a reported FAIL, never silently patched.

The inverter is injectable: `inverter=None` uses the real process-wide vec2text; tests pass
a fake so the everyday suite stays fast and model-free.
"""
from __future__ import annotations

from dataclasses import dataclass

from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.config import Config, InversionConfig, Options, VectorStoreConfig
from leaklens.finding import Verdict
from leaklens.modules.inversion import InversionModule


@dataclass
class ValidationResult:
    """One target's outcome: what we expected, what the tool said, and whether they agree."""

    name: str
    expected: str            # "LEAK" | "NO_LEAK"
    actual: str              # the tool's Verdict value
    passed: bool
    mean_recall: float
    defense: str
    content_profile: str


def _config_for(target: dict, *, encoder: str, threshold: float, seed: int,
                sample_size: int) -> Config:
    """Build the Config the inversion module expects for one target store."""
    return Config(
        vector_store=VectorStoreConfig(
            type="chroma", path=target["path"],
            collection=target["collection"], encoder=encoder),
        modules=["inversion"],
        options=Options(i_own_this_target=True, seed=seed),
        inversion=InversionConfig(recovery_threshold=threshold, sample_size=sample_size),
    )


def run_validation(manifest: dict, *, inverter=None,
                   sample_size: int = 100) -> list[ValidationResult]:
    """Run the inversion module over every target in `manifest`; return one result each.

    `manifest` carries the run-wide `encoder`, `threshold`, and `seed`, plus a `targets`
    list. Reproducible: the same manifest + inverter + seed yields identical verdicts.
    """
    encoder = manifest["encoder"]
    threshold = manifest["threshold"]
    seed = manifest["seed"]

    results: list[ValidationResult] = []
    for target in manifest["targets"]:
        cfg = _config_for(target, encoder=encoder, threshold=threshold, seed=seed,
                          sample_size=sample_size)
        adapter = ChromaAdapter(target["path"], target["collection"])
        finding = InversionModule(inverter=inverter).run(adapter, cfg)
        actual = finding.verdict.value
        # A leaky target passes when flagged LEAK; a hardened one when cleared NO_LEAK.
        # INCONCLUSIVE never satisfies either expectation -> reported as a FAIL, honestly.
        passed = actual == target["expected"]
        results.append(ValidationResult(
            name=target["name"], expected=target["expected"], actual=actual,
            passed=passed, mean_recall=round(float(finding.score), 3),
            defense=target["defense"], content_profile=target["content_profile"]))
    return results


def all_passed(results: list[ValidationResult]) -> bool:
    return all(r.passed for r in results)


def render_matrix(results: list[ValidationResult], *, threshold: float,
                  sigma: float, seed: int) -> str:
    """Render the results as a Markdown validation matrix — the README-grade artifact."""
    n_pass = sum(1 for r in results if r.passed)
    total = len(results)
    lines = [
        "# LeakLens — Validation Matrix",
        "",
        "Known-safe / known-vulnerable matrix. Each **leaky** store holds raw embeddings; "
        "its **hardened** pair holds the *same content* with Gaussian noise "
        f"(sigma={sigma}) baked in — so the defense is the only variable within a pair. "
        f"A store is flagged LEAK when mean key-entity recall >= {threshold} "
        "(the calibrated threshold, DECISIONS D9).",
        "",
        "Expected labels are **derived from measured per-category recall**, not assumed: "
        "a raw store is labelled LEAK only when its content's measured recall exceeds the "
        "threshold. The tool must reproduce these labels.",
        "",
        f"**Result: {n_pass}/{total} correctly classified** "
        f"(reproducible at seed={seed}).",
        "",
        "| Target | Content | Defense | Expected | Verdict | Mean recall | Pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| {r.name} | {r.content_profile} | {r.defense} | {r.expected} | "
            f"{r.actual} | {r.mean_recall:.3f} | {mark} |")
    lines.append("")
    return "\n".join(lines)
