"""Fast tests for the validation harness — no real model.

These prove the harness *logic*: it runs the tool over each target, checks the verdict
against the derived expectation, and is deterministic under a fixed seed. The 2GB vec2text
inverter is replaced by a fake that recovers a row's text only when it sees that row's raw
vector; a noised vector misses -> recall 0. So raw stores classify LEAK and their noised
pairs classify NO_LEAK, exercising the whole classification path model-free.

The real-model matrix (live vec2text over the built stores) is script-driven / Colab,
consistent with how the slow inversion and the defense sweep are handled.
"""
import numpy as np

from leaklens.adapters.base import Sample
from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.inversion import inverter as inverter_mod
from leaklens.inversion.defenses import gaussian_noise
from leaklens.validation.harness import (
    all_passed, render_matrix, run_validation)

SIGMA = 0.05
SEED = 42
THRESHOLD = 0.6

# Three content sets; plain-dominant so a real store would clear the threshold too. Here the
# fake inverter drives recall, so the point is only that raw recovers and noised does not.
_CONTENT = {
    "setA": [
        Sample(id="a1", vector=[0.11, 0.22, 0.33, 0.44], type="plain",
               text="The library extended its opening hours downtown.",
               key_entities=["library", "opening hours", "downtown"]),
        Sample(id="a2", vector=[0.21, 0.12, 0.43, 0.34], type="plain",
               text="The debate team won the regional housing policy round.",
               key_entities=["debate team", "housing policy"]),
    ],
    "setB": [
        Sample(id="b1", vector=[0.91, 0.13, 0.15, 0.17], type="plain",
               text="Quarterly revenue rose twelve percent this spring.",
               key_entities=["Quarterly revenue", "twelve percent", "spring"]),
        Sample(id="b2", vector=[0.31, 0.52, 0.19, 0.28], type="plain",
               text="The museum reopened the ancient sculpture wing.",
               key_entities=["museum", "sculpture wing"]),
    ],
    "setC": [
        Sample(id="c1", vector=[0.41, 0.62, 0.23, 0.18], type="plain",
               text="The city council approved the new bike lanes.",
               key_entities=["city council", "bike lanes"]),
        Sample(id="c2", vector=[0.15, 0.25, 0.65, 0.35], type="pii",
               text="Reach Mei Lin about the Brookline project.",
               key_entities=["Mei Lin", "Brookline"]),
    ],
}


def _key(vec):
    return tuple(round(float(x), 3) for x in vec)


class RawOnlyInverter:
    """Recovers a row's text only for its RAW vector; any other (noised) vector -> filler.

    Mimics the real behaviour we validate: raw embeddings invert back to their text (high
    recall); noised embeddings do not (recall ~0). No model, fully deterministic.
    """

    def __init__(self, raw_samples):
        self._by_vec = {_key(s.vector): s.text for s in raw_samples}

    def invert(self, embeddings, num_steps=None):
        return [self._by_vec.get(_key(v), "unrelated filler zzz")
                for v in embeddings.tolist()]

    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _build_pair(base_dir, name, samples):
    """Build a raw store and its sigma-noised pair; return two manifest target dicts."""
    raw_dir = base_dir / f"{name}-raw"
    noised_dir = base_dir / f"{name}-noised"
    ChromaAdapter(raw_dir, "docs").add(samples)

    noised_vecs = gaussian_noise([s.vector for s in samples], SIGMA, seed=SEED)
    noised = [Sample(id=s.id, vector=noised_vecs[i].tolist(), text=s.text,
                     type=s.type, key_entities=s.key_entities)
              for i, s in enumerate(samples)]
    ChromaAdapter(noised_dir, "docs").add(noised)

    profile = f"{len(samples)} rows"
    return [
        {"name": f"{name}-raw", "path": str(raw_dir), "collection": "docs",
         "content_profile": profile, "defense": "none", "expected": "LEAK"},
        {"name": f"{name}-noised", "path": str(noised_dir), "collection": "docs",
         "content_profile": profile, "defense": f"gaussian sigma={SIGMA}",
         "expected": "NO_LEAK"},
    ]


def _build_manifest(tmp_path):
    targets, raw_samples = [], []
    for name, samples in _CONTENT.items():
        targets += _build_pair(tmp_path, name, samples)
        raw_samples += samples
    manifest = {"encoder": inverter_mod.ENCODER, "threshold": THRESHOLD, "seed": SEED,
                "sigma": SIGMA, "targets": targets}
    return manifest, RawOnlyInverter(raw_samples)


def test_all_targets_classified_correctly(tmp_path):
    manifest, inv = _build_manifest(tmp_path)
    results = run_validation(manifest, inverter=inv)

    assert len(results) == 6
    assert all_passed(results)
    # leaky flagged, hardened cleared
    for r in results:
        if r.defense == "none":
            assert r.expected == "LEAK" and r.actual == "LEAK"
            assert r.mean_recall >= THRESHOLD
        else:
            assert r.expected == "NO_LEAK" and r.actual == "NO_LEAK"
            assert r.mean_recall < THRESHOLD


def test_matrix_reports_all_pass(tmp_path):
    manifest, inv = _build_manifest(tmp_path)
    results = run_validation(manifest, inverter=inv)
    md = render_matrix(results, threshold=THRESHOLD, sigma=SIGMA, seed=SEED)

    assert "6/6 correctly classified" in md
    assert "FAIL" not in md
    assert md.count("| PASS |") == 6


def test_deterministic_under_fixed_seed(tmp_path):
    manifest, inv = _build_manifest(tmp_path)
    first = run_validation(manifest, inverter=inv)
    second = run_validation(manifest, inverter=inv)

    assert [(r.name, r.actual, r.mean_recall) for r in first] == \
           [(r.name, r.actual, r.mean_recall) for r in second]


def test_wrong_expectation_is_reported_as_fail(tmp_path):
    # Proves the harness actually checks: mislabel a raw (leaky) store as NO_LEAK and it
    # must surface as a FAIL, not be rubber-stamped.
    manifest, inv = _build_manifest(tmp_path)
    raw = next(t for t in manifest["targets"] if t["defense"] == "none")
    raw["expected"] = "NO_LEAK"

    results = run_validation(manifest, inverter=inv)
    bad = next(r for r in results if r.name == raw["name"])
    assert bad.actual == "LEAK"          # the tool still says LEAK
    assert bad.passed is False           # ...so the mislabel fails
    assert not all_passed(results)
