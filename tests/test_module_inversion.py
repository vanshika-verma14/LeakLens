"""Tests for the inversion module — fast, no real model (fake inverter + adapter).

Two guards are load-bearing: a normal run assembles a coherent LEAK Finding, and the
honest-failure path returns INCONCLUSIVE *without* touching the inverter — proving we
never load a 2GB model or fabricate a score for an encoder we can't invert.
"""
import pytest

from leaklens.adapters.base import Sample
from leaklens.config import Config, InversionConfig, Options, VectorStoreConfig
from leaklens.finding import Severity, Verdict
from leaklens.inversion.inverter import ENCODER
from leaklens.modules.inversion import InversionModule, _truncate

SAMPLES = [
    Sample(id="plain-001", vector=[0.1, 0.2, 0.3, 0.4], type="plain",
           text="The library extended its opening hours.",
           key_entities=["library", "opening hours"]),
    Sample(id="pii-001", vector=[0.5, 0.4, 0.3, 0.2], type="pii",
           text="Email Priya Sharma at priya@example.com.",
           key_entities=["Priya Sharma", "priya@example.com"]),
    Sample(id="cred-001", vector=[0.9, 0.1, 0.1, 0.1], type="credential",
           text="Reset the admin password to Th1sIsSecret!.",
           key_entities=["admin password", "Th1sIsSecret!"]),
]


class FakeAdapter:
    """Minimal in-memory stand-in for a VectorStoreAdapter."""

    def __init__(self, samples):
        self._samples = list(samples)

    def count(self):
        return len(self._samples)

    def sample(self, n, *, seed=None):
        return list(self._samples[:n])


def _key(vec):
    # torch float32 shifts 0.1 -> 0.1000000014; round so lookups survive the round-trip.
    return tuple(round(float(x), 3) for x in vec)


class FakeInverter:
    """Returns canned recovered text keyed by each row's vector — batching-agnostic."""

    def __init__(self, recovered_by_vector):
        self._by_vec = {_key(v): t for v, t in recovered_by_vector.items()}

    def invert(self, embeddings, num_steps=None):
        return [self._by_vec[_key(vec)] for vec in embeddings.tolist()]

    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class RaisingInverter:
    """Any use is a failure — proves the honest-failure gate never touches the model."""

    def invert(self, *a, **k):
        raise AssertionError("inverter must not be used when no inverter is available")

    def encode(self, *a, **k):
        raise AssertionError("inverter must not be used when no inverter is available")


def make_cfg(*, encoder=ENCODER, recovery_threshold=0.6, sample_size=100):
    return Config(
        vector_store=VectorStoreConfig(type="chroma", path="./s", collection="docs",
                                       encoder=encoder),
        modules=["inversion"],
        options=Options(i_own_this_target=True, seed=42),
        inversion=InversionConfig(recovery_threshold=recovery_threshold,
                                  sample_size=sample_size),
    )


def test_normal_run_produces_leak_finding():
    # Recovered text contains every key entity -> recall 1.0 -> LEAK.
    recovered = {tuple(s.vector): s.text for s in SAMPLES}
    module = InversionModule(inverter=FakeInverter(recovered))

    f = module.run(FakeAdapter(SAMPLES), make_cfg())

    assert f.module == "inversion"
    assert f.verdict is Verdict.LEAK
    assert 0.0 <= f.score <= 1.0
    assert f.score >= 0.6
    assert f.severity in (Severity.HIGH, Severity.CRITICAL)
    assert f.owasp.startswith("LLM08")
    assert f.evidence, "a LEAK should carry original<->recovered evidence"
    assert {"id", "original", "recovered", "recall"} <= set(f.evidence[0])
    assert "per_category_recall" in f.details
    assert f.remediation  # concrete fix offered


def test_no_inverter_encoder_returns_inconclusive_without_loading():
    # Different encoder -> honest failure, and the inverter must never be touched.
    module = InversionModule(inverter=RaisingInverter())

    f = module.run(FakeAdapter(SAMPLES), make_cfg(encoder="openai/text-embedding-3-small"))

    assert f.verdict is Verdict.INCONCLUSIVE
    assert f.score == 0.0
    assert "no inverter available" in f.summary
    assert f.details["encoder"] == "openai/text-embedding-3-small"


def test_low_recovery_returns_no_leak():
    # Recovered text shares nothing with the entities -> recall 0 -> NO_LEAK.
    recovered = {tuple(s.vector): "unrelated filler text zzz" for s in SAMPLES}
    module = InversionModule(inverter=FakeInverter(recovered))

    f = module.run(FakeAdapter(SAMPLES), make_cfg())

    assert f.verdict is Verdict.NO_LEAK
    assert f.severity is Severity.LOW
    assert f.remediation == ""


def test_empty_store_returns_inconclusive():
    module = InversionModule(inverter=RaisingInverter())
    f = module.run(FakeAdapter([]), make_cfg())
    assert f.verdict is Verdict.INCONCLUSIVE
    assert "no vectors" in f.summary


def test_truncate_keeps_multibyte_chars_intact():
    # A recovered string can contain real multibyte chars (accents, CJK, emoji). Truncation
    # must not split one into a replacement char, and the marker must be ASCII.
    s = "café résumé 日本語 " + "字" * 60  # comfortably longer than the clip width
    out = _truncate(s, 20)

    assert "�" not in out                 # no replacement char introduced
    assert out.endswith("...")                 # ASCII marker, not the U+2026 ellipsis
    assert "…" not in out
    assert s.startswith(out[:-3])              # body is an exact prefix — no split char
    out.encode("utf-8")                        # round-trips cleanly


def test_truncate_leaves_short_multibyte_string_unchanged():
    s = "café 日本語 ☕"
    assert _truncate(s, 80) == s


def test_module_isolates_adapter_errors():
    class BoomAdapter:
        def count(self):
            return 3

        def sample(self, n, *, seed=None):
            raise RuntimeError("store connection dropped")

    f = InversionModule(inverter=RaisingInverter()).run(BoomAdapter(), make_cfg())
    assert f.verdict is Verdict.INCONCLUSIVE
    assert "errored" in f.summary
