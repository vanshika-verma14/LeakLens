"""Resume semantics of studies/defense_sweep.py — fast, no model load.

The property that matters (Colab disconnects mid---full-run): the output JSON on disk is
always a complete, valid document for the sigmas finished so far, and re-running the same
command resumes — skipping done sigmas — such that chunked runs converge to byte-identical
output vs one uninterrupted run. The heavy inverter is faked with a pure function of its
input (determinism is what makes resume-equivalence testable at all).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studies"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import defense_sweep  # noqa: E402
import merge_sweeps  # noqa: E402
import plot_tradeoff  # noqa: E402

from leaklens.adapters.base import Sample  # noqa: E402
from leaklens.adapters.chroma_adapter import ChromaAdapter  # noqa: E402

# Entities are the σ=0 rendering of each vector's first component: the fake inverter
# "recovers" a vector as its formatted values, so leakage is 1.0 undefended and decays
# once noise perturbs the values — a miniature of the real curve, fully deterministic.
SAMPLES = [
    Sample(id="plain-001", vector=[0.1, 0.2, 0.3, 0.4], text="t1", type="plain",
           key_entities=["0.1000"]),
    Sample(id="pii-001", vector=[0.5, 0.4, 0.3, 0.2], text="t2", type="pii",
           key_entities=["0.5000"]),
    Sample(id="cred-001", vector=[0.9, 0.1, 0.1, 0.1], text="t3", type="credential",
           key_entities=["0.9000"]),
    Sample(id="struct-001", vector=[0.2, 0.2, 0.7, 0.1], text="t4", type="structured",
           key_entities=["0.2000"]),
]


class FakeInverter:
    """Pure function of the input embeddings — same defended vectors, same 'recovery'."""

    def invert(self, emb, num_steps=None):
        return [" ".join(f"{v:.4f}" for v in row) for row in emb.tolist()]

    def encode(self, texts):
        return [np.array([1.0 + sum(map(ord, t)) % 97, 2.0, 3.0, 4.0]) for t in texts]


class CrashingInverter(FakeInverter):
    """Dies on the Nth invert call — simulates Colab disconnecting mid-sigma."""

    def __init__(self, fail_on_call: int):
        self.fail_on_call = fail_on_call
        self.calls = 0

    def invert(self, emb, num_steps=None):
        self.calls += 1
        if self.calls >= self.fail_on_call:
            raise RuntimeError("colab disconnected")
        return super().invert(emb, num_steps=num_steps)


@pytest.fixture
def store_dir(temp_store_dir):
    ChromaAdapter(temp_store_dir, "docs").add(SAMPLES)
    return temp_store_dir


def run_sweep(store_dir, out, sigmas, *, num_steps=1):
    # --full + batch-size >= 4 rows -> exactly one invert call per sigma (CrashingInverter
    # counts on this).
    return defense_sweep.main([
        "--store", str(store_dir), "--collection", "docs", "--full",
        "--num-steps", str(num_steps), "--batch-size", "8", "--k", "2",
        "--sigmas", sigmas, "--out", str(out)])


def test_chunked_resume_equals_single_full_run(store_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(defense_sweep, "get_inverter", lambda: FakeInverter())
    single, chunked = tmp_path / "single.json", tmp_path / "chunked.json"

    assert run_sweep(store_dir, single, "0,0.1,0.2") == 0
    assert run_sweep(store_dir, chunked, "0,0.1") == 0            # session 1 (chunk)
    assert run_sweep(store_dir, chunked, "0,0.1,0.2") == 0        # session 2 (resume)

    assert chunked.read_text(encoding="utf-8") == single.read_text(encoding="utf-8")
    data = json.loads(single.read_text(encoding="utf-8"))
    assert [r["sigma"] for r in data["results"]] == [0.0, 0.1, 0.2]
    assert data["results"][0]["leakage_recall"] == 1.0            # undefended baseline


def test_crash_leaves_valid_partial_json(store_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(defense_sweep, "get_inverter",
                        lambda: CrashingInverter(fail_on_call=3))
    out = tmp_path / "sweep.json"
    with pytest.raises(RuntimeError, match="colab disconnected"):
        run_sweep(store_dir, out, "0,0.1,0.2")
    data = json.loads(out.read_text(encoding="utf-8"))            # valid JSON, not a stub
    assert [r["sigma"] for r in data["results"]] == [0.0, 0.1]
    assert data["num_steps"] == 1 and data["rows"] == len(SAMPLES)


def test_fully_done_resume_never_loads_the_model(store_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(defense_sweep, "get_inverter", lambda: FakeInverter())
    out = tmp_path / "sweep.json"
    assert run_sweep(store_dir, out, "0,0.1") == 0
    before = out.read_text(encoding="utf-8")

    def boom():
        raise AssertionError("model loaded on a no-op resume")
    monkeypatch.setattr(defense_sweep, "get_inverter", boom)
    assert run_sweep(store_dir, out, "0,0.1") == 0
    assert out.read_text(encoding="utf-8") == before


def test_mismatched_config_is_refused(store_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(defense_sweep, "get_inverter", lambda: FakeInverter())
    out = tmp_path / "sweep.json"
    assert run_sweep(store_dir, out, "0,0.1", num_steps=1) == 0
    before = out.read_text(encoding="utf-8")
    # resuming under a different num_steps would mix incomparable points -> refuse
    assert run_sweep(store_dir, out, "0,0.1,0.2", num_steps=2) == 2
    assert out.read_text(encoding="utf-8") == before


def test_merge_chunks_equals_single_full_run(store_dir, tmp_path, monkeypatch):
    """Chunks run into SEPARATE --out files (resume can't see across files) merge
    back to the byte-identical JSON of one uninterrupted run; the overlapping
    sigma (0.1, deterministic -> identical in both chunks) dedupes silently."""
    monkeypatch.setattr(defense_sweep, "get_inverter", lambda: FakeInverter())
    single = tmp_path / "single.json"
    chunk_a, chunk_b = tmp_path / "chunk_a.json", tmp_path / "chunk_b.json"
    assert run_sweep(store_dir, single, "0,0.1,0.2") == 0
    assert run_sweep(store_dir, chunk_a, "0,0.1") == 0
    assert run_sweep(store_dir, chunk_b, "0.1,0.2") == 0

    merged = tmp_path / "merged.json"
    assert merge_sweeps.main([str(chunk_a), str(chunk_b), "--out", str(merged)]) == 0
    assert merged.read_text(encoding="utf-8") == single.read_text(encoding="utf-8")


def test_merge_refuses_mismatched_config(store_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(defense_sweep, "get_inverter", lambda: FakeInverter())
    chunk_a, chunk_b = tmp_path / "a.json", tmp_path / "b.json"
    assert run_sweep(store_dir, chunk_a, "0", num_steps=1) == 0
    assert run_sweep(store_dir, chunk_b, "0.1", num_steps=2) == 0
    out = tmp_path / "merged.json"
    assert merge_sweeps.main([str(chunk_a), str(chunk_b), "--out", str(out)]) == 2
    assert not out.exists()


def test_merge_refuses_conflicting_duplicate_sigma(store_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(defense_sweep, "get_inverter", lambda: FakeInverter())
    chunk_a, chunk_b = tmp_path / "a.json", tmp_path / "b.json"
    assert run_sweep(store_dir, chunk_a, "0,0.1") == 0
    doc = json.loads(chunk_a.read_text(encoding="utf-8"))
    doc["results"][1]["leakage_recall"] = 0.123          # same sigma, different value
    chunk_b.write_text(json.dumps(doc), encoding="utf-8")
    out = tmp_path / "merged.json"
    assert merge_sweeps.main([str(chunk_a), str(chunk_b), "--out", str(out)]) == 2
    assert not out.exists()


def test_plot_renders_partial_sweep(store_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(defense_sweep, "get_inverter",
                        lambda: CrashingInverter(fail_on_call=3))
    out = tmp_path / "sweep.json"
    with pytest.raises(RuntimeError):
        run_sweep(store_dir, out, "0,0.1,0.2")                    # leaves a 2-sigma partial
    png = tmp_path / "tradeoff.png"
    assert plot_tradeoff.main(["--in", str(out), "--out", str(png)]) == 0
    assert png.exists() and png.stat().st_size > 0
