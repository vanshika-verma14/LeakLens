"""End-to-end CLI test on a tiny Chroma store.

Exercises the whole path — argparse -> load_config -> build adapter -> runner -> module ->
scorecard + reports — for real. The only thing stubbed is the 2GB inverter (monkeypatched
get_inverter), so the test stays fast while everything else runs as it will in production.
"""
import json

import yaml

from leaklens import cli
from leaklens.adapters.base import Sample
from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.inversion import inverter as inverter_mod

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


def _key(vec):
    return tuple(round(float(x), 3) for x in vec)


class FakeInverter:
    """Recovers each row's own text (keyed by vector) — no model load."""

    def __init__(self):
        self._by_vec = {_key(s.vector): s.text for s in SAMPLES}

    def invert(self, embeddings, num_steps=None):
        return [self._by_vec[_key(vec)] for vec in embeddings.tolist()]

    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def _write_config(tmp_path, store_dir, out_dir):
    cfg = {
        "target": {"vector_store": {"type": "chroma", "path": str(store_dir),
                                    "collection": "docs",
                                    "encoder": inverter_mod.ENCODER}},
        "modules": ["inversion"],
        "inversion": {"recovery_threshold": 0.6, "num_steps": 1},
        "options": {"i_own_this_target": True, "output_dir": str(out_dir), "seed": 42},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_scan_end_to_end(tmp_path, monkeypatch, capsys):
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    ChromaAdapter(store_dir, "docs").add(SAMPLES)
    out_dir = tmp_path / "out"
    config_path = _write_config(tmp_path, store_dir, out_dir)

    monkeypatch.setattr(inverter_mod, "get_inverter", lambda **kw: FakeInverter())

    rc = cli.main(["scan", "--config", str(config_path)])
    assert rc == 0

    # scorecard reached stdout
    out = capsys.readouterr().out
    assert "inversion" in out

    # both reports written
    json_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    assert json_path.is_file() and html_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["findings"]) == 1
    finding = data["findings"][0]
    assert finding["module"] == "inversion"
    assert finding["verdict"] in ("LEAK", "NO_LEAK", "INCONCLUSIVE")
    assert "owasp_categories" in finding
    assert html_path.read_text(encoding="utf-8").lstrip().startswith("<!doctype html")


def test_missing_config_returns_2(tmp_path):
    assert cli.main(["scan", "--config", str(tmp_path / "nope.yaml")]) == 2


def test_no_subcommand_returns_2():
    assert cli.main([]) == 2
