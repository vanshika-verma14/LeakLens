"""Run LeakLens over the built validation targets and write results/validation_matrix.md.

Reads the manifest from `build_validation_targets.py`, runs the inversion module against
each store, and renders the pass/fail matrix — the README-grade "N/N correctly classified"
artifact.

Two modes:

* default (real): loads vec2text and inverts for real. This is the citable run and it is
  slow (~48s/sentence on CPU) — run it manually or on Colab, like the defense sweep.
* `--fake`: keys a stand-in inverter off the built RAW stores (raw recovers its own text,
  noised misses), so the matrix STRUCTURE is inspectable instantly without a model. It
  validates the harness plumbing, not the real recovery numbers — the header says so.

Run:  python scripts/run_validation.py            # real inverter (slow)
      python scripts/run_validation.py --fake      # instant structural check
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.validation.harness import all_passed, render_matrix, run_validation

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "results" / "validation" / "manifest.json"
DEFAULT_MATRIX = ROOT / "results" / "validation_matrix.md"


def _round_key(vec):
    return tuple(round(float(x), 3) for x in vec)


class _RawOnlyInverter:
    """Recovers a row's text only for its RAW vector; noised vectors miss -> filler.

    Keyed off the built raw stores so the matrix is structurally exercisable without the
    2GB model. NOT the real recovery — for plumbing/inspection only.
    """

    def __init__(self, manifest):
        self._by_vec = {}
        for t in manifest["targets"]:
            if t["defense"] != "none":
                continue
            for s in ChromaAdapter(t["path"], t["collection"]).get_all():
                self._by_vec[_round_key(s.vector)] = s.text

    def invert(self, embeddings, num_steps=None):
        return [self._by_vec.get(_round_key(v), "unrelated filler zzz")
                for v in embeddings.tolist()]

    def encode(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the validation matrix.")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_MATRIX)
    ap.add_argument("--fake", action="store_true",
                    help="use a model-free stand-in inverter (structure check only)")
    args = ap.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    inverter = _RawOnlyInverter(manifest) if args.fake else None
    if args.fake:
        print("[--fake] structural check only — recovery numbers are NOT the real model")

    results = run_validation(manifest, inverter=inverter)
    md = render_matrix(results, threshold=manifest["threshold"],
                       sigma=manifest["sigma"], seed=manifest["seed"])
    if args.fake:
        md = md.replace(
            "# LeakLens — Validation Matrix",
            "# LeakLens — Validation Matrix\n\n"
            "> _(--fake structural run: harness plumbing only, not real recovery numbers.)_")

    args.out.write_text(md, encoding="utf-8")
    for r in results:
        print(f"  {r.name:16s} expect={r.expected:8s} got={r.actual:8s} "
              f"recall={r.mean_recall:.3f} {'PASS' if r.passed else 'FAIL'}")
    print(f"\n{sum(1 for r in results if r.passed)}/{len(results)} correct -> {args.out}")
    return 0 if all_passed(results) else 1


if __name__ == "__main__":
    sys.exit(main())
