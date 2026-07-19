"""Build the validation matrix targets: 3 leaky (raw) + 3 hardened (noised) Chroma stores.

Design (paired same-content): each content set is built twice — raw and Gaussian-noised at
sigma=0.05 — so within a pair the *defense is the only variable*. That makes "the tool
flips its verdict correctly" a claim about the defense, not about two unrelated stores.

Honesty rules baked in:

* **Leaky content must genuinely recover.** A raw store is only a valid known-LEAK target
  if its content clears the leak threshold. We compose the sets from high-recovery content
  (plain, plain-dominant) and *derive* the expected mean recall from measured per-category
  recall (default `results/inversion_demo.json`, the real --limit run). If a set we intend
  as leaky is predicted below threshold, we ABORT — a false "known-LEAK" label is worse
  than no matrix.
* **Hardened expectation is measured too.** sigma=0.05 sits past the leakage-collapse point
  in the defense sweep; where a sweep point is available we read the measured leakage,
  otherwise we record the provenance rather than asserting a number we didn't measure.

Encoding is the fast part (seconds); only inversion is slow, and inversion happens later in
`scripts/run_validation.py`. Reproducible under --seed.

Run:  python scripts/build_validation_targets.py
      -> results/validation/<name>-{raw,noised}/  +  results/validation/manifest.json
"""
import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # cert store blocks huggingface.co; cached
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leaklens.adapters.base import Sample
from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.inversion.defenses import gaussian_noise
from leaklens.inversion.inverter import ENCODER, get_inverter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "corpus" / "corpus.jsonl"
DEFAULT_OUT = ROOT / "results" / "validation"
DEFAULT_RECALL_SOURCE = ROOT / "results" / "inversion_demo.json"
DEFAULT_SWEEP_SOURCE = ROOT / "results" / "sweep.json"

# Content-set recipes: {set_name: {category: count}}. Plain-dominant so the raw stores
# legitimately exceed the threshold (verified below against measured recall, not assumed).
SET_RECIPES = {
    "setA": {"plain": 20},
    "setB": {"plain": 20},
    "setC": {"plain": 12, "pii": 8},
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def measured_per_category_recall(path: Path) -> dict[str, float]:
    """Mean recall per category from a real inversion run (list of scored rows)."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        buckets[r["type"]].append(float(r["recall"]))
    return {t: sum(v) / len(v) for t, v in buckets.items()}


def measured_leakage_at_sigma(path: Path, sigma: float) -> float | None:
    """Measured overall leakage recall at `sigma` from a sweep file, or None if absent."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for point in data.get("results", []):
        if abs(float(point["sigma"]) - sigma) < 1e-9:
            return float(point["leakage_recall"])
    return None


def compose_sets(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Draw disjoint, reproducible content sets per SET_RECIPES from the corpus."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    rng = random.Random(seed)
    for group in by_type.values():
        rng.shuffle(group)                 # reproducible order
    cursors: dict[str, int] = defaultdict(int)
    sets: dict[str, list[dict]] = {}
    for name, recipe in SET_RECIPES.items():
        chosen: list[dict] = []
        for cat, count in recipe.items():
            avail = by_type.get(cat, [])
            start = cursors[cat]
            picked = avail[start:start + count]
            if len(picked) < count:
                raise SystemExit(
                    f"corpus has only {len(avail) - start} unused '{cat}' rows left; "
                    f"set '{name}' needs {count}")
            cursors[cat] += count
            chosen += picked
        sets[name] = chosen
    return sets


def predict_mean_recall(set_rows: list[dict], per_cat: dict[str, float]) -> float:
    """Row-weighted expected mean recall for a content set from measured per-category recall."""
    vals = []
    for r in set_rows:
        if r["type"] not in per_cat:
            raise SystemExit(
                f"no measured recall for category '{r['type']}' in the recall source")
        vals.append(per_cat[r["type"]])
    return sum(vals) / len(vals)


def to_samples(rows: list[dict], vectors) -> list[Sample]:
    return [Sample(id=r["id"], vector=vectors[i].tolist(), text=r["text"],
                   type=r["type"], key_entities=r["key_entities"])
            for i, r in enumerate(rows)]


def build_store(path: Path, collection: str, samples: list[Sample]) -> None:
    if path.exists():
        shutil.rmtree(path)                # fresh build -> idempotent, no duplicate ids
    path.mkdir(parents=True, exist_ok=True)
    ChromaAdapter(path, collection).add(samples)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build validation matrix target stores.")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--collection", default="docs")
    ap.add_argument("--sigma", type=float, default=0.05,
                    help="Gaussian sigma for the hardened pair (past the sweep's collapse)")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="leak threshold the labels are derived against (DECISIONS D9)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--recall-source", type=Path, default=DEFAULT_RECALL_SOURCE,
                    help="measured per-category recall for deriving LEAK labels")
    ap.add_argument("--sweep-source", type=Path, default=DEFAULT_SWEEP_SOURCE,
                    help="measured sweep for confirming hardened NO_LEAK at sigma")
    args = ap.parse_args(argv)

    rows = load_rows(args.corpus)
    per_cat = measured_per_category_recall(args.recall_source)
    print(f"measured per-category recall ({args.recall_source.name}): "
          + ", ".join(f"{k}={v:.3f}" for k, v in sorted(per_cat.items())))

    sets = compose_sets(rows, args.seed)

    # Derive + guard the leaky labels BEFORE embedding anything.
    predicted = {name: predict_mean_recall(r, per_cat) for name, r in sets.items()}
    for name, pred in predicted.items():
        verdict = "LEAK" if pred >= args.threshold else "below-threshold"
        print(f"  {name}: predicted mean recall {pred:.3f} -> {verdict}")
        if pred < args.threshold:
            raise SystemExit(
                f"ABORT: set '{name}' predicts recall {pred:.3f} < threshold "
                f"{args.threshold} — it would be a FALSE known-LEAK target. Recompose it "
                "from higher-recovery content.")

    # Hardened expectation: confirm from the sweep where possible, else record provenance.
    sweep_leak = measured_leakage_at_sigma(args.sweep_source, args.sigma)
    if sweep_leak is not None:
        hardened_provenance = (f"measured sweep leakage {sweep_leak:.3f} at sigma="
                               f"{args.sigma} ({args.sweep_source.name})")
    else:
        hardened_provenance = (f"no sweep point at sigma={args.sigma}; sigma is past the "
                               "documented leakage collapse (PROGRESS F4: overall recall "
                               "0.51->0.10 by sigma=0.02)")
    print(f"hardened basis: {hardened_provenance}")

    # Embed every needed row once (the fast part).
    all_rows = [r for rs in sets.values() for r in rs]
    print(f"encoding {len(all_rows)} rows with GTR...")
    emb = get_inverter().encode([r["text"] for r in all_rows])
    emb_by_id = {r["id"]: emb[i] for i, r in enumerate(all_rows)}

    targets = []
    for name, set_rows in sets.items():
        vecs = [emb_by_id[r["id"]] for r in set_rows]
        raw_samples = to_samples(set_rows, vecs)
        profile = " + ".join(f"{c} {t}" for t, c in SET_RECIPES[name].items())

        raw_dir = args.out / f"{name}-raw"
        build_store(raw_dir, args.collection, raw_samples)
        targets.append({
            "name": f"{name}-raw", "path": str(raw_dir), "collection": args.collection,
            "content_profile": profile, "defense": "none", "sigma": 0.0,
            "predicted_mean_recall": round(predicted[name], 3), "expected": "LEAK",
            "label_basis": f"measured per-category recall ({args.recall_source.name})"})

        noised_vecs = gaussian_noise(vecs, args.sigma, seed=args.seed)
        noised_samples = to_samples(set_rows, noised_vecs)
        noised_dir = args.out / f"{name}-noised"
        build_store(noised_dir, args.collection, noised_samples)
        targets.append({
            "name": f"{name}-noised", "path": str(noised_dir),
            "collection": args.collection, "content_profile": profile,
            "defense": f"gaussian sigma={args.sigma}", "sigma": args.sigma,
            "predicted_mean_recall": (round(sweep_leak, 3) if sweep_leak is not None
                                      else None),
            "expected": "NO_LEAK", "label_basis": hardened_provenance})

    manifest = {
        "encoder": ENCODER, "threshold": args.threshold, "seed": args.seed,
        "sigma": args.sigma, "recall_source": str(args.recall_source),
        "measured_per_category_recall": {k: round(v, 4) for k, v in per_cat.items()},
        "targets": targets,
    }
    manifest_path = args.out / "manifest.json"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"built {len(targets)} targets -> {args.out}\nmanifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
