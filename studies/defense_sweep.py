"""Defense tradeoff sweep — the finding.

At each Gaussian-noise level σ, measure BOTH sides of the tradeoff:
  * leakage  = key-entity recall from inverting the defended embeddings
  * utility  = recall@k of a clean query against the defended store
so the output is a privacy/utility curve, not a binary. σ=0 is the undefended baseline
(high leakage, utility 1.0). No leak threshold is applied — raw numbers for you to read.

The two sides use different evaluation sets ON PURPOSE:
  * leakage is scored on the --limit stratified subset (inversion is the expensive part);
  * utility is always measured against the FULL store (numpy-only, effectively free), so a
    target competes with all ~240 real neighbors. Scoring utility on a tiny subset would
    read a flat 1.0 (everything trivially retrieves itself) and hide the degradation.
The subset's defended vectors are sliced out of the SAME full defended array, so the exact
vector scored for leakage is the one sitting in the retrieval store — one coherent point.

--limit-aware like the demo: develop on a small subset, then --full (Colab) for the real
curve. Cost ≈ len(sigmas) x rows x num_steps inversions, so keep --limit small while
iterating (utility cost is independent of --limit).

Run:  python studies/defense_sweep.py --limit 12 --num-steps 5
      python studies/defense_sweep.py --full            # the long run
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # cached models; cert store blocks HF
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from leaklens.adapters.base import stratified_sample
from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.inversion import defenses, metrics, retrieval
from leaklens.inversion.inverter import get_inverter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "results" / "corpus_store"
DEFAULT_OUT = ROOT / "results" / "sweep.json"


def _invert_all(inv, defended, num_steps, batch_size):
    import torch
    out = []
    for start in range(0, len(defended), batch_size):
        chunk = defended[start:start + batch_size]
        recovered = inv.invert(torch.tensor(chunk, dtype=torch.float32), num_steps=num_steps)
        rec_emb = inv.encode(recovered)
        out.append((recovered, rec_emb))
        print(f"    ...inverted {min(start + batch_size, len(defended))}/{len(defended)}")
    recovered = [r for rs, _ in out for r in rs]
    rec_embs = [e for _, es in out for e in es]
    return recovered, rec_embs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Defense privacy/utility tradeoff sweep.")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--collection", default="corpus")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--full", action="store_true", help="use the whole store (slow)")
    ap.add_argument("--sigmas", default="0,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5",
                    help="comma-separated Gaussian-noise levels")
    ap.add_argument("--k", type=int, default=5, help="recall@k for utility")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.store.exists():
        print(f"ERROR: no store at {args.store}. Run scripts/embed_corpus.py first.",
              file=sys.stderr)
        return 2
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip() != ""]

    adapter = ChromaAdapter(args.store, args.collection)
    all_samples = adapter.get_all()          # every stored vector, with a loud guard on empties
    # Full store: utility is always measured against ALL vectors so a target competes with
    # every real neighbor (a tiny subset would trivially retrieve itself → flat 1.0).
    full_base = np.array([s.vector for s in all_samples], dtype=np.float64)
    full_ids = [s.id for s in all_samples]
    idx_of = {s.id: i for i, s in enumerate(all_samples)}
    # Leakage eval set: the --limit stratified subset (inversion is the expensive half).
    limit = None if args.full else args.limit
    rows = stratified_sample(all_samples, limit, seed=args.seed)
    subset_idx = [idx_of[s.id] for s in rows]           # positions within the full array
    base = full_base[subset_idx]                        # clean subset → orig_emb for cosine
    row_dicts = [{"id": s.id, "type": s.type, "key_entities": s.key_entities} for s in rows]
    print(f"store has {len(all_samples)} vectors; leakage on {len(rows)} "
          f"({'full' if args.full else f'stratified subset, limit={args.limit}'}), "
          f"utility on all {len(all_samples)}; "
          f"sigmas={sigmas}, k={args.k}, num_steps={args.num_steps}\n")

    inv = get_inverter()
    results = []
    for sigma in sigmas:
        print(f"sigma={sigma}:")
        # Defend the WHOLE store once; the subset we invert is sliced from this same array.
        full_defended = defenses.gaussian_noise(full_base, sigma, seed=args.seed)
        # utility: clean queries vs the full defended store (all neighbors compete).
        # recall@1 (strict: nearest must be the target) alongside recall@k — @1 bends
        # earlier, exposing the utility cost sooner. Set → dedup when args.k == 1.
        util_ks = sorted({1, args.k})
        util = {f"utility_recall_at_{kk}":
                retrieval.recall_at_k(full_base, full_defended, full_ids, full_ids, kk)
                for kk in util_ks}
        # leakage: invert the subset's rows FROM the same defended array (coherent point)
        subset_defended = full_defended[subset_idx]
        recovered, rec_embs = _invert_all(inv, subset_defended, args.num_steps, args.batch_size)
        scores = [metrics.score_row(row_dicts[i], recovered[i], mode="ci",
                                    orig_emb=base[i], rec_emb=rec_embs[i])
                  for i in range(len(rows))]
        overall = sum(s.recall for s in scores) / len(scores)
        per_cat = metrics.per_category_recall(scores)
        results.append({"sigma": sigma, "leakage_recall": overall,
                        "leakage_per_category": per_cat, **util})
        print(f"  leakage(overall)={overall:.3f}  " +
              "  ".join(f"utility@{kk}={util[f'utility_recall_at_{kk}']:.3f}"
                        for kk in util_ks) + "\n")

    print("=" * 78)
    print(f"{'sigma':>7s} {'leakage':>8s} {f'utility@{args.k}':>10s}   per-category leakage")
    print("-" * 78)
    for r in results:
        pc = " ".join(f"{c[:4]}={v:.2f}" for c, v in r["leakage_per_category"].items())
        print(f"{r['sigma']:7.3f} {r['leakage_recall']:8.3f} "
              f"{r[f'utility_recall_at_{args.k}']:10.3f}   {pc}")
    print("=" * 78)
    _print_tradeoff_summary(results, args.k)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"k": args.k, "num_steps": args.num_steps, "rows": len(rows),
         "utility_store_size": len(all_samples), "seed": args.seed,
         "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print("NOTE: no leak threshold applied — raw leakage vs utility for you to read.")
    return 0


def _print_tradeoff_summary(results, k):
    """Describe the measured tradeoff without imposing a threshold verdict.

    Reports the sigma with the widest privacy/utility margin (utility - leakage) as a
    descriptive pointer, plus how far leakage and utility actually moved across the grid.
    """
    util_key = f"utility_recall_at_{k}"
    best = max(results, key=lambda r: r[util_key] - r["leakage_recall"])
    gap = best[util_key] - best["leakage_recall"]
    leaks = [r["leakage_recall"] for r in results]
    utils = [r[util_key] for r in results]
    print(f"leakage ranged {min(leaks):.2f}–{max(leaks):.2f}; "
          f"utility@{k} ranged {min(utils):.2f}–{max(utils):.2f}.")
    print(f"widest privacy/utility margin: sigma={best['sigma']} "
          f"(utility={best[util_key]:.2f}, leakage={best['leakage_recall']:.2f}, gap={gap:.2f}).")
    if max(utils) - min(utils) < 0.05:
        print("  note: utility barely moved on this grid (measured over the full store) — "
              "the noise is too small to disrupt retrieval, or the corpus is well-separated "
              "enough to survive it. Extend sigmas upward to find where utility degrades; "
              "read the whole curve, not this one line.")


if __name__ == "__main__":
    sys.exit(main())
