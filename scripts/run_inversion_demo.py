"""The Slice-1 spine: Chroma store -> sample -> invert -> score -> print.

Defaults to a small stratified subset (--limit 30) so you can confirm the pipeline and the
per-category recovery shape in minutes. The full 240-row run (~2-3 h on CPU) is opt-in via
--full. No leak threshold is applied here — the point is to emit the raw per-category
recall distribution so a human can choose the threshold later (DECISIONS D5).

Run:  python scripts/run_inversion_demo.py --limit 30
      python scripts/run_inversion_demo.py --full            # the long run
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # cached models; cert store blocks HF
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leaklens.adapters.base import stratified_sample
from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.inversion import metrics
from leaklens.inversion.inverter import get_inverter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORE = ROOT / "results" / "corpus_store"
DEFAULT_OUT = ROOT / "results" / "inversion_demo.json"


def _truncate(s: str, n: int = 60) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inversion demo spine over a Chroma store.")
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--collection", default="corpus")
    ap.add_argument("--limit", type=int, default=30, help="rows to invert (default 30)")
    ap.add_argument("--full", action="store_true", help="invert the whole store (slow)")
    ap.add_argument("--num-steps", type=int, default=20, help="vec2text correction steps")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.store.exists():
        print(f"ERROR: no store at {args.store}. Run scripts/embed_corpus.py first.",
              file=sys.stderr)
        return 2
    adapter = ChromaAdapter(args.store, args.collection)
    total = adapter.count()
    if total == 0:
        print(f"ERROR: store {args.store} is empty. Run scripts/embed_corpus.py first.",
              file=sys.stderr)
        return 2

    all_samples = adapter.sample(total, seed=args.seed)
    limit = None if args.full else args.limit
    rows = stratified_sample(all_samples, limit, seed=args.seed)
    print(f"store has {total} vectors; inverting {len(rows)} "
          f"({'full corpus' if args.full else f'stratified subset, limit={args.limit}'}), "
          f"num_steps={args.num_steps}\n")

    inv = get_inverter()
    scores = []
    import torch  # local: only needed once models are in play
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        emb = torch.tensor([s.vector for s in batch])
        recovered = inv.invert(emb, num_steps=args.num_steps)
        rec_emb = inv.encode(recovered)
        for i, s in enumerate(batch):
            row = {"id": s.id, "type": s.type, "key_entities": s.key_entities}
            score = metrics.score_row(row, recovered[i], mode="ci",
                                      orig_emb=s.vector, rec_emb=rec_emb[i])
            scores.append((score, s.text, recovered[i]))
        print(f"  ...inverted {min(start + args.batch_size, len(rows))}/{len(rows)}")

    print("\n" + "=" * 100)
    print(f"{'id':12s} {'type':11s} {'recall':>6s} {'cos':>5s}  original -> recovered")
    print("-" * 100)
    for score, original, recovered in scores:
        cos = f"{score.cosine:.2f}" if score.cosine is not None else "  - "
        print(f"{score.id:12s} {score.type or '-':11s} {score.recall:6.2f} {cos:>5s}  "
              f"{_truncate(original)}  ->  {_truncate(recovered)}")

    plain_scores = [s for s, _, _ in scores]
    print("\nper-category recall (mean):")
    for cat, r in metrics.per_category_recall(plain_scores).items():
        print(f"  {cat:11s} {r:.3f}")

    dist = metrics.recall_distribution(plain_scores)
    print("\nrecall distribution (threshold-free — you set the leak threshold from this):")
    _print_summary("overall", dist["overall"])
    for cat, summ in dist.get("by_category", {}).items():
        _print_summary(f"  {cat}", summ)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"id": s.id, "type": s.type, "recall": s.recall, "hits": s.hits,
                "total": s.total, "matched": s.matched, "missed": s.missed,
                "cosine": s.cosine, "details": s.details}
               for s, _, _ in scores]
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(payload)} scores to {args.out}")
    print("NOTE: no leak threshold applied — this is raw recall for you to threshold.")
    return 0


def _print_summary(label, summ):
    if summ is None:
        print(f"  {label}: (none)")
        return
    print(f"  {label:12s} n={summ['n']:<3d} min={summ['min']:.2f} p25={summ['p25']:.2f} "
          f"median={summ['median']:.2f} mean={summ['mean']:.2f} p75={summ['p75']:.2f} "
          f"max={summ['max']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
