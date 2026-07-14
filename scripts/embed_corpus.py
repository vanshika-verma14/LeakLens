"""Embed corpus/corpus.jsonl with the GTR encoder and load it into a Chroma store.

One-time builder for the inversion demo (T1.6) and later the validation matrix. Embedding
is cheap (seconds for all 240 rows) — only inversion is slow — so this always embeds the
whole corpus. Rebuilds from scratch by default so re-running is idempotent.

Run:  python scripts/embed_corpus.py            # -> results/corpus_store (count 240)
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Models are pre-cached; this machine's cert store blocks huggingface.co (see conftest /
# memory). Default to offline so loading the encoder works; override with HF_HUB_OFFLINE=0.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `leaklens` importable

from leaklens.adapters.base import Sample
from leaklens.adapters.chroma_adapter import ChromaAdapter
from leaklens.inversion.inverter import get_inverter

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "corpus" / "corpus.jsonl"
DEFAULT_OUT = ROOT / "results" / "corpus_store"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Embed the corpus into a Chroma store.")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--collection", default="corpus")
    ap.add_argument("--append", action="store_true",
                    help="add to an existing store instead of rebuilding from scratch")
    args = ap.parse_args(argv)

    rows = load_rows(args.corpus)
    print(f"loaded {len(rows)} rows from {args.corpus}")

    if args.out.exists() and not args.append:
        shutil.rmtree(args.out)  # fresh build so re-runs don't duplicate ids
    args.out.mkdir(parents=True, exist_ok=True)

    print("encoding with GTR (this is the fast part)...")
    emb = get_inverter().encode([r["text"] for r in rows])
    samples = [Sample(id=r["id"], vector=emb[i].tolist(), text=r["text"],
                      type=r["type"], key_entities=r["key_entities"])
               for i, r in enumerate(rows)]

    adapter = ChromaAdapter(args.out, args.collection)
    adapter.add(samples)
    print(f"count: {adapter.count()}  ->  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
