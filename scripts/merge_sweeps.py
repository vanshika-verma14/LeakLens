"""Merge chunked defense-sweep JSONs into one sorted sweep file.

A long --full sweep can be split across sessions or machines by running
studies/defense_sweep.py with different --sigmas into DIFFERENT --out files
(the built-in resume only covers chunks that share one --out). This stitches
those chunk files back into the single JSON one uninterrupted run would have
produced -- byte-identical, guarded by tests/test_defense_sweep.py.

Refusals (both would silently mix incomparable points into one curve):
  * inputs from different run configurations (num_steps / rows / k / seed /
    utility-store size) -- same guard defense_sweep applies on resume;
  * the same sigma appearing in two inputs with different values. Identical
    duplicates (an overlap between chunks) merge fine.

Run:  python scripts/merge_sweeps.py results/sweep_a.json results/sweep_b.json --out results/sweep.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# The run-configuration keys defense_sweep.py guards on resume, in the exact
# order it writes them -- key order matters for byte-identity with its output.
META_KEYS = ("k", "num_steps", "rows", "utility_store_size", "seed")


def _atomic_write_json(path: Path, doc: dict) -> None:
    # Must serialize exactly like defense_sweep._atomic_write_json, or a merged
    # file could never be byte-identical to a single-run file.
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def merge(docs: list[tuple[str, dict]]) -> dict:
    """Merge (name, sweep-doc) pairs into one doc; ValueError on any refusal."""
    ref_name, ref = docs[0]
    for name, doc in docs[1:]:
        mismatched = {key: (ref.get(key), doc.get(key)) for key in META_KEYS
                      if doc.get(key) != ref.get(key)}
        if mismatched:
            detail = ", ".join(f"{k}: {a!r} vs {b!r}"
                               for k, (a, b) in sorted(mismatched.items()))
            raise ValueError(f"{name} came from a different run configuration "
                             f"than {ref_name} ({detail})")
    by_sigma: dict[float, tuple[str, dict]] = {}
    for name, doc in docs:
        for r in doc.get("results", []):
            prev = by_sigma.setdefault(r["sigma"], (name, r))
            if prev[1] != r:
                raise ValueError(f"sigma={r['sigma']} has conflicting values in "
                                 f"{prev[0]} and {name}")
    return {**{key: ref.get(key) for key in META_KEYS},
            "results": sorted((r for _, r in by_sigma.values()),
                              key=lambda r: r["sigma"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Merge chunked defense-sweep JSONs into one sorted file.")
    ap.add_argument("inputs", nargs="+", type=Path, help="sweep JSON chunk files")
    ap.add_argument("--out", type=Path, required=True, help="merged JSON to write")
    args = ap.parse_args(argv)

    docs = []
    for path in args.inputs:
        if not path.exists():
            print(f"ERROR: {path} does not exist.", file=sys.stderr)
            return 2
        docs.append((str(path), json.loads(path.read_text(encoding="utf-8"))))
    try:
        merged = merge(docs)
    except ValueError as err:
        print(f"ERROR: {err}. Refusing to merge -- fix the inputs instead.",
              file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(args.out, merged)
    sigmas = [r["sigma"] for r in merged["results"]]
    print(f"merged {len(docs)} file(s) -> {args.out} "
          f"({len(sigmas)} sigma(s): {sigmas})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
