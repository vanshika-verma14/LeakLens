"""Render the defense privacy/utility tradeoff from results/sweep.json → tradeoff.png.

Pure and fast: reads the sweep's JSON and draws it. No torch, no encoder, no model load —
this is the presentation layer for the numbers `defense_sweep.py` already measured.

The plot's whole job is to make ONE thing obvious: the "comfortable middle" — the band of
Gaussian-noise σ where leakage has already collapsed to 0 but retrieval utility is still
~1.0. That band is the finding. Its definition is not hidden in the code: the leakage and
utility cutoffs are CLI args (`--leak-max`, `--util-min`), printed on the figure, so the
shaded region always states the rule that produced it (project rule: no threshold chosen
silently).

Utility lines are auto-discovered from every `utility_recall_at_N` key in the JSON, so a
sweep that emits both recall@1 and recall@5 plots both (recall@1 bends earlier and shows
the utility cost sooner) with no change here.

Run:  python studies/plot_tradeoff.py
      python studies/plot_tradeoff.py --in results/sweep.json --out results/tradeoff.png
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # headless: write a file, never open a window
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "results" / "sweep.json"
DEFAULT_OUT = ROOT / "results" / "tradeoff.png"

_UTIL_KEY = re.compile(r"^utility_recall_at_(\d+)$")


def _utility_series(results):
    """Return {k: [values aligned to results]} for every utility_recall_at_k key present."""
    keys = sorted(
        {key for row in results for key in row if _UTIL_KEY.match(key)},
        key=lambda key: int(_UTIL_KEY.match(key).group(1)),
    )
    return {int(_UTIL_KEY.match(key).group(1)): [row.get(key) for row in results]
            for key in keys}


def _comfortable_band(sigmas, leakage, util_series, leak_max, util_min):
    """Contiguous σ-range where leakage <= leak_max AND every utility line >= util_min.

    Returns (lo_sigma, hi_sigma) or None. "Contiguous" over the sorted σ grid: we take the
    longest run of qualifying points so a single noisy dip doesn't split the band.
    """
    def ok(i):
        if leakage[i] is None or leakage[i] > leak_max:
            return False
        return all(vals[i] is not None and vals[i] >= util_min
                   for vals in util_series.values())

    best = cur = None
    for i in range(len(sigmas)):
        if ok(i):
            cur = (cur[0] if cur else i, i)
            if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                best = cur
        else:
            cur = None
    if best is None:
        return None
    return sigmas[best[0]], sigmas[best[1]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Plot the defense privacy/utility tradeoff.")
    ap.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--leak-max", type=float, default=0.0,
                    help="max leakage recall counted as 'no leak' for the band (default 0.0)")
    ap.add_argument("--util-min", type=float, default=0.99,
                    help="min utility recall counted as '~1.0' for the band (default 0.99)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args(argv)

    if not args.inp.exists():
        print(f"ERROR: no sweep at {args.inp}. Run studies/defense_sweep.py first.",
              file=sys.stderr)
        return 2

    data = json.loads(args.inp.read_text(encoding="utf-8"))
    results = sorted(data.get("results", []), key=lambda r: r["sigma"])
    if not results:
        print(f"ERROR: {args.inp} has no results to plot.", file=sys.stderr)
        return 2

    sigmas = [r["sigma"] for r in results]
    leakage = [r.get("leakage_recall") for r in results]
    util_series = _utility_series(results)
    store_n = data.get("utility_store_size", "?")

    fig, ax = plt.subplots(figsize=(8, 5))

    band = _comfortable_band(sigmas, leakage, util_series, args.leak_max, args.util_min)
    if band is not None:
        lo, hi = band
        ax.axvspan(lo, hi, color="tab:green", alpha=0.12, zorder=0)
        ax.annotate(
            f"comfortable middle\nσ∈[{lo:g}, {hi:g}]\nleakage ≤{args.leak_max:g}, "
            f"utility ≥{args.util_min:g}",
            xy=((lo + hi) / 2, 0.5), ha="center", va="center", fontsize=8,
            color="green", weight="bold",
            bbox=dict(boxstyle="round", fc="white", ec="tab:green", alpha=0.85))
    else:
        # lower-left open area — clear of the descending utility lines and the legend
        ax.text(0.30, 0.24, f"no comfortable band\nat leakage ≤{args.leak_max:g}, "
                            f"utility ≥{args.util_min:g}",
                transform=ax.transAxes, ha="center", va="center", fontsize=9,
                color="darkred",
                bbox=dict(boxstyle="round", fc="white", ec="darkred", alpha=0.85))

    ax.plot(sigmas, leakage, marker="o", color="tab:red", lw=2,
            label="leakage (key-entity recall)")
    for k, vals in util_series.items():
        ax.plot(sigmas, vals, marker="s", lw=2,
                label=f"utility@{k} (recall over {store_n} store)")

    ax.set_xlabel("Gaussian noise σ applied to stored embeddings")
    ax.set_ylabel("recall")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=min(sigmas))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9)

    ax.set_title("LeakLens — embedding-inversion defense tradeoff", fontsize=13, pad=22)
    caption = (
        f"PREVIEW — num_steps={data.get('num_steps', '?')}, "
        f"n={data.get('rows', '?')} leakage rows, utility over N={store_n} store · "
        f"{len(results)} σ points · seed {data.get('seed', '?')}.  "
        f"Not final: regenerate from the --full run (num_steps=20) before citing.")
    ax.text(0.5, 1.015, caption, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color="gray", style="italic")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    if band is not None:
        print(f"comfortable band: sigma in [{band[0]:g}, {band[1]:g}] "
              f"(leakage <={args.leak_max:g}, utility >={args.util_min:g})")
    else:
        print(f"no comfortable band at leakage <={args.leak_max:g}, "
              f"utility >={args.util_min:g}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
