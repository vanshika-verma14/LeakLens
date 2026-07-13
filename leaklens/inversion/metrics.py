"""The recovery metric — what "leaked" means (DECISIONS D5).

Primary signal: **key-entity recall** — the fraction of a row's `key_entities` that appear
in the recovered text. "Appears" defaults to **case-insensitive substring** (`mode="ci"`):
it catches recapitalization (a real `library`->`Library` recovery that exact-match would
miss) without the false-positive risk of fuzzy matching, where a *wrong* value (recovered
phone 555-0291 vs real 555-0112) scores high similarity and would be miscounted as leaked.

Secondary signal: **cosine similarity** of the original vs recovered embeddings, reported
as context only. It is deliberately not the verdict: cosine stays high (0.88-0.99 in real
runs) even when the actual secret is destroyed, so it overstates leakage.

There is **no leak threshold in this module on purpose.** It emits raw recall; a human sets
the threshold after reviewing `recall_distribution` (see docs/DECISIONS.md).

Known limitation (documented, not hidden): under `ci`, a common-word *label* like
"Order"/"Invoice"/"Transaction" can match coincidentally — the recovered text may contain
that word without having recovered that specific record. So label recall can overstate
leakage; the honest signal is recovery of the high-entropy *value*. This is exactly why
labels and values are tagged separately (corpus D5): a reader can look at value recovery,
not just the headline recall. See tests/test_metrics.py::test_common_word_label_*. A
future refinement could word-boundary-match short labels, but the chosen semantics here
are plain substring — we surface the caveat rather than silently change the metric.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from statistics import mean

MODES = ("exact", "ci", "fuzzy")
DEFAULT_MODE = "ci"
DEFAULT_FUZZY_THRESHOLD = 0.8


@dataclass
class RecoveryScore:
    """Per-row recovery result. `recall` is the primary number (using `mode`)."""

    id: str
    type: str | None
    recall: float
    hits: int
    total: int
    matched: list[str]
    missed: list[str]
    cosine: float | None = None      # secondary / context; None if embeddings not supplied
    details: dict = field(default_factory=dict)   # {"exact": r, "ci": r, "fuzzy": r}


def entity_found(entity: str, recovered: str, *, mode: str = DEFAULT_MODE,
                 fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD) -> bool:
    """Is `entity` present in `recovered` under the given match mode?

    - exact: case-sensitive substring.
    - ci:    case-insensitive substring (the primary/default).
    - fuzzy: best sliding-window difflib ratio >= fuzzy_threshold (opt-in; can count a
             near-miss WRONG value as found, so it is not the default).
    """
    if mode not in MODES:
        raise ValueError(f"unknown match mode {mode!r}; expected one of {MODES}")
    if entity == "":
        return False
    if mode == "exact":
        return entity in recovered
    if mode == "ci":
        return entity.lower() in recovered.lower()
    return _fuzzy_ratio(entity, recovered) >= fuzzy_threshold


def _fuzzy_ratio(entity: str, recovered: str) -> float:
    """Best case-insensitive difflib ratio of `entity` against any equal-length window."""
    e, t = entity.lower(), recovered.lower()
    span = len(e)
    if span == 0 or len(t) < span:
        return difflib.SequenceMatcher(None, e, t).ratio()
    best = 0.0
    for i in range(len(t) - span + 1):
        best = max(best, difflib.SequenceMatcher(None, e, t[i:i + span]).ratio())
    return best


def key_entity_recall(key_entities: list[str], recovered: str, *,
                      mode: str = DEFAULT_MODE,
                      fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
                      ) -> tuple[float, list[str], list[str]]:
    """Return (recall, matched, missed) for one row's entities under `mode`.

    Recall of an entity-less row is 0.0 (nothing to recover -> nothing recovered).
    """
    if not key_entities:
        return 0.0, [], []
    matched, missed = [], []
    for ent in key_entities:
        (matched if entity_found(ent, recovered, mode=mode,
                                 fuzzy_threshold=fuzzy_threshold) else missed).append(ent)
    return len(matched) / len(key_entities), matched, missed


def cosine_similarity(a, b) -> float:
    """Cosine similarity of two vectors (lists or 1-D tensors). Secondary/context only."""
    av = [float(x) for x in a]
    bv = [float(x) for x in b]
    dot = sum(x * y for x, y in zip(av, bv))
    na = sum(x * x for x in av) ** 0.5
    nb = sum(y * y for y in bv) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def score_row(row: dict, recovered: str, *, mode: str = DEFAULT_MODE,
              fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
              orig_emb=None, rec_emb=None) -> RecoveryScore:
    """Score one corpus row against its recovered text.

    `row` is a dict with id / type / key_entities (a corpus row or a Sample's fields).
    Cosine is computed only if both embeddings are supplied — the metric never re-encodes
    on its own, so it is usable without loading the encoder.
    """
    ents = row.get("key_entities", [])
    recall, matched, missed = key_entity_recall(
        ents, recovered, mode=mode, fuzzy_threshold=fuzzy_threshold)
    details = {m: key_entity_recall(ents, recovered, mode=m,
                                    fuzzy_threshold=fuzzy_threshold)[0] for m in MODES}
    cosine = None
    if orig_emb is not None and rec_emb is not None:
        cosine = cosine_similarity(orig_emb, rec_emb)
    return RecoveryScore(
        id=row.get("id", "<no-id>"), type=row.get("type"), recall=recall,
        hits=len(matched), total=len(ents), matched=matched, missed=missed,
        cosine=cosine, details=details)


def per_category_recall(scores: list[RecoveryScore]) -> dict[str, float]:
    """Mean recall grouped by row `type` (the per-category finding). Sorted by type."""
    buckets: dict[str, list[float]] = {}
    for s in scores:
        buckets.setdefault(s.type or "unknown", []).append(s.recall)
    return {t: mean(v) for t, v in sorted(buckets.items())}


def _summary(values: list[float]) -> dict[str, float]:
    vs = sorted(values)
    n = len(vs)

    def pct(p: float) -> float:
        if n == 1:
            return vs[0]
        idx = p * (n - 1)
        lo = int(idx)
        frac = idx - lo
        hi = min(lo + 1, n - 1)
        return vs[lo] + (vs[hi] - vs[lo]) * frac

    return {"n": n, "min": vs[0], "p25": pct(0.25), "median": pct(0.5),
            "mean": mean(vs), "p75": pct(0.75), "max": vs[-1]}


def recall_distribution(scores: list[RecoveryScore], *, by_category: bool = True) -> dict:
    """Raw recall spread (min/p25/median/mean/p75/max) overall and per category.

    Deliberately threshold-free: this is the evidence a human reviews to CHOOSE the leak
    threshold, not a verdict.
    """
    if not scores:
        return {"overall": None}
    out = {"overall": _summary([s.recall for s in scores])}
    if by_category:
        cats: dict[str, list[float]] = {}
        for s in scores:
            cats.setdefault(s.type or "unknown", []).append(s.recall)
        out["by_category"] = {t: _summary(v) for t, v in sorted(cats.items())}
    return out
