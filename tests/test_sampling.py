"""Tests for stratified_sample — the balanced subset the demo's --limit relies on.
Fast: synthetic Samples, no model, no store.
"""
from leaklens.adapters.base import Sample, stratified_sample


def _make(types_counts):
    """Build Samples: e.g. {'plain':60,'pii':60} -> 60 plain + 60 pii samples."""
    out = []
    for t, n in types_counts.items():
        out.extend(Sample(id=f"{t}-{i}", vector=[0.0], type=t) for i in range(n))
    return out


def test_returns_all_when_limit_zero_or_negative_or_too_big():
    s = _make({"plain": 3, "pii": 3})
    assert len(stratified_sample(s, 0)) == 6
    assert len(stratified_sample(s, -1)) == 6
    assert len(stratified_sample(s, 999)) == 6
    assert len(stratified_sample(s, None)) == 6


def test_balances_across_categories():
    s = _make({"plain": 60, "pii": 60, "credential": 60, "structured": 60})
    picked = stratified_sample(s, 40, seed=42)
    assert len(picked) == 40
    counts = {}
    for p in picked:
        counts[p.type] = counts.get(p.type, 0) + 1
    # 40 across 4 categories, round-robin -> 10 each
    assert counts == {"plain": 10, "pii": 10, "credential": 10, "structured": 10}


def test_reproducible_under_seed():
    s = _make({"plain": 60, "pii": 60, "credential": 60, "structured": 60})
    a = [x.id for x in stratified_sample(s, 20, seed=7)]
    b = [x.id for x in stratified_sample(s, 20, seed=7)]
    assert a == b


def test_handles_category_smaller_than_its_share():
    # credential has only 2; a limit of 12 should still return 12, drawing extra from the
    # larger categories once credential is exhausted.
    s = _make({"plain": 20, "pii": 20, "credential": 2})
    picked = stratified_sample(s, 12, seed=1)
    assert len(picked) == 12
    counts = {}
    for p in picked:
        counts[p.type] = counts.get(p.type, 0) + 1
    assert counts["credential"] == 2          # all of the small category, no more
    assert counts["plain"] + counts["pii"] == 10
