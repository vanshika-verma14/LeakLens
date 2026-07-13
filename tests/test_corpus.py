"""Guards the ground-truth corpus: schema, verbatim key_entities, fake-data, counts.

If any of these break, the recovery metric downstream would be scoring against a
corrupt oracle — so this is a load-bearing test, not a formality.
"""
import json
from pathlib import Path

import pytest

from corpus import build_corpus

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "corpus.jsonl"


def _load():
    if not CORPUS_PATH.exists():
        pytest.skip("corpus/corpus.jsonl not generated yet — run corpus/build_corpus.py")
    return [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generated_corpus_validates():
    """A freshly built corpus passes its own validator with zero problems."""
    rows = build_corpus.build(seed=42)
    assert build_corpus.validate(rows) == []


def test_build_is_deterministic():
    """Same seed reproduces the corpus exactly (NFR-3 reproducibility)."""
    assert build_corpus.build(seed=42) == build_corpus.build(seed=42)


def test_checked_in_corpus_is_clean():
    """The committed corpus.jsonl still satisfies every rule."""
    rows = _load()
    assert build_corpus.validate(rows) == []


def test_category_counts():
    rows = _load()
    counts = {c: sum(1 for r in rows if r["type"] == c) for c in build_corpus.CATEGORIES}
    assert counts == {c: build_corpus.TARGET_PER_CAT for c in build_corpus.CATEGORIES}


def test_every_entity_is_a_substring():
    rows = _load()
    for r in rows:
        for ent in r["key_entities"]:
            assert ent in r["text"], f"{r['id']}: {ent!r} not in text"
