"""Shared pytest fixtures for LeakLens.

Importing the compat shims first means any test that later reaches vec2text works
on Windows without each test repeating the stub.
"""
import leaklens._compat  # noqa: F401  (installs resource stub + CPU pin on import)

import pytest

# A tiny, dependency-free corpus for fast plumbing tests. Mirrors the
# corpus.jsonl schema (text / type / key_entities) without needing the real file
# or an embedding model.
_TINY_CORPUS = [
    {
        "text": "The quarterly report shows a 12 percent increase in revenue.",
        "type": "plain",
        "key_entities": ["quarterly report", "12 percent", "revenue"],
    },
    {
        "text": "Contact Priya Sharma at priya.sharma@acme.com for the invoice.",
        "type": "pii",
        "key_entities": ["Priya Sharma", "priya.sharma@acme.com"],
    },
    {
        "text": "The launch code for project atlas is 7719.",
        "type": "secret",
        "key_entities": ["project atlas", "7719"],
    },
]


@pytest.fixture
def tiny_corpus():
    """Return a small in-memory corpus (list of dicts) for plumbing tests."""
    return [dict(row) for row in _TINY_CORPUS]


@pytest.fixture
def temp_store_dir(tmp_path):
    """A throwaway directory for a temporary vector store (auto-cleaned per test)."""
    d = tmp_path / "store"
    d.mkdir()
    return d
