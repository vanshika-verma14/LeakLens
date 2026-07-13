"""Shared pytest fixtures for LeakLens.

Importing the compat shims first means any test that later reaches vec2text works
on Windows without each test repeating the stub.
"""
import os
import sys

# Slow tests exercise pre-cached models. Default them to offline so a broken cert store
# or absent network can't fail them. This MUST run before anything imports
# huggingface_hub (it reads these vars at import time), hence the argv peek at the very
# top of conftest, before the test modules that import vec2text are collected. Override
# with HF_HUB_OFFLINE=0 to allow a download.
if "--runslow" in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import leaklens._compat  # noqa: F401,E402  (installs resource stub + CPU pin on import)

import pytest  # noqa: E402


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="run tests marked 'slow' (real ~2GB vec2text inversion)")


def pytest_collection_modifyitems(config, items):
    """Skip `slow` tests unless --runslow is given, so the everyday suite stays fast."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow (real inversion is slow)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

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
