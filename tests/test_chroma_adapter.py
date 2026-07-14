"""Round-trip tests for the Chroma adapter — fast, no model load (hand-made vectors).

The load-bearing assertion is that id / text / type / key_entities survive the store
intact, especially key_entities as a *list* (it is JSON-serialized inside Chroma).
Without that, per-category recovery scoring in T1.4 has no ground truth to key off.
"""
import pytest

from leaklens.adapters.base import Sample
from leaklens.adapters.chroma_adapter import ChromaAdapter

SAMPLES = [
    Sample(id="plain-001", vector=[0.1, 0.2, 0.3, 0.4],
           text="The library extended its opening hours.", type="plain",
           key_entities=["library", "opening hours"]),
    Sample(id="pii-001", vector=[0.5, 0.4, 0.3, 0.2],
           text="Email Priya Sharma at priya.sharma@example.com.", type="pii",
           key_entities=["Priya Sharma", "priya.sharma@example.com"]),
    Sample(id="cred-001", vector=[0.9, 0.1, 0.1, 0.1],
           text="Reset the admin password to Th1sIsSecret!.", type="credential",
           key_entities=["admin password", "Th1sIsSecret!"]),
    Sample(id="struct-001", vector=[0.2, 0.2, 0.7, 0.1],
           text="Order ORD-2026-00817 shipped on 2026-03-14.", type="structured",
           key_entities=["Order", "ORD-2026-00817", "2026-03-14"]),
]


@pytest.fixture
def adapter(temp_store_dir):
    a = ChromaAdapter(temp_store_dir, "docs")
    a.add(SAMPLES)
    return a


def test_count_matches(adapter):
    assert adapter.count() == len(SAMPLES)


def test_roundtrip_preserves_metadata(adapter):
    by_id = {s.id: s for s in adapter.sample(len(SAMPLES))}
    assert set(by_id) == {s.id for s in SAMPLES}
    for original in SAMPLES:
        got = by_id[original.id]
        assert got.text == original.text
        assert got.type == original.type
        # the crux: key_entities comes back as an intact list (survived JSON in Chroma)
        assert got.key_entities == original.key_entities
        assert isinstance(got.key_entities, list)
        assert got.vector == pytest.approx(original.vector, abs=1e-6)


def test_sample_more_than_count_returns_all(adapter):
    assert len(adapter.sample(999)) == len(SAMPLES)


def test_sampling_is_reproducible(adapter):
    first = [s.id for s in adapter.sample(2, seed=42)]
    second = [s.id for s in adapter.sample(2, seed=42)]
    assert first == second
    assert len(first) == 2


def test_empty_store_samples_nothing(temp_store_dir):
    empty = ChromaAdapter(temp_store_dir, "empty")
    assert empty.count() == 0
    assert empty.sample(5) == []


def test_get_all_returns_all_samples_with_vectors(temp_store_dir):
    # The regression guard: get_all() must return every row WITH a non-empty embedding of
    # the right dimension. A store that silently drops embeddings would fail here, not
    # 30 minutes into a --full sweep with a numpy AxisError.
    a = ChromaAdapter(temp_store_dir, "docs")
    a.add(SAMPLES[:3])
    got = a.get_all()
    assert len(got) == 3
    for s in got:
        assert s.vector, "get_all() must return a non-empty embedding"
        assert len(s.vector) == 4            # SAMPLES are 4-dim
    assert {s.id for s in got} == {s.id for s in SAMPLES[:3]}


def test_get_all_raises_when_embeddings_missing(temp_store_dir):
    # If Chroma hands back no embeddings, fail loudly instead of a degenerate 0-vector store.
    a = ChromaAdapter(temp_store_dir, "docs")
    a.add(SAMPLES[:3])
    a._col.get(include=["embeddings", "metadatas", "documents"])  # sanity: real path works
    a._col = _NoEmbeddingCollection(a._col)
    with pytest.raises(RuntimeError, match="embeddings not returned"):
        a.get_all()


class _NoEmbeddingCollection:
    """Wraps a real collection but strips embeddings from get() — simulates a Chroma
    version/call that doesn't return them, exercising the get_all() guard."""

    def __init__(self, inner):
        self._inner = inner

    def get(self, *args, **kwargs):
        got = dict(self._inner.get(*args, **kwargs))
        got["embeddings"] = None
        return got
