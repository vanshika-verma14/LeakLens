"""Tests for recall@k retrieval utility — fast, no model."""
import numpy as np

from leaklens.inversion import retrieval

# Four well-separated store vectors.
STORE = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0]])
IDS = ["a", "b", "c", "d"]


def test_identical_query_hits_at_k1():
    # querying with each store vector itself retrieves it at k=1
    r = retrieval.recall_at_k(STORE, STORE, IDS, IDS, k=1)
    assert r == 1.0


def test_recall_at_k_counts_target_in_topk():
    # query near 'a' but slightly toward 'd'; at k=1 may miss, at k=2 should include 'a'
    q = np.array([[0.9, 0.3, 0.0]])
    assert retrieval.recall_at_k(q, STORE, IDS, ["a"], k=1) in (0.0, 1.0)
    assert retrieval.recall_at_k(q, STORE, IDS, ["a"], k=2) == 1.0


def test_recall_drops_when_target_is_perturbed_away():
    # defend 'a' by moving its stored vector far from the clean query -> misses at k=1
    clean_query = np.array([[1.0, 0.0, 0.0]])
    defended_store = STORE.copy().astype(float)
    defended_store[0] = [0.0, 0.0, 1.0]          # 'a' now points where 'c' was
    r = retrieval.recall_at_k(clean_query, defended_store, IDS, ["a"], k=1)
    assert r == 0.0


def test_k_is_clamped_to_store_size():
    r = retrieval.recall_at_k(STORE, STORE, IDS, IDS, k=999)
    assert r == 1.0


def test_empty_inputs_return_zero():
    assert retrieval.recall_at_k(np.empty((0, 3)), STORE, IDS, [], k=1) == 0.0
