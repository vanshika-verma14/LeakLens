"""Retrieval utility measurement — the other half of the tradeoff.

`recall_at_k` answers "does the store still return the right chunk?" after a defense is
applied. The realistic RAG model (DECISIONS, Phase-2): the defense hardens the *stored*
index, while a live query is embedded fresh and clean — so the caller passes clean query
vectors against the defended store vectors. Pure numpy, no model.
"""
from __future__ import annotations

import numpy as np


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)         # zero-norm rows -> leave as zeros
    return matrix / norms


def recall_at_k(query_vecs, store_vecs, store_ids: list, target_ids: list, k: int) -> float:
    """Fraction of queries whose `target_id` appears in the top-k nearest store vectors.

    Similarity is cosine. `store_ids[j]` labels store row j; `target_ids[i]` is the id row i
    should retrieve. k is clamped to the number of stored vectors.
    """
    q = _unit(np.asarray(query_vecs, dtype=np.float64))
    s = _unit(np.asarray(store_vecs, dtype=np.float64))
    if q.shape[0] == 0 or s.shape[0] == 0:
        return 0.0
    k = max(1, min(k, s.shape[0]))
    sims = q @ s.T                                   # (n_queries, n_store)
    topk = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    hits = 0
    for i, target in enumerate(target_ids):
        if any(store_ids[j] == target for j in topk[i]):
            hits += 1
    return hits / len(target_ids)
