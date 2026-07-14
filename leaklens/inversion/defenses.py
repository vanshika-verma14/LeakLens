"""Embedding defenses for the tradeoff study — dimension-preserving on purpose.

Each defense transforms stored embeddings to make them harder to invert. We keep the
768-dim GTR shape because the leakage side of the tradeoff feeds the defended vectors back
through vec2text, which only inverts a valid GTR embedding. Dimensionality-reduction /
pooling defenses are real but excluded here: their leakage can't be measured with our
inverter, and a half-measured point would undercut the whole point of the study
(see docs/DECISIONS.md, Phase-2 notes).

All defenses are pure and reproducible under `seed`.
"""
from __future__ import annotations

import numpy as np


def gaussian_noise(vectors, sigma: float, *, seed: int | None = None) -> np.ndarray:
    """Add isotropic Gaussian noise N(0, sigma) to each component.

    sigma=0 returns the input unchanged (the undefended baseline). GTR embeddings are
    roughly unit-norm (per-dim std ~0.036), so meaningful sigma lives near that scale.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if sigma <= 0:
        return arr.copy()
    rng = np.random.default_rng(seed)
    return arr + rng.normal(0.0, sigma, size=arr.shape)


def quantize(vectors, levels: int) -> np.ndarray:
    """Round each component to one of `levels` uniform steps between its column min/max.

    Fewer levels = coarser storage = less recoverable detail. `levels<=1` collapses to the
    per-column midpoint; very large `levels` is effectively identity. Shape preserved.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if levels >= 2 ** 16:
        return arr.copy()
    lo = arr.min(axis=0, keepdims=True)
    hi = arr.max(axis=0, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)          # avoid /0 on constant columns
    if levels <= 1:
        return np.broadcast_to((lo + hi) / 2.0, arr.shape).copy()
    steps = levels - 1
    normalized = (arr - lo) / span                  # -> [0, 1]
    return lo + np.round(normalized * steps) / steps * span
