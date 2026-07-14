"""Tests for embedding defenses — fast, no model."""
import numpy as np

from leaklens.inversion import defenses

BASE = np.array([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.4, 0.3, 0.2],
                 [0.9, 0.1, 0.1, 0.8]])


def test_gaussian_sigma_zero_is_identity():
    out = defenses.gaussian_noise(BASE, 0.0)
    assert np.allclose(out, BASE)


def test_gaussian_changes_vectors_and_preserves_shape():
    out = defenses.gaussian_noise(BASE, 0.05, seed=1)
    assert out.shape == BASE.shape
    assert not np.allclose(out, BASE)


def test_gaussian_is_reproducible_under_seed():
    a = defenses.gaussian_noise(BASE, 0.05, seed=7)
    b = defenses.gaussian_noise(BASE, 0.05, seed=7)
    assert np.allclose(a, b)
    c = defenses.gaussian_noise(BASE, 0.05, seed=8)
    assert not np.allclose(a, c)


def test_quantize_reduces_distinct_values_and_preserves_shape():
    # a column with many distinct values collapses to few levels
    col = np.linspace(0, 1, 50).reshape(-1, 1)
    out = defenses.quantize(col, levels=4)
    assert out.shape == col.shape
    assert len(np.unique(out)) <= 4
    assert len(np.unique(out)) < len(np.unique(col))


def test_quantize_large_levels_is_near_identity():
    out = defenses.quantize(BASE, levels=2 ** 16)
    assert np.allclose(out, BASE)
