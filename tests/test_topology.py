"""Tests for the spatial assortativity in ``spatialaim.analysis.topology``."""

import numpy as np
import pytest
import scipy.sparse as sp

from spatialaim.analysis.topology import assortativity


def _grid(n: int = 10) -> sp.coo_matrix:
    """The 4-neighbour graph of an n×n grid, each edge stored in both directions
    (spot index = row * n + col)."""
    rows, cols = [], []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    rows.append(i)
                    cols.append(rr * n + cc)
    return sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n * n, n * n))


def _labels(fn, n: int = 10) -> np.ndarray:
    return np.array([fn(r, c) for r in range(n) for c in range(n)], dtype=int)


def test_halves_and_quadrants():
    # 180 edges; halves cut 10 of them, quadrants 20.
    halves = assortativity(_labels(lambda r, c: int(c >= 5)), _grid())
    assert halves.value == pytest.approx((170 / 180 - 0.5) / 0.5)
    quadrants = assortativity(_labels(lambda r, c: 2 * (r >= 5) + (c >= 5)), _grid())
    assert quadrants.value == pytest.approx((160 / 180 - 0.25) / 0.75)


def test_checkerboard_is_minus_one():
    result = assortativity(_labels(lambda r, c: (r + c) % 2), _grid())
    assert result.value == pytest.approx(-1.0)
    assert np.allclose(result.per_state, -1.0)


def test_global_is_weighted_mean_of_states():
    # A 4×4 corner block in a 10×10 grid: 56 of its 360 edge ends, 48 of
    # them on edges that stay inside.
    labels = _labels(lambda r, c: int(r < 4 and c < 4))
    result = assortativity(labels, _grid())
    b = np.array([304, 56]) / 360  # a = b on a symmetric graph
    assert result.per_state[1] == pytest.approx((48 / 56 - b[1]) / (1 - b[1]))
    weights = b * (1 - b)
    assert result.value == pytest.approx(
        np.sum(weights * result.per_state) / weights.sum()
    )
    assert result.mean_state == pytest.approx(result.per_state.mean())


def test_single_state_is_undefined():
    result = assortativity(np.zeros(100, dtype=int), _grid())
    assert np.isnan(result.value)
    assert np.isnan(result.mean_state)


def test_random_labels_near_zero():
    rng = np.random.default_rng(0)
    result = assortativity(rng.integers(0, 3, size=10_000), _grid(100))
    assert abs(result.value) < 0.02
    assert np.all(np.abs(result.per_state) < 0.03)
