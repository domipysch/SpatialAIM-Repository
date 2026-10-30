"""Tests for the small shared helpers in ``spatialaim.utils``."""

import numpy as np

from spatialaim.utils import box_stats


def test_box_stats_matches_plotly():
    # Plotly's raw-data box: Hazen quartiles, whiskers at the most extreme values
    # within 1.5 IQR of the box (here 100 lies beyond the upper fence).
    stats = box_stats(list(range(1, 11)) + [100])
    assert stats["n"] == 11
    assert np.isclose(stats["q1"], 3.25)
    assert np.isclose(stats["median"], 6.0)
    assert np.isclose(stats["q3"], 8.75)
    assert stats["lowerfence"] == 1.0
    assert stats["upperfence"] == 10.0
    assert stats["min"] == 1.0 and stats["max"] == 100.0
    assert np.isclose(stats["mean"], 155 / 11)


def test_box_stats_ignores_non_finite_values():
    assert box_stats([np.nan, 2.0, np.inf])["n"] == 1
    empty = box_stats([np.nan])
    assert empty["n"] == 0 and empty["median"] is None
