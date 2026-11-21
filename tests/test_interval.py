import numpy as np
import pytest

from bayes.metrics.interval import interval_coverage, interval_score


def test_interval_score_is_the_width_when_every_observation_is_inside():
    lower, upper = np.array([0.0, 1.0, 2.0]), np.array([2.0, 4.0, 3.0])
    assert interval_score(np.array([1.0, 2.0, 2.5]), lower, upper, alpha=0.1) == pytest.approx(2.0)


def test_interval_score_adds_a_penalty_of_two_over_alpha_per_unit_outside():
    # Widths are 2 and 2. The second observation lies 3 above the interval: 2 + (2 / 0.1) * 3 = 62. The mean is 32.
    assert interval_score(np.array([1.0, 5.0]), np.array([0.0, 0.0]), np.array([2.0, 2.0]), alpha=0.1) == pytest.approx(
        32.0
    )


def test_interval_score_penalises_below_and_above_equally():
    below = interval_score(np.array([-1.0]), np.array([0.0]), np.array([2.0]), alpha=0.2)
    above = interval_score(np.array([3.0]), np.array([0.0]), np.array([2.0]), alpha=0.2)
    assert below == pytest.approx(above)


def test_interval_score_rejects_alpha_outside_the_unit_interval():
    with pytest.raises(ValueError):
        interval_score(np.array([1.0]), np.array([0.0]), np.array([2.0]), alpha=1.5)


def test_interval_score_rejects_arrays_of_different_length():
    with pytest.raises(ValueError):
        interval_score(np.array([1.0, 2.0]), np.array([0.0]), np.array([2.0]), alpha=0.1)


def test_interval_score_is_not_lowered_by_a_narrower_interval_that_misses_the_tail():
    generator = np.random.default_rng(0)
    observed = generator.standard_normal(20000)
    honest = interval_score(observed, np.full(20000, -1.645), np.full(20000, 1.645), alpha=0.1)
    too_narrow = interval_score(observed, np.full(20000, -0.5), np.full(20000, 0.5), alpha=0.1)
    assert honest < too_narrow


def test_interval_coverage_counts_the_share_inside():
    assert interval_coverage(np.array([1.0, 3.0]), np.array([0.0, 0.0]), np.array([2.0, 2.0])) == pytest.approx(0.5)
