import numpy as np
import pandas as pd
import pytest

from bayes import decision

OUTCOME = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # an event rate of 0.2


def test_retiring_only_above_the_ratio_gives_the_hand_computed_expense():
    probability = np.array([0.9, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    # ratio 0.25: the two cells with 0.9 are retired (0.25 each); the others are left and none is an event: 0.5 / 10
    assert decision.expense(probability, OUTCOME, 0.25) == pytest.approx(0.05)


def test_a_missed_event_costs_the_whole_loss():
    probability = np.zeros(10)
    assert decision.expense(probability, OUTCOME, 0.25) == pytest.approx(0.2)  # two events of loss 1, over 10 cells


def test_a_perfect_forecast_has_a_value_of_one_for_every_ratio():
    for ratio in (0.05, 0.3, 0.7):
        assert decision.value(OUTCOME.astype(float), OUTCOME, ratio) == pytest.approx(1.0)


def test_the_long_run_rate_has_a_value_of_zero():
    constant = np.full(10, 0.2)
    for ratio in (0.1, 0.5):
        assert decision.value(constant, OUTCOME, ratio) == pytest.approx(0.0)


def test_a_forecast_worse_than_the_rate_has_a_negative_value():
    inverted = 1.0 - OUTCOME.astype(float)
    assert decision.value(inverted, OUTCOME, 0.3) < 0.0


def test_the_value_is_undefined_without_an_event_or_for_a_free_or_certain_loss():
    assert np.isnan(decision.value(np.full(5, 0.5), np.zeros(5), 0.3))
    assert np.isnan(decision.value(np.full(5, 0.5), np.ones(5), 0.3))
    assert np.isnan(decision.value(OUTCOME.astype(float), OUTCOME, 0.0))
    assert np.isnan(decision.value(OUTCOME.astype(float), OUTCOME, 1.0))


def test_the_value_is_defined_at_the_event_rate():
    assert decision.value(OUTCOME.astype(float), OUTCOME, 0.2) == pytest.approx(1.0)


def test_a_probability_equal_to_the_ratio_does_not_retire():
    assert decision.expense(np.array([0.3]), np.array([1]), 0.3) == pytest.approx(1.0)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        decision.expense(np.array([0.5, 0.5]), np.array([1]), 0.3)
    with pytest.raises(ValueError):
        decision.expense(np.array([1.2]), np.array([1]), 0.3)
    with pytest.raises(ValueError):
        decision.value(np.array([]), np.array([]), 0.3)


def test_the_value_curve_has_one_row_per_ratio():
    curve = decision.value_curve(OUTCOME.astype(float), OUTCOME, np.array([0.1, 0.3, 0.6]))
    assert list(curve.columns) == ["ratio", "expense", "value"] and len(curve) == 3


def test_value_frame_scores_every_strategy_and_number_of_cycles_seen():
    rows = []
    for strategy in ("a", "b"):
        for seen in (0, 100):
            rows.append(
                pd.DataFrame(
                    {
                        "strategy": strategy,
                        "cycles_seen": seen,
                        "soh": [70, 90, 95, 60, 99],
                        "p_below": [0.9, 0.1, 0.2, 0.8, 0.0],
                    }
                )
            )
    frame = decision.value_frame(pd.concat(rows), np.array([0.3, 0.5]), 80.0)
    assert len(frame) == 2 * 2 * 2 and set(frame["strategy"]) == {"a", "b"}
    assert frame["value"].dropna().between(0.99, 1.0).all()  # the probabilities separate the events perfectly
