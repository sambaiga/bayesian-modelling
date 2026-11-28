import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special

from bayes import survival


def frame() -> pd.DataFrame:
    rows = []
    for cell, kind, start, drop in (
        ("A_1", "A", 0, 10.0),
        ("A_2", "A", 0, 20.0),
        ("B_1", "B", 5, 2.0),
        ("B_2", "B", 0, 0.0),
    ):
        cycles = np.arange(start, start + 100, 10)
        soh = 100.0 - drop * np.arange(len(cycles)) / 2.0
        rows.append(pd.DataFrame({"cell": cell, "CellType": kind, "cycle": cycles, "soh": soh}))
    return pd.concat(rows, ignore_index=True)


def test_end_of_life_is_the_first_cycle_below_the_threshold_and_nan_for_a_cell_that_never_crosses():
    table = survival.end_of_life(frame())
    eol = table.set_index("cell")["eol"]
    assert eol["A_1"] == 50 and eol["A_2"] == 30  # 100 - 5k < 80 first at k = 5, 100 - 10k < 80 at k = 3, cycles 10 k
    assert np.isnan(eol["B_1"]) and np.isnan(eol["B_2"])
    assert table.set_index("cell").loc["B_1", "first"] == 5 and table.set_index("cell").loc["A_1", "last"] == 90
    assert table["cell"].is_unique and list(table["CellType"]) == sorted(table["CellType"])


def test_a_threshold_equal_to_the_state_of_health_is_not_a_crossing():
    data = pd.DataFrame({"cell": "A", "CellType": "A", "cycle": [1, 2, 3], "soh": [100.0, 80.0, 79.9]})
    assert survival.end_of_life(data)["eol"].iloc[0] == 3  # strictly below, as in the scoring of the decision


def test_a_cell_that_starts_below_the_threshold_is_refused():
    data = pd.DataFrame({"cell": "A", "CellType": "A", "cycle": [1, 2], "soh": [79.0, 70.0]})
    with pytest.raises(ValueError, match="already below"):
        survival.end_of_life(data)


def test_truncate_turns_the_cells_that_have_not_crossed_into_censored_records():
    table = survival.end_of_life(frame())
    out = survival.truncate(table, 40).set_index("cell")
    assert bool(out.loc["A_2", "event"]) and out.loc["A_2", "time"] == 30  # crossed before the test stopped
    assert not bool(out.loc["A_1", "event"]) and out.loc["A_1", "time"] == 40  # would cross at 50, still running
    assert out.loc["B_1", "time"] == 40 and not bool(out.loc["B_1", "event"])
    # a record shorter than the test is watched until it ends
    short = survival.truncate(table, 500).set_index("cell")
    assert short.loc["B_2", "time"] == 90 and not bool(short.loc["B_2", "event"])
    assert short.loc["A_1", "event"] and short.loc["A_1", "time"] == 50


def test_truncating_late_enough_censors_no_cell_that_crosses():
    out = survival.truncate(survival.end_of_life(frame()), 1e9)
    assert out["event"].sum() == 2  # only A_1 and A_2 ever cross
    assert sorted(out.loc[out["event"], "time"]) == [30.0, 50.0]
    assert out.loc[~out["event"], "time"].eq(out.loc[~out["event"], "last"]).all()  # watched until the record ends


def test_survival_is_one_at_age_zero_exp_minus_one_at_the_scale_and_falls():
    assert survival.weibull_survival(2.0, 500.0, 0.0) == 1.0
    assert survival.weibull_survival(2.0, 500.0, 500.0) == pytest.approx(np.exp(-1.0))
    ages = np.linspace(0, 2000, 50)
    assert np.all(np.diff(survival.weibull_survival(1.7, 800.0, ages)) < 0)


def test_remaining_life_matches_numerical_integration_of_the_survival_function():
    for alpha, beta, age in ((2.0, 500.0, 300.0), (0.8, 300.0, 100.0), (3.5, 900.0, 700.0), (1.0, 400.0, 250.0)):
        tail = integrate.quad(lambda u, a, b: np.exp(-((u / b) ** a)), age, np.inf, args=(alpha, beta))[0]
        expected = tail / np.exp(-((age / beta) ** alpha))
        assert survival.weibull_remaining_life(alpha, beta, age) == pytest.approx(expected, rel=1e-8)


def test_remaining_life_is_the_mean_life_at_age_zero_and_the_scale_for_a_constant_hazard():
    assert survival.weibull_remaining_life(2.0, 500.0, 0.0) == pytest.approx(500.0 * special.gamma(1.5))
    assert survival.weibull_remaining_life(1.0, 400.0, np.array([0.0, 100.0, 900.0])) == pytest.approx(
        400.0
    )  # memoryless


def test_the_known_wrong_formula_that_drops_the_gamma_factor_is_far_from_the_right_answer():
    right = survival.weibull_remaining_life(2.0, 500.0, 300.0)
    wrong = 500.0 * np.exp((300.0 / 500.0) ** 2) - 300.0  # omits Gamma(1 + 1 / alpha)
    assert right == pytest.approx(251.6, abs=0.1) and wrong > 1.6 * right


def test_remaining_life_stays_finite_far_in_the_tail():
    value = survival.weibull_remaining_life(2.0, 100.0, 2000.0)
    assert np.isfinite(value) and value > 0.0
    assert value == pytest.approx(
        100.0**2 / (2 * 2000.0), rel=0.01
    )  # the mean excess of a Gaussian-like tail, beta^2 / (alpha age)


def test_invalid_arguments_are_refused():
    with pytest.raises(ValueError):
        survival.weibull_survival(0.0, 500.0, 10.0)
    with pytest.raises(ValueError):
        survival.weibull_survival(2.0, 500.0, -1.0)
    with pytest.raises(ValueError):
        survival.weibull_remaining_life(2.0, -1.0, 10.0)
    with pytest.raises(ValueError):
        survival.weibull_remaining_life(2.0, 500.0, -5.0)
