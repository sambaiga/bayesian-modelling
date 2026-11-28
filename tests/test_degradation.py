import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from bayes import degradation, survival


def synthetic(noise: float = 0.3, seed: int = 0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    rows = []
    for cell, chemistry, a, b in (
        ("A_1", "A", 3.9, 2.4),
        ("A_2", "A", 3.8, 2.2),
        ("B_1", "B", 3.2, 1.1),
        ("B_2", "B", 3.1, 1.0),
    ):
        cycles = np.arange(8, 1400, 8, dtype=float)
        loss = np.exp(a) * (cycles / 1000.0) ** b + generator.normal(0, noise, len(cycles))
        rows.append(pd.DataFrame({"cell": cell, "CellType": chemistry, "cycle": cycles, "soh": 100.0 - loss}))
    return pd.concat(rows, ignore_index=True)


def test_crossing_cycle_solves_the_path_for_the_loss():
    for a, b, loss in ((3.9, 2.4, 20.0), (3.2, 1.1, 20.0), (2.0, 0.8, 35.0)):
        root = brentq(lambda t, a, b, loss: np.exp(a) * (t / 1000.0) ** b - loss, 1e-3, 1e6, args=(a, b, loss))
        assert degradation.crossing_cycle(a, b, loss, unit=1000.0) == pytest.approx(root, rel=1e-9)


def test_crossing_is_later_for_a_smaller_loss_after_1000_cycles_and_vectorised():
    cycles = degradation.crossing_cycle(np.array([3.0, 3.5]), np.array([1.5, 1.5]))
    assert cycles[0] > cycles[1] > 0
    assert degradation.crossing_cycle(np.log(20.0), 1.0) == pytest.approx(degradation.CYCLE_UNIT)  # loss 20 at the unit
    assert degradation.crossing_cycle(np.log(10.0), 1.0) == pytest.approx(
        2 * degradation.CYCLE_UNIT
    )  # linear, half the rate


def test_invalid_arguments_to_the_crossing_are_refused():
    with pytest.raises(ValueError):
        degradation.crossing_cycle(3.0, 0.0)
    with pytest.raises(ValueError):
        degradation.crossing_cycle(3.0, 1.0, loss=0.0)


def test_path_model_has_a_cell_path_for_every_cell_and_the_loss_as_observed_data():
    model = degradation.path_model(synthetic())
    assert {"a_cell", "b_cell", "noise", "loss"} <= set(model.named_vars)
    assert [variable.name for variable in model.observed_RVs] == ["loss"]
    assert list(model.coords["cell"]) == ["A_1", "A_2", "B_1", "B_2"]
    assert list(model.coords["chemistry"]) == ["A", "B"]


def test_path_model_recovers_the_crossing_of_a_noisy_synthetic_cell(tmp_path):
    from bayes.regression import hierarchical as hier

    frame = synthetic()
    idata = hier.sample(degradation.path_model(frame), tmp_path / "path.nc", seed=1, draws=300, tune=400, chains=2)
    crossings = degradation.cell_crossings(idata)
    truth = {"A_1": 1000 * (20 / np.exp(3.9)) ** (1 / 2.4), "B_1": 1000 * (20 / np.exp(3.2)) ** (1 / 1.1)}
    for cell, value in truth.items():
        assert crossings[cell].median() == pytest.approx(value, rel=0.03)
    assert crossings.shape[1] == 4 and crossings.shape[0] == 600


def test_weibull_model_censors_the_records_that_have_not_crossed(tmp_path):
    table = pd.DataFrame(
        {
            "cell": ["a", "b", "c", "d"],
            "CellType": ["X", "X", "Y", "Y"],
            "time": [600.0, 700.0, 500.0, 500.0],
            "event": [True, True, False, False],
        }
    )
    model = survival.weibull_model(table)
    point = {"alpha": np.log(2.0), "mu": np.log(0.8), "delta": 0.0}
    from scipy import stats

    manual = (
        stats.weibull_min.logpdf(0.6, 2.0, scale=0.8)
        + stats.weibull_min.logpdf(0.7, 2.0, scale=0.8)
        + 2 * stats.weibull_min.logsf(0.5, 2.0, scale=0.8)
    )
    value = model.compile_logp(vars=[model["lifetime"]], sum=True)(
        {"alpha_log__": point["alpha"], "mu": point["mu"], "delta": point["delta"]}
    )
    assert float(value) == pytest.approx(float(manual), rel=1e-9)


def test_weibull_model_needs_exactly_two_chemistries():
    table = pd.DataFrame({"cell": ["a", "b"], "CellType": ["X", "X"], "time": [1.0, 2.0], "event": [True, True]})
    with pytest.raises(ValueError):
        survival.weibull_model(table)


def test_conditional_draws_stay_above_the_age_and_follow_the_conditional_median():
    generator = np.random.default_rng(0)
    alpha, beta, age = np.full(40000, 2.0), np.full(40000, 0.8), 0.6
    draws = survival.weibull_conditional_draws(alpha, beta, age, generator)
    assert draws.min() > age
    median = beta[0] * ((age / beta[0]) ** 2.0 + np.log(2.0)) ** 0.5  # S(t) = S(age) / 2
    assert np.median(draws) == pytest.approx(median, rel=0.01)
    with pytest.raises(ValueError):
        survival.weibull_conditional_draws(alpha, beta, -1.0, generator)


def test_interval_and_pinball_score_the_draws_against_the_outcome():
    draws = np.arange(1.0, 101.0)
    inside = survival.interval_and_pinball(draws, 50.0, level=0.9, tau=0.1)
    assert inside["covered"] == 1.0 and inside["interval"] == pytest.approx(inside["upper"] - inside["lower"])
    assert inside["error"] == pytest.approx(inside["median"] - 50.0)
    outside = survival.interval_and_pinball(draws, 130.0, level=0.9, tau=0.1)
    assert outside["covered"] == 0.0
    assert outside["interval"] == pytest.approx(outside["upper"] - outside["lower"] + 20.0 * (130.0 - outside["upper"]))
    assert outside["pinball"] == pytest.approx(0.1 * (130.0 - outside["rated"]))  # a promise below the outcome
    late = survival.interval_and_pinball(draws, 5.0, tau=0.1)
    assert late["pinball"] == pytest.approx(0.9 * (late["rated"] - 5.0))  # a promise above the outcome costs more
    with pytest.raises(ValueError):
        survival.interval_and_pinball(np.array([]), 1.0)
    with pytest.raises(ValueError):
        survival.interval_and_pinball(draws, 1.0, tau=1.0)


def test_path_bands_follow_the_true_path_of_a_synthetic_cell(tmp_path):
    from bayes.regression import hierarchical as hier

    idata = hier.sample(
        degradation.path_model(synthetic()), tmp_path / "bands.nc", seed=2, draws=200, tune=300, chains=2
    )
    cycles = np.array([300.0, 600.0, 900.0, 1200.0])
    bands = degradation.path_bands(idata, cycles, draws=100)
    assert set(bands["cell"]) == {"A_1", "A_2", "B_1", "B_2"} and len(bands) == 16
    truth = np.exp(3.9) * (cycles / 1000.0) ** 2.4
    mine = bands[bands["cell"] == "A_1"].sort_values("cycle")
    assert mine["median"].to_numpy() == pytest.approx(truth, rel=0.1)
    assert (mine["lower"] <= mine["median"]).all() and (mine["median"] <= mine["upper"]).all()
    assert np.all(np.diff(mine["median"]) > 0)  # the loss only grows
