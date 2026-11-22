import numpy as np
import pandas as pd

from bayes.plot import lifetime
from bayes.plot.inline import inline_svg


def labels_of(figure) -> set[str]:
    """Texts of the direct labels of a figure, read from the data of its layers."""
    found = set()
    for layer in figure.as_dict()["layers"]:
        data = layer.get("data")
        if data is not None and "label" in data:
            found |= set(pd.Series(data["label"]).tolist())
    return found


def test_rates_labels_the_three_curves():
    grid = np.linspace(0.01, 0.6, 50)
    figure = lifetime.rates(grid, np.exp(-grid), np.exp(-2 * grid), np.exp(-3 * grid))
    assert {"Prior", "Likelihood", "Posterior"} <= labels_of(figure) and "<svg" in inline_svg(figure)


def test_prior_check_marks_the_observed_count():
    figure = lifetime.prior_check(np.array([0.6, 0.3, 0.08, 0.02]), observed=1)
    assert "Observed: 1" in labels_of(figure) and "<svg" in inline_svg(figure)


def test_traces_draws_one_line_for_each_chain():
    figure = lifetime.traces(np.random.default_rng(0).random((3, 20)))
    assert "<svg" in inline_svg(figure)


def test_survival_curves_labels_each_prior_at_the_end_of_its_curve():
    years = np.linspace(0, 10, 11)
    frame = pd.DataFrame(
        {
            "years": np.tile(years, 2),
            "survival": np.concatenate([np.exp(-0.1 * years), np.exp(-0.3 * years)]),
            "prior": np.repeat(["Engineer", "Sceptic"], len(years)),
        }
    )
    assert {"Engineer", "Sceptic"} <= labels_of(lifetime.survival_curves(frame, horizon=3.0, subject="Engineer"))


def test_sensitivity_has_two_panels_and_one_row_for_each_prior():
    table = pd.DataFrame({"prior": ["a", "b", "c"], "cost": [10.0, 20.0, 40.0], "probability": [0.1, 0.4, 0.9]})
    figure = lifetime.sensitivity(table, price=20.0)
    assert "<svg" in inline_svg(figure)
