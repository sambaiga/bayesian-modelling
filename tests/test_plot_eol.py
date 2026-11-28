import numpy as np
import pandas as pd
import pytest

from bayes import survival
from bayes.plot import eol, hierarchy
from bayes.plot.inline import inline_svg

COLOURS = hierarchy.palette(["A", "B"])


def table() -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "cell": ["A_1", "A_2", "B_1", "B_2"],
            "CellType": ["A", "A", "B", "B"],
            "first": 5,
            "last": [900, 950, 1900, 1800],
            "eol": [530.0, 600.0, 700.0, 1400.0],
        }
    )
    return survival.truncate(base, 650)


def labels_of(spec: dict) -> set[str]:
    return {text for layer in spec["layers"] for text in layer.get("data", {}).get("label", [])} | set(
        spec.get("data", {}).get("label", [])
    )


def test_stops_draw_every_cell_with_a_hollow_point_for_each_cell_still_running():
    spec = eol.stops(table(), 650, COLOURS).as_dict()
    assert "Test stops" in labels_of(spec)
    shapes = [layer for layer in spec["layers"] if layer["geom"] == "point"]
    assert len(shapes) == 3  # the hidden end of life, the crossings seen, the cells running
    assert "<svg" in inline_svg(eol.stops(table(), 650, COLOURS))
    limits = [scale for scale in spec["scales"] if scale.get("aesthetic") == "y"][0]["limits"]
    assert limits == ["B_2", "B_1", "A_2", "A_1"]  # chemistry by chemistry, the earliest end of life at the top


def test_prior_lifetimes_label_the_prior_and_the_observations():
    prior = survival.weibull_prior_lifetimes(4000, np.random.default_rng(0))
    spec = eol.prior_lifetimes(prior, np.array([529.0, 600.0, 700.0]), COLOURS["A"]).as_dict()
    assert labels_of(spec) == {"Prior", "Observed"}
    assert "<svg" in inline_svg(eol.prior_lifetimes(prior, np.array([529.0, 600.0]), COLOURS["A"]))


def test_survival_bands_name_each_chemistry_and_mark_the_crossings_seen():
    cycles = np.linspace(0, 2000, 41)
    rows = []
    for name, scale in (("A", 600.0), ("B", 1100.0)):
        median = np.exp(-((cycles / scale) ** 3))
        rows.append(
            pd.DataFrame(
                {
                    "chemistry": name,
                    "cycle": cycles,
                    "median": median,
                    "lower": median * 0.8,
                    "upper": np.minimum(median * 1.2, 1),
                }
            )
        )
    spec = eol.survival_bands(pd.concat(rows, ignore_index=True), table(), 650, COLOURS).as_dict()
    assert labels_of(spec) == {"A", "B"}
    assert "<svg" in inline_svg(eol.survival_bands(pd.concat(rows, ignore_index=True), table(), 650, COLOURS))


def test_paths_separate_what_was_seen_from_what_was_not():
    cycles = np.arange(8, 1500, 8, dtype=float)
    observed = pd.DataFrame({"cell": "A_1", "cycle": cycles, "loss": 20.0 * (cycles / 530.0) ** 2})
    curves = pd.DataFrame({"cell": "A_1", "cycle": cycles, "median": 20.0 * (cycles / 560.0) ** 2})
    one = table().query("cell == 'A_1'")
    spec = eol.paths(curves, observed, one, 650, COLOURS["A"]).as_dict()
    counts = sorted(
        len(layer["data"]["loss"]) for layer in spec["layers"] if "data" in layer and "loss" in layer["data"]
    )
    assert counts == sorted([int((cycles <= 650).sum()), int((cycles > 650).sum())])  # seen and unseen split the data
    assert "<svg" in inline_svg(eol.paths(curves, observed, one, 650, COLOURS["A"]))


def test_intervals_cut_a_long_interval_at_the_edge_and_keep_the_observed_end_of_life():
    scores = pd.DataFrame(
        {
            "cell": ["A_1", "B_2"],
            "CellType": ["A", "B"],
            "eol": [700.0, 1466.0],
            "median": [800.0, 2500.0],
            "lower": [750.0, 1200.0],
            "upper": [900.0, 5000.0],
        }
    )
    spec = eol.intervals(scores, 650, COLOURS, upto=2000.0).as_dict()
    assert max(spec["data"]["upper"]) == 2000.0 and max(spec["data"]["median"]) == 2000.0
    assert list(spec["data"]["eol"]) == [700.0, 1466.0]


def test_accuracy_draws_one_line_for_each_method_with_a_key_below_the_axis():
    summary = pd.DataFrame(
        {
            "stop": [500, 600, 500, 600],
            "method": ["Path", "Path", "Weibull", "Weibull"],
            "median_error": [170.0, 175.0, 940.0, 370.0],
        }
    )
    spec = eol.accuracy(
        summary, "median_error", "Median absolute error (cycles)", {"Path": COLOURS["A"], "Weibull": "#64748B"}
    ).as_dict()
    assert spec["theme"]["legend_position"] == "bottom"
    assert set(spec["data"]["method"]) == {"Path", "Weibull"}
    with_reference = eol.accuracy(
        summary, "median_error", "Error", {"Path": COLOURS["A"], "Weibull": "#64748B"}, reference=0.9
    ).as_dict()
    assert any(layer["geom"] == "hline" for layer in with_reference["layers"])


def test_prior_lifetimes_of_the_weibull_model_cover_the_range_of_a_battery_life():
    draws = survival.weibull_prior_lifetimes(20000, np.random.default_rng(1))
    low, median, high = np.percentile(draws, [5, 50, 95])
    assert 50 < low < 200 and 400 < median < 900 and 2000 < high < 4000  # plausible, and wide enough to learn from
    assert draws.min() > 0


def test_survival_bands_of_a_fitted_model_fall_with_the_cycle(tmp_path):
    from bayes.regression import hierarchical as hier

    records = table().assign(time=lambda t: t["time"], event=lambda t: t["event"])
    idata = hier.sample(survival.weibull_model(records), tmp_path / "w.nc", seed=1, draws=100, tune=100, chains=2)
    bands = survival.weibull_survival_bands(idata, np.array([0.0, 500.0, 1000.0, 3000.0]), draws=50)
    assert set(bands["chemistry"]) == {"A", "B"} and len(bands) == 8
    for _, group in bands.groupby("chemistry"):
        assert group["median"].iloc[0] == pytest.approx(1.0) and np.all(np.diff(group["median"]) <= 0)
        assert (group["lower"] <= group["median"]).all() and (group["median"] <= group["upper"]).all()
