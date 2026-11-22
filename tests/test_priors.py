import pandas as pd
import pymc as pm
import pytest

from bayes import lifetime, priors
from bayes.plot import tiles


def test_the_priors_tables_have_the_three_columns_and_no_empty_cell():
    for table in (priors.complete_pooling(), priors.hierarchical()):
        assert list(table.columns) == ["Parameter", "Prior", "Role"] and not table.isna().any().any()


def test_failure_rate_prior_table_gives_the_mean_life_and_the_weight():
    table = priors.failure_rate({"Engineer": (4.0, 40.0), "Weak": (1.0, 1.0)})
    assert table["Mean life (years)"].tolist() == ["10.0", "1.0"]
    assert table["Worth (machine-years)"].tolist() == ["40", "1"]


def test_the_complete_pooling_table_agrees_with_the_model_priors():
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from bayes.regression import hierarchical as hier

    frame = pd.DataFrame(
        {"cell": ["a"] * 3, "CellType": "X", "cycle": [1.0, 2.0, 3.0], "y": [0.8, 0.7, 0.6], "f": [0.1, 0.2, 0.3]}
    )
    model = hier.hierarchical_model(frame, StandardScaler().fit(frame[["f"]]), ["f"], "complete")
    names = {variable.name for variable in model.free_RVs}
    assert names == {"alpha", "loglambda", "amplitude", "beta", "phi"}
    assert len(priors.complete_pooling()) == len(names)


def test_decision_tiles_data_prints_the_median_life_with_its_interval():
    frame, labels = lifetime.decision_tiles_data(
        [
            {
                "prior": "a",
                "median": 9.2,
                "low": 4.7,
                "high": 21.8,
                "p1": 0.1,
                "p3": 0.3,
                "cost": 21.0,
                "probability": 0.48,
            }
        ]
    )
    assert labels[("median", "a")] == "9.2 (4.7 to 21.8)" and frame.loc["a", "cost"] == 21.0


def test_score_tiles_draw_a_panel_for_each_measure_and_outline_low_coverage():
    scores = pd.DataFrame(
        {
            "cell": ["a", "b", "a", "b"],
            "strategy": ["complete", "complete", "control", "control"],
            "mae": [1.0, 5.0, 4.0, 3.0],
            "coverage": [0.9, 0.0, 0.95, 0.9],
        }
    )
    figure = tiles.score_tiles(
        scores,
        {"mae": ("(a) Error", "{:.1f}"), "coverage": ("(b) Coverage", "{:.2f}")},
        {"complete": "With", "control": "Without"},
        flag_below=("coverage", 0.5),
    )
    layers = [layer["data"] for layer in figure.as_dict()["layers"] if layer.get("data") is not None]
    flagged = [data["flag"] for data in layers if "flag" in data]
    assert flagged and "flag" in set(flagged[0])
    from bayes.plot.inline import inline_svg

    svg = inline_svg(figure)
    assert "0.00" in svg and "5.0" in svg
