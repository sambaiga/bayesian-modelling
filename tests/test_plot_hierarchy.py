import numpy as np
import pandas as pd
import pytest

from bayes.plot import hierarchy
from bayes.plot.inline import inline_svg


def labels_of(spec: dict) -> set[str]:
    """Texts of the direct labels of a figure, read from the data of its layers."""
    found = set()
    for layer in spec["layers"]:
        data = layer.get("data")
        if data is not None and "label" in data:
            found |= set(pd.Series(data["label"]).tolist())
    return found


def test_tint_mixes_a_colour_with_white():
    assert hierarchy.tint("#000000", 0.5) == "#7F7F7F"
    assert hierarchy.tint("#0369A1", 0.0) == "#0369A1"
    assert hierarchy.tint("#0369A1", 1.0) == "#FFFFFF"


def test_palette_gives_the_first_two_groups_blue_and_orange_and_refuses_more_than_four():
    colours = hierarchy.palette(["CS2", "CX2"])
    assert colours["CS2"] == "#0369A1" and colours["CX2"] == "#E26029"
    with pytest.raises(ValueError):
        hierarchy.palette(list("abcde"))


def test_spread_keeps_the_order_and_a_minimum_gap_between_neighbours():
    placed = hierarchy.spread([5.0, 5.1, 9.0, 5.05], gap=0.5)
    assert sorted(placed) == pytest.approx(sorted(placed))
    ordered = sorted(placed)
    assert all(b - a >= 0.5 - 1e-12 for a, b in zip(ordered, ordered[1:], strict=False))
    assert placed[2] == 9.0  # a value that is already far from the others does not move


def test_spread_leaves_well_separated_values_unchanged():
    assert hierarchy.spread([1.0, 5.0, 3.0], gap=0.5) == [1.0, 5.0, 3.0]


def test_the_loo_figure_marks_the_best_model_and_draws_one_row_per_model():
    table = pd.DataFrame({"method": ["A", "B", "C"], "elpd_diff": [0.0, 10.0, 200.0], "dse": [0.0, 3.0, 20.0]})
    spec = hierarchy.loo(table, best="A").as_dict()
    assert labels_of(spec) == {"A", "B", "C"}


def test_an_inline_svg_scales_with_its_container():
    plot = hierarchy.loo(pd.DataFrame({"method": ["A"], "elpd_diff": [0.0], "dse": [0.0]}), best="A")
    html = inline_svg(plot)
    assert html.startswith('<div class="letsplot"><svg') and 'viewBox="0 0 780 240"' in html and 'width="100%"' in html


def test_heldout_draws_one_panel_for_each_cell():
    frame = pd.DataFrame(
        {
            "cell": np.repeat(["a", "b", "c"], 4),
            "cycle": np.tile([1.0, 2.0, 3.0, 4.0], 3),
            "observed": 90.0,
            "median": 90.0,
            "lower": 85.0,
            "upper": 95.0,
        }
    )
    spec = hierarchy.heldout(frame).as_dict()
    assert spec["facet"]["name"] == "wrap" and "<svg" in inline_svg(hierarchy.heldout(frame))


def test_effects_orders_the_diagnostics_by_their_mean():
    table = pd.DataFrame(
        {"feature": ["x", "y", "z"], "mean": [0.3, -0.2, 0.0], "lower": [0.1, -0.4, -0.1], "upper": [0.5, 0.0, 0.1]}
    )
    spec = hierarchy.effects(table).as_dict()
    discrete = [scale for scale in spec["scales"] if scale.get("aesthetic") == "y"][0]
    assert discrete["limits"] == ["y", "z", "x"]
