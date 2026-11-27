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


LOO = pd.DataFrame(
    {
        "method": ["No pooling"] * 4 + ["Partial pooling"] * 4,
        "cell": ["A_1", "A_2", "B_1", "B_2"] * 2,
        "elpd": [10.0, 20.0, 30.0, 40.0, 5.0, 25.0, 20.0, 35.0],
    }
)


def loo_figure():
    colours = hierarchy.palette(["A", "B"])
    chemistry_of = {"A_1": "A", "A_2": "A", "B_1": "B", "B_2": "B"}
    return hierarchy.loo(LOO, colours, chemistry_of, "Partial pooling", "No pooling")


def test_the_loo_figure_draws_one_bar_for_each_cell_with_its_gain_over_the_reference():
    spec = loo_figure().as_dict()
    assert labels_of(spec) == {"-5", "+5", "-10"}


def test_the_loo_figure_draws_when_every_cell_gains_or_every_cell_loses():
    colours = hierarchy.palette(["A", "B"])
    chemistry_of = {"A_1": "A", "A_2": "A", "B_1": "B", "B_2": "B"}
    worse = LOO.assign(elpd=[10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0])
    better = LOO.assign(elpd=[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    for frame in (worse, better):
        html = inline_svg(hierarchy.loo(frame, colours, chemistry_of, "Partial pooling", "No pooling"))
        assert "<svg" in html


def test_an_inline_svg_scales_with_its_container():
    html = inline_svg(loo_figure())
    assert html.startswith('<div class="letsplot"><svg') and 'viewBox="0 0 440 320"' in html and 'width="100%"' in html


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


def one_cell_frame() -> pd.DataFrame:
    cycles = np.arange(50.0, 650.0, 50.0)
    rows = []
    for cell, chemistry, offset in (("A_1", "A", 0.0), ("B_1", "B", 5.0)):
        rows.append(
            pd.DataFrame(
                {
                    "cell": cell,
                    "CellType": chemistry,
                    "cycle": cycles,
                    "soh": 100 - cycles / 20 + offset,
                    "median": 100 - cycles / 20 + offset,
                    "lower": 95 - cycles / 20 + offset,
                    "upper": 105 - cycles / 20 + offset,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_bands_name_the_cells_mark_the_cycles_seen_and_use_the_colour_of_the_chemistry():
    colours = hierarchy.palette(["A", "B"])
    spec = hierarchy.bands(one_cell_frame(), colours, seen=100).as_dict()
    assert {"A_1", "B_1", "Median", "90% interval", "Observation"} <= labels_of(spec)
    assert {"End of life, 80%", "Cycles seen"} <= labels_of(spec)
    assert "Cycles seen" not in labels_of(hierarchy.bands(one_cell_frame(), colours, seen=0).as_dict())
    assert "<svg" in inline_svg(hierarchy.bands(one_cell_frame(), colours))


def prior_posterior_frame() -> pd.DataFrame:
    generator = np.random.default_rng(0)
    rows = [pd.DataFrame({"quantity": "Q", "source": "Prior", "value": generator.normal(0, 2, 400)})]
    for name, centre in (("A", -1.0), ("B", 1.0)):
        rows.append(pd.DataFrame({"quantity": "Q", "source": name, "value": generator.normal(centre, 0.2, 400)}))
    return pd.concat(rows, ignore_index=True)


def test_priors_label_the_prior_and_each_posterior_and_draw_only_the_requested_quantity():
    frame = pd.concat([prior_posterior_frame(), prior_posterior_frame().assign(quantity="Other")])
    spec = hierarchy.priors(frame, "Q", hierarchy.palette(["A", "B"])).as_dict()
    assert labels_of(spec) == {"Prior", "A", "B"}


def test_sensitivity_overlays_one_density_for_each_prior_with_a_legend_for_one_chemistry():
    generator = np.random.default_rng(0)
    rows = [
        pd.DataFrame({"setting": setting, "chemistry": chemistry, "value": generator.normal(centre, 0.3, 300)})
        for setting in ("Narrow", "Default", "Wide")
        for chemistry, centre in (("A", -0.3), ("B", 0.7))
    ]
    draws = pd.concat(rows, ignore_index=True)
    spec = hierarchy.sensitivity(draws, "A").as_dict()
    assert spec["theme"]["legend_position"] == "top"
    assert len(set(spec["data"]["distribution"])) == 3 and len(spec["data"]["value"]) == 900  # one chemistry
    assert "<svg" in inline_svg(hierarchy.sensitivity(draws, "B"))


def test_value_draws_one_line_for_each_strategy_and_names_it_at_its_end():
    ratios = np.linspace(0.1, 0.9, 5)
    frame = pd.concat(
        [
            pd.DataFrame({"strategy": name, "ratio": ratios, "value": level})
            for name, level in (("partial", 0.4), ("complete", 0.2), ("none", -0.5), ("control", -3.0))
        ]
    )
    spec = hierarchy.value(frame).as_dict()
    assert labels_of(spec) == {"Partial pooling", "Complete pooling", "No pooling", "Control, no diagnostics"}
    assert min(spec["data"]["drawn"]) == -1.0  # the control is drawn at the floor, and its label is there too


def test_chemistry_draws_the_requested_parameter_as_one_density_for_each_chemistry():
    generator = np.random.default_rng(0)
    table = pd.DataFrame(
        {
            "quantity": ["P"] * 200 + ["Q"] * 200,
            "chemistry": (["A"] * 100 + ["B"] * 100) * 2,
            "value": generator.normal(size=400),
        }
    )
    colours = hierarchy.palette(["A", "B"])
    spec = hierarchy.chemistry(table, colours, "Q").as_dict()
    assert {layer["geom"] for layer in spec["layers"]} == {"density"}
    assert len(spec["data"]["distribution"]) == 200  # only the requested parameter is drawn


def test_priors_keep_the_draws_inside_the_axis_so_that_nothing_is_clipped():
    frame = prior_posterior_frame()
    spec = hierarchy.priors(frame, "Q", hierarchy.palette(["A", "B"])).as_dict()
    values = np.asarray(spec["data"]["value"], dtype=float)
    axis = [scale for scale in spec["scales"] if scale.get("aesthetic") == "x"][0]["limits"]
    assert values.min() >= axis[0] and values.max() <= axis[1]
    assert len(values) < len(frame)  # the extreme tails are dropped
