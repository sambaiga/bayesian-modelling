import pandas as pd

from bayes.plot import hierarchy, pooling
from bayes.plot.inline import inline_svg

CELLS = {"A_1": "A", "A_2": "A", "A_3": "A", "B_1": "B", "B_2": "B"}
HELD = ["A_3", "B_2"]
COLOURS = hierarchy.palette(["A", "B"])


def texts(figure) -> set[str]:
    found = set()
    for layer in figure.as_dict()["layers"]:
        data = layer.get("data")
        if data is not None and "label" in data:
            found |= set(pd.Series(data["label"]).tolist())
    return found


def test_every_diagram_draws_the_cells_and_labels_the_held_out_cell():
    for diagram in (pooling.complete, pooling.none, pooling.partial):
        figure = diagram(CELLS, HELD, COLOURS)
        assert "Held-out cell" in texts(figure) and "<svg" in inline_svg(figure)


def test_only_partial_pooling_has_a_level_between_the_fleet_and_the_cells():
    assert any(text.startswith("A: ") for text in texts(pooling.partial(CELLS, HELD, COLOURS)))
    assert not any(text.startswith("A: ") for text in texts(pooling.complete(CELLS, HELD, COLOURS)))


def test_cells_are_placed_in_the_order_of_their_chemistries_with_a_gap_between_chemistries():
    cells = pooling._cells(CELLS, HELD)
    assert cells["cell"].tolist() == list(CELLS) and cells["x"].is_monotonic_increasing
    assert cells["x"].iloc[3] - cells["x"].iloc[2] > cells["x"].iloc[1] - cells["x"].iloc[0]
    assert cells["held"].tolist() == [False, False, True, False, True]
