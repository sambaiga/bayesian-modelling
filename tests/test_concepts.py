import numpy as np
import pytest

from bayes.plot import concepts, hierarchy, icons
from bayes.plot.inline import diagram_svg

CELLS = {"A_1": "A", "A_2": "A", "A_3": "A", "B_1": "B", "B_2": "B"}
HELD = ["A_3", "B_2"]
COLOURS = hierarchy.palette(["A", "B"])


def texts(figure) -> list[str]:
    return [text.get_text() for text in figure.axes[0].texts]


def test_an_icon_is_one_glyph_and_an_unknown_name_raises():
    assert len(icons.icon("battery-half")) == 1
    with pytest.raises(KeyError):
        icons.icon("no-such-icon")


def test_every_icon_that_the_diagrams_use_exists_in_the_font():
    figures = [
        concepts.complete(CELLS, HELD, COLOURS),
        concepts.none(CELLS, HELD, COLOURS),
        concepts.partial(CELLS, HELD, COLOURS),
        concepts.decision(),
    ]
    glyphs = {text for figure in figures for text in texts(figure) if len(text) == 1}
    assert len(glyphs) >= 8  # the batteries, the curves, the group, the layers, the signpost and the rest


def test_each_pooling_diagram_labels_the_held_out_cell_and_says_what_a_new_cell_receives():
    assert "Held out" in texts(concepts.complete(CELLS, HELD, COLOURS))
    assert any("same curve" in text for text in texts(concepts.complete(CELLS, HELD, COLOURS)))
    assert any("only the prior" in text for text in texts(concepts.none(CELLS, HELD, COLOURS)))
    assert any("starts there" in text for text in texts(concepts.partial(CELLS, HELD, COLOURS)))


def test_only_partial_pooling_has_a_level_for_each_chemistry():
    assert any(text.startswith("A: ") for text in texts(concepts.partial(CELLS, HELD, COLOURS)))
    assert not any(text.startswith("A: ") for text in texts(concepts.complete(CELLS, HELD, COLOURS)))


def test_cells_are_placed_by_chemistry_with_the_held_out_cell_last_and_a_gap_between_chemistries():
    cells = concepts._layout(CELLS, HELD)
    assert [cell["cell"] for cell in cells] == ["A_1", "A_2", "A_3", "B_1", "B_2"]
    assert [cell["held"] for cell in cells] == [False, False, True, False, True]
    positions = np.array([cell["x"] for cell in cells])
    assert np.all(np.diff(positions) > 0) and positions[3] - positions[2] > positions[1] - positions[0]
    assert positions.min() > 0 and positions.max() < concepts.WIDTH


def test_the_decision_names_the_rule_the_two_costs_and_both_outcomes():
    found = " ".join(texts(concepts.decision()))
    for part in ("Inspect", "probability p", "Retire", "pay C", "Leave", "risk L", "yes", "no", "C / L"):
        assert part in found
    assert any("alpha" in text for text in texts(concepts.decision()))


def test_a_diagram_is_inline_svg_that_scales_and_needs_no_font():
    html = diagram_svg(concepts.decision())
    assert html.startswith('<div class="concept"') and 'viewBox="0 0 ' in html and 'width="100%"' in html
    assert "<text" not in html  # text and icons are outlines, so the page needs neither the icon font nor the typeface
