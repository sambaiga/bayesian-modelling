import pandas as pd
import pytest

from bayes.plot import tables, tiles
from bayes.plot.inline import inline_svg


@pytest.fixture
def wide() -> pd.DataFrame:
    return pd.DataFrame({"mae": [1.0, 3.0], "coverage": [0.9, 0.2]}, index=["a", "b"])


def test_long_frame_scales_the_shade_within_a_panel_and_prints_the_formatted_value(wide):
    long = tiles.long_frame(wide, {"Error": {"mae": "Error"}}, {"Error": "{:.1f}"}, ["a", "b"])
    assert long["shade"].tolist() == [0.0, 1.0]
    assert long["label"].tolist() == ["1.0", "3.0"]
    assert long["ink"].tolist() == ["dark", "white"]


def test_long_frame_flags_the_requested_tiles_and_prefers_a_given_label(wide):
    long = tiles.long_frame(
        wide,
        {"Coverage": {"coverage": "Coverage"}},
        {"Coverage": "{:.2f}"},
        ["a", "b"],
        flags={"Coverage": {("coverage", "b")}},
        labels={("coverage", "a"): "90%"},
    )
    assert long["flag"].tolist() == ["plain", "flag"] and long["label"].tolist() == ["90%", "0.20"]


def test_long_frame_gives_a_constant_panel_the_middle_shade():
    frame = pd.DataFrame({"x": [2.0, 2.0]}, index=["a", "b"])
    long = tiles.long_frame(frame, {"P": {"x": "x"}}, {"P": "{:.0f}"}, ["a", "b"])
    assert long["shade"].tolist() == [0.5, 0.5]


def test_long_frame_raises_for_an_unknown_row(wide):
    with pytest.raises(KeyError):
        tiles.long_frame(wide, {"P": {"mae": "m"}}, {"P": "{}"}, ["a", "zzz"])


def test_tiles_print_every_value(wide):
    long = tiles.long_frame(
        wide,
        {"Error": {"mae": "Error"}, "Coverage": {"coverage": "Coverage"}},
        {"Error": "{:.1f}", "Coverage": "{:.2f}"},
        ["a", "b"],
    )
    svg = inline_svg(tiles.tiles(long, ["a", "b"], ncol=2))
    assert "3.0" in svg and "0.20" in svg


def test_themed_gt_returns_a_gt_with_the_book_header_colour_and_stripes():
    table = tables.themed_gt(pd.DataFrame({"prior": ["a", "b", "c"], "mean": [1, 2, 3]}))
    html = table.as_raw_html()
    assert "#0369A1" in html and "#EAF3FA" in html
