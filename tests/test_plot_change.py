import pandas as pd

from bayes.plot import change
from bayes.plot.inline import inline_svg


def test_percent_uses_the_typographic_minus_and_a_plus_sign():
    assert change.percent(-52.4) == "−52%"
    assert change.percent(22.0) == "+22%"
    assert change.percent(0.2) == "0%"


def test_percent_takes_a_unit_for_percentage_points():
    assert change.percent(-3.4, " pts") == "\u22123 pts" and change.percent(0.1, " pts") == "0 pts"


def test_roles_colour_worse_bars_orange_and_the_improving_subject_blue():
    frame = pd.DataFrame({"name": ["a", "b", "c"], "value": [-40.0, -10.0, 30.0]})
    assert list(change.roles(frame, subject="a")) == ["subject", "other", "worse"]
    assert list(change.roles(frame.assign(value=[30.0, -10.0, -5.0]), subject="a")) == ["worse", "other", "other"]


def test_the_change_chart_names_the_reference_and_prints_every_value():
    frame = pd.DataFrame(
        {
            "panel": ["(a) One"] * 2 + ["(b) Two"] * 2,
            "name": ["a", "b"] * 2,
            "value": [-40.0, 30.0, -10.0, 120.0],
        }
    )
    figure = change.change(frame, reference="Complete pooling", order=["a", "b"], subject="a", ncol=2)
    svg = inline_svg(figure)
    assert "Complete pooling" in svg and "−40%" in svg and "+120%" in svg


def test_a_role_column_replaces_the_colouring_by_sign_and_the_unit_is_printed():
    frame = pd.DataFrame(
        {"panel": ["(a) One"] * 2, "name": ["a", "b"], "value": [10.0, -40.0], "role": ["other", "worse"]}
    )
    svg = inline_svg(change.change(frame, "Nominal", ["a", "b"], ncol=1, unit=" pts"))
    assert "+10 pts" in svg and "\u221240 pts" in svg


def test_show_builds_a_desktop_and_a_phone_variant_from_a_function(monkeypatch):
    from bayes.plot import inline

    shown = []
    monkeypatch.setattr(inline, "display", lambda html: shown.append(html.data))
    frame = pd.DataFrame({"panel": ["(a) A", "(b) B"], "name": ["x", "x"], "value": [-10.0, 20.0]})
    inline.show(lambda: change.change(frame, "Reference", ["x"], ncol=2))
    html = shown[0]
    assert 'class="letsplot letsplot-desktop"' in html and 'class="letsplot letsplot-phone"' in html
    wide, narrow = html.split('class="letsplot letsplot-phone"')
    assert 'viewBox="0 0 900' in wide and 'viewBox="0 0 420' in narrow
