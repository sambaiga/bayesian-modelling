"""A table of metrics drawn as a figure: tiles with the value printed, shaded within each panel.

The reader sees the pattern first and reads the exact value from the tile. Shade is single-hue, from white to the blue
of the site, scaled within a panel, so the darkest tile of a panel is its largest value. A tile can be outlined in
orange to mark the values that the text discusses, such as a coverage far below its nominal level.
"""

from __future__ import annotations

from lets_plot import (
    aes,
    element_text,
    facet_wrap,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_gradient,
    scale_y_discrete,
    theme,
)
import numpy as np
import pandas as pd

from .hierarchy import tint
from .theme import BLUE, INK, ORANGE, canvas, columns, site_theme


def long_frame(
    frame: pd.DataFrame,
    panels: dict[str, dict[str, str]],
    formats: dict[str, str],
    rows: list[str],
    flags: dict[str, set[tuple[str, str]]] | None = None,
    labels: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Reshape a wide table into one row per tile, with its shade, printed label and flag.

    Args:
        frame: Table with the row names as its index and one column for each value.
        panels: For each panel title, the columns that it shows, with the header of each column.
        formats: For each panel title, the format of the printed value, such as ``"{:.1f}"``.
        rows: Row names from the top row to the bottom row.
        flags: For each panel title, the (column, row) pairs to outline in orange.
        labels: Text to print instead of the formatted value, by (column, row).

    Returns:
        Rows with ``panel``, ``row``, ``header``, ``value``, ``shade``, ``label``, ``flag`` and ``ink``.

    Raises:
        KeyError: If a row or a column is not in ``frame``.
    """
    flags = flags or {}
    labels = labels or {}
    parts = []
    for panel, shown in panels.items():
        values = np.array([[frame.loc[row, column] for column in shown] for row in rows], dtype=float)
        low, high = float(values.min()), float(values.max())
        for position, (column, header) in enumerate(shown.items()):
            for index, row in enumerate(rows):
                value = float(values[index, position])
                shade = 0.5 if high == low else (value - low) / (high - low)
                parts.append(
                    {
                        "panel": panel,
                        "row": row,
                        "header": header,
                        "value": value,
                        "shade": shade,
                        "label": labels.get((column, row), formats[panel].format(value)),
                        "flag": "flag" if (column, row) in flags.get(panel, set()) else "plain",
                        "ink": "white" if shade > 0.55 else "dark",
                    }
                )
    return pd.DataFrame(parts)


def tiles(  # noqa: ANN201
    long: pd.DataFrame, rows: list[str], ncol: int = 3, width: int = 900, height: int | None = None, phone_ncol: int = 1
):
    """Draw the tiles of a long frame, one panel for each group of columns.

    Args:
        long: The result of :func:`long_frame`.
        rows: Row names from the top row to the bottom row.
        ncol: Panels in a row.
        width: Width of the figure in pixels.
        height: Height in pixels; by default 70 plus 38 for each row of tiles and panel.
        phone_ncol: Panels in a row on a phone.

    Returns:
        The figure.
    """
    panels = long["panel"].nunique()
    height = height or int(70 + 38 * len(rows) * np.ceil(panels / ncol))
    flagged = long[long["flag"] == "flag"]
    return (
        ggplot(long, aes("header", "row"))
        + geom_tile(aes(fill="shade"), color="white", size=1.2)
        + scale_fill_gradient(low=tint(BLUE, 0.94), high=BLUE, limits=[0, 1])
        + geom_tile(data=flagged, fill="rgba(0,0,0,0)", color=ORANGE, size=1.6)
        + geom_text(aes(label="label", color="ink"), size=5.6)
        + scale_color_manual(values={"white": "white", "dark": INK})
        + facet_wrap("panel", ncol=columns(ncol, phone_ncol), scales="free_x")
        + scale_y_discrete(limits=rows[::-1])
        + labs(x="", y="")
        + site_theme()
        + theme(
            axis_line="blank",
            axis_ticks="blank",
            axis_text_x=element_text(size=10.5),
            axis_text_y=element_text(size=11),
        )
        + canvas(width, height, panels, ncol, phone_ncol)
    )


def score_tiles(
    scores: pd.DataFrame,
    measures: dict[str, tuple[str, str]],
    strategies: dict[str, str],
    key: str = "cell",
    flag_below: tuple[str, float] | None = None,
):
    """Tiles of the scores of each row key under each strategy, one panel for each measure.

    Args:
        scores: Rows with ``strategy``, the ``key`` column and the measures, such as the result of ``leave_one_out``.
        measures: For each measure column, the title of its panel and the format of its values.
        strategies: Header of each strategy in a panel, by strategy name, in the order of the columns.
        key: Column that identifies a row of tiles, such as the held-out cell.
        flag_below: A measure column and a threshold; tiles of that measure below the threshold are outlined.

    Returns:
        The figure.
    """
    rows = sorted(scores[key].unique())
    wide = scores.pivot(index=key, columns="strategy")
    panels = {
        title: {f"{measure}|{strategy}": header for strategy, header in strategies.items()}
        for measure, (title, _) in measures.items()
    }
    frame = pd.DataFrame(
        {f"{measure}|{strategy}": wide[measure][strategy] for measure in measures for strategy in strategies}
    )
    formats = {title: fmt for _, (title, fmt) in measures.items()}
    flags: dict[str, set[tuple[str, str]]] = {}
    if flag_below is not None:
        measure, threshold = flag_below
        title = measures[measure][0]
        flags[title] = {
            (f"{measure}|{strategy}", row)
            for strategy in strategies
            for row in rows
            if frame.loc[row, f"{measure}|{strategy}"] < threshold
        }
    return tiles(long_frame(frame, panels, formats, rows, flags), rows, ncol=len(measures))
