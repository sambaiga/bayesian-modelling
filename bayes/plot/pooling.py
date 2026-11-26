"""Diagrams of the three levels of pooling: how the cells of a fleet are tied to one another.

Each diagram draws the 13 cells as dots along the bottom, grouped by chemistry, with the two held-out cells hollow. What
differs is what lies above them: one shared curve, none, or a chemistry level between the fleet and the cells. The
functions take the same inputs and return a Lets-Plot figure; none reads a model or a file.
"""

from __future__ import annotations

from lets_plot import (
    aes,
    element_blank,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
import pandas as pd

from .hierarchy import label
from .theme import INK, SLATE, canvas, site_theme

LEVELS = {"cell": 0.0, "chemistry": 1.5, "fleet": 3.0}


def _cells(chemistry_of: dict[str, str], held_out: list[str]) -> pd.DataFrame:
    """One row per cell, in the order of the chemistries, with its position and whether it is held out."""
    chemistries = list(dict.fromkeys(chemistry_of.values()))
    rows: list[dict[str, str | bool]] = []
    for name in chemistries:
        members = [cell for cell, chemistry in chemistry_of.items() if chemistry == name]
        rows.extend({"cell": cell, "chemistry": name, "held": cell in held_out} for cell in members)
    frame = pd.DataFrame(rows)
    gap = (frame["chemistry"] != frame["chemistry"].shift()).cumsum() - 1  # an empty place between the chemistries
    frame["x"] = range(1, len(frame) + 1)
    frame["x"] = frame["x"] + gap * 0.8
    frame["y"] = LEVELS["cell"]
    return frame


def _frame(cells: pd.DataFrame, title: str, drawn):  # noqa: ANN001, ANN202
    """The common frame of the three diagrams: the cells, the held-out cells as rings, and a title in the figure."""
    held = cells[cells["held"]]
    kept = cells[~cells["held"]]
    reach = float(cells["x"].max())
    ring = held.iloc[[0]].assign(y=-0.3, label="Held-out cell")
    return (
        ggplot()
        + drawn
        + geom_point(aes("x", "y", color="chemistry"), data=kept, shape=16, size=5.2)
        + geom_point(aes("x", "y", color="chemistry"), data=held, shape=21, fill="white", size=5.2, stroke=1.6)
        + label(pd.DataFrame({"x": [0.4], "y": [-0.7], "label": [title]}), SLATE, size=5.6)
        + label(ring, INK, hjust=0.5, size=5.0)
        + scale_x_continuous(limits=[0.2, reach + 0.8])
        + scale_y_continuous(limits=[-0.95, 3.35])
        + labs(x="", y="")
        + site_theme()
        + theme(
            axis_line=element_blank(), axis_text=element_blank(), axis_ticks=element_blank(), axis_title=element_blank()
        )
        + canvas(780, 235)
    )


def complete(chemistry_of: dict[str, str], held_out: list[str], colours: dict[str, str]):  # noqa: ANN201
    """Complete pooling: one curve for the whole fleet, so every cell, and any new cell, receives it.

    Args:
        chemistry_of: Chemistry of each cell, in the order to be drawn.
        held_out: Cells kept back for testing, drawn as rings.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    cells = _cells(chemistry_of, held_out)
    top = float(cells["x"].mean())
    links = cells.assign(x0=top, y0=LEVELS["fleet"], x1=cells["x"], y1=LEVELS["cell"])
    layers = (
        geom_segment(aes(x="x0", xend="x1", y="y0", yend="y1"), data=links, color=SLATE, size=0.5)
        + geom_point(aes("x", "y"), data=pd.DataFrame({"x": [top], "y": [LEVELS["fleet"]]}), color=SLATE, size=8.5)
        + label(
            pd.DataFrame({"x": [top + 0.7], "y": [LEVELS["fleet"]], "label": ["One curve for the fleet"]}),
            INK,
            size=5.6,
        )
    )
    return _frame(cells, "Every cell, and a new cell, receives the same curve", layers).__add__(_colours(colours))


def none(chemistry_of: dict[str, str], held_out: list[str], colours: dict[str, str]):  # noqa: ANN201
    """No pooling: every cell has a curve of its own, and a new cell has nothing but the prior.

    Args:
        chemistry_of: Chemistry of each cell, in the order to be drawn.
        held_out: Cells kept back for testing, drawn as rings.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    cells = _cells(chemistry_of, held_out)
    own = cells[~cells["held"]].assign(y=LEVELS["chemistry"])
    links = own.assign(x0=own["x"], y0=own["y"], x1=own["x"], y1=LEVELS["cell"])
    layers = (
        geom_segment(aes(x="x0", xend="x1", y="y0", yend="y1"), data=links, color=SLATE, size=0.5)
        + geom_point(aes("x", "y", color="chemistry"), data=own, shape=15, size=4.6)
        + label(
            pd.DataFrame(
                {
                    "x": [0.4],
                    "y": [LEVELS["chemistry"] + 0.55],
                    "label": ["A curve for each cell, learned from its own record"],
                }
            ),
            INK,
            size=5.6,
        )
    )
    return _frame(cells, "A new cell has no curve, only the prior", layers).__add__(_colours(colours))


def partial(chemistry_of: dict[str, str], held_out: list[str], colours: dict[str, str]):  # noqa: ANN201
    """Partial pooling: the fleet sets what chemistries are like, and each chemistry sets what its cells are like.

    A held-out cell is joined to its chemistry by a dashed segment: a new cell starts from the chemistry it belongs to.

    Args:
        chemistry_of: Chemistry of each cell, in the order to be drawn.
        held_out: Cells kept back for testing, drawn as rings.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    cells = _cells(chemistry_of, held_out)
    middle = cells.groupby("chemistry")["x"].mean().rename("cx").reset_index()
    top = float(middle["cx"].mean())
    chems = middle.assign(y=LEVELS["chemistry"])
    fleet_links = chems.assign(x0=top, y0=LEVELS["fleet"], x1=chems["cx"], y1=LEVELS["chemistry"])
    cell_links = cells.merge(middle, on="chemistry").assign(
        x0=lambda rows: rows["cx"], y0=LEVELS["chemistry"], x1=lambda rows: rows["x"], y1=LEVELS["cell"]
    )
    layers = (
        geom_segment(aes(x="x0", xend="x1", y="y0", yend="y1"), data=fleet_links, color=SLATE, size=0.7)
        + geom_segment(
            aes(x="x0", xend="x1", y="y0", yend="y1"), data=cell_links[~cell_links["held"]], color=SLATE, size=0.5
        )
        + geom_segment(
            aes(x="x0", xend="x1", y="y0", yend="y1"),
            data=cell_links[cell_links["held"]],
            color=SLATE,
            size=0.7,
            linetype="dashed",
        )
        + geom_point(aes("cx", "y", color="chemistry"), data=chems, size=8.5)
        + geom_point(aes("x", "y"), data=pd.DataFrame({"x": [top], "y": [LEVELS["fleet"]]}), color=SLATE, size=8.5)
        + label(
            pd.DataFrame({"x": [top + 0.7], "y": [LEVELS["fleet"]], "label": ["Fleet: what chemistries are like"]}),
            INK,
            size=5.6,
        )
        + label(
            chems.iloc[[0]].assign(
                x=chems["cx"].iloc[0] - 0.6,
                y=LEVELS["chemistry"] + 0.45,
                label=chems["chemistry"].iloc[0] + ": level, rate, amplitude",
            ),
            INK,
            hjust=1,
            size=5.4,
        )
        + label(
            chems.iloc[[1]].assign(
                x=chems["cx"].iloc[1] + 0.6,
                y=LEVELS["chemistry"] + 0.45,
                label=chems["chemistry"].iloc[1] + ": level, rate, amplitude",
            ),
            INK,
            hjust=0,
            size=5.4,
        )
    )
    return _frame(cells, "Cells stay close to their chemistry, and a new cell starts there", layers).__add__(
        _colours(colours)
    )


def _colours(colours: dict[str, str]):  # noqa: ANN202
    """The colour scale of the chemistries."""
    from lets_plot import scale_color_manual

    return scale_color_manual(values={**colours, "fleet": INK})
