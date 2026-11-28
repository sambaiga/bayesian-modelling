"""Figures of the time-to-end-of-life study: the records of a stopped test, the two models and how well they predict.

Every function takes tidy data frames and returns a Lets-Plot figure; none reads a model or a file. Colour is by role:
the first chemistry is blue, the second orange, the reference slate and what the test did not see grey.
"""

from __future__ import annotations

from lets_plot import (
    aes,
    element_blank,
    element_text,
    geom_density,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_segment,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_linetype_manual,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_discrete,
    theme,
)
import numpy as np
import pandas as pd

from .hierarchy import label, tint
from .theme import GREY, INK, SLATE, canvas, site_theme


def _order(table: pd.DataFrame, colours: dict[str, str]) -> list[str]:
    """Cells grouped by chemistry in the order of ``colours``, each group by end of life."""
    rank = {name: index for index, name in enumerate(colours)}
    ordered = table.assign(rank=table["CellType"].map(rank)).sort_values(["rank", "eol"], ascending=[True, True])
    return ordered["cell"].tolist()


def stops(table: pd.DataFrame, stop: float, colours: dict[str, str]):  # noqa: ANN201
    """The records of a test that stops at a cycle: crossings that were seen and cells that were still running.

    A filled point is a crossing that the test saw. A hollow point at the stop is a cell that was still in service, and
    the dashed grey line runs on to the end of life that the test did not see.

    Args:
        table: Rows with ``cell``, ``CellType``, ``eol``, ``time`` and ``event``, from ``bayes.survival.truncate``.
        stop: The cycle at which the test stops.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    order = _order(table, colours)
    rows = table.assign(colour=table["CellType"], start=0.0)
    hidden = rows[~rows["event"]]
    seen = rows[rows["event"]]
    plot = (
        ggplot(rows, aes(y="cell"))
        + geom_segment(aes(x="start", xend="time", y="cell", yend="cell", color="colour"), size=1.1)
        + geom_segment(
            aes(x="time", xend="eol", y="cell", yend="cell"),
            data=hidden,
            color=GREY,
            linetype="dashed",
            size=0.8,
        )
        + geom_point(aes("eol", "cell"), data=hidden, color=GREY, shape=21, fill="white", size=2.2, stroke=0.8)
        + geom_point(aes("time", "cell", color="colour"), data=seen, size=3.2)
        + geom_point(aes("time", "cell"), data=hidden, color=INK, shape=21, fill="white", size=3.2, stroke=1.0)
        + geom_vline(xintercept=stop, color=SLATE, linetype="dashed", size=0.6)
        + label(pd.DataFrame({"x": [stop + 25], "y": [order[0]], "label": ["Test stops"]}), SLATE, size=5.0)
        + scale_color_manual(values=colours)
        + scale_y_discrete(limits=order[::-1])
        + scale_x_continuous(limits=[0, float(rows["eol"].max()) * 1.05], breaks=[0, 500, 1000, 1500])
        + labs(x="Cycle", y="")
        + site_theme()
        + canvas(330, 300)
    )
    return plot


def prior_lifetimes(prior: np.ndarray, observed: np.ndarray, colour: str):  # noqa: ANN201
    """The lifetimes that the priors imply, against the end of life of the cells that were observed.

    Args:
        prior: Draws of the prior predictive lifetime in cycles.
        observed: The observed end of life of each cell in cycles.
        colour: Colour of the observed points.

    Returns:
        The figure.
    """
    low, high = np.percentile(prior, [1.0, 96.0])
    trimmed = pd.DataFrame({"value": prior[(prior >= low) & (prior <= high)]})
    top = float(np.max(np.histogram(trimmed["value"], bins=40, density=True)[0]))
    points = pd.DataFrame({"x": np.asarray(observed, float), "y": 0.0})
    names = pd.DataFrame(
        {
            "x": [float(np.median(trimmed["value"])), float(np.median(observed))],
            "y": [top * 1.08, top * 0.12],
            "label": ["Prior", "Observed"],
        }
    )
    return (
        ggplot()
        + geom_density(aes("value"), data=trimmed, color=SLATE, fill=SLATE, alpha=0.2, size=1.0, trim=True)
        + geom_point(aes("x", "y"), data=points, color=colour, shape=21, fill="white", size=3.0, stroke=1.0)
        + label(names.iloc[[0]], INK, hjust=0.5, size=5.4)
        + label(names.iloc[[1]], colour, hjust=0.5, size=5.4)
        + scale_x_continuous(limits=[0, float(high)])
        + labs(x="Cycles to end of life", y="")
        + theme(axis_line_y=element_blank(), axis_text_y=element_blank(), axis_ticks_y=element_blank())
        + site_theme()
        + canvas(440, 300)
    )


def survival_bands(bands: pd.DataFrame, table: pd.DataFrame, stop: float, colours: dict[str, str]):  # noqa: ANN201
    """Posterior survival curve of each chemistry, with the crossings that the test saw marked on the axis.

    Args:
        bands: Rows with ``chemistry``, ``cycle``, ``median``, ``lower`` and ``upper``.
        table: Rows with ``CellType``, ``eol`` and ``event``, from ``bayes.survival.truncate``.
        stop: The cycle at which the test stopped.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    seen = table[table["event"]].assign(y=-0.04)
    middle = bands.reset_index(drop=True).assign(distance=lambda rows: (rows["median"] - 0.5).abs())
    names = middle.loc[middle.groupby("chemistry")["distance"].idxmin()]
    names = names.assign(x=names["cycle"] + 40, y=names["median"], label=names["chemistry"])
    plot = (
        ggplot(bands, aes("cycle"))
        + geom_vline(xintercept=stop, color=SLATE, linetype="dashed", size=0.6)
        + geom_ribbon(aes(ymin="lower", ymax="upper", group="chemistry", fill="chemistry"), size=0)
        + scale_fill_manual(values={name: tint(colour, 0.75) for name, colour in colours.items()})
        + geom_line(aes(y="median", group="chemistry", color="chemistry"), size=1.2)
        + scale_color_manual(values=colours)
        + geom_point(aes("eol", "y", color="CellType"), data=seen, size=2.6)
        + label(names, INK, size=5.4)
    )
    return (
        plot
        + scale_x_continuous(limits=[0, float(bands["cycle"].max()) + 260], breaks=[0, 500, 1000, 1500, 2000])
        + scale_y_continuous(limits=[-0.08, 1.0], breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + labs(x="Cycle", y="Probability still in service")
        + site_theme()
        + canvas(330, 300)
    )


def paths(  # noqa: ANN201
    curves: pd.DataFrame,
    observed: pd.DataFrame,
    table: pd.DataFrame,
    stop: float,
    colour: str,
    loss: float = 20.0,
):
    """Fitted loss paths of one chemistry with the measurements seen and not seen, and the observed end of life.

    Args:
        curves: Rows with ``cell``, ``cycle`` and ``median`` for the cells of one chemistry.
        observed: Rows with ``cell``, ``cycle`` and ``loss`` for the same cells, every measurement.
        table: Rows with ``cell`` and ``eol`` for the same cells.
        stop: The cycle at which the test stopped.
        colour: Colour of the chemistry.
        loss: The loss in points that defines the end of life.

    Returns:
        The figure.
    """
    seen = observed[observed["cycle"] <= stop]
    unseen = observed[observed["cycle"] > stop]
    before, after = curves[curves["cycle"] <= stop], curves[curves["cycle"] >= stop]
    ends = table.assign(y=loss)
    top = loss * 1.6
    return (
        ggplot()
        + geom_hline(yintercept=loss, color=SLATE, linetype="dotted", size=0.5)
        + geom_vline(xintercept=stop, color=SLATE, linetype="dashed", size=0.6)
        + geom_point(aes("cycle", "loss", group="cell"), data=unseen, color=GREY, size=0.8)
        + geom_point(
            aes("cycle", "loss", group="cell"), data=seen, color=INK, shape=21, fill="white", size=0.9, stroke=0.4
        )
        + geom_line(aes("cycle", "median", group="cell"), data=before, color=colour, size=1.0)
        + geom_line(aes("cycle", "median", group="cell"), data=after, color=colour, size=0.9, linetype="dashed")
        + geom_point(aes("eol", "y"), data=ends, color=INK, shape=23, fill="white", size=3.0, stroke=0.9)
        + scale_y_continuous(limits=[-2, top], breaks=[0, 10, 20, 30])
        + scale_x_continuous(limits=[0, 1500], breaks=[0, 500, 1000, 1500])
        + labs(x="Cycle", y="Capacity lost (points of state of health)")
        + site_theme()
        + canvas(440, 300)
    )


def intervals(  # noqa: ANN201
    scores: pd.DataFrame, stop: float, colours: dict[str, str], upto: float = 2000.0
):
    """The 90% interval and median predicted for each running cell, against the end of life that it reached.

    A diamond is the observed end of life. An interval that runs past ``upto`` is cut there.

    Args:
        scores: Rows with ``cell``, ``CellType``, ``eol``, ``median``, ``lower`` and ``upper``, for one method.
        stop: The cycle at which the test stopped.
        colours: Colour of each chemistry.
        upto: Highest cycle drawn.

    Returns:
        The figure.
    """
    rows = scores.assign(upper=scores["upper"].clip(upper=upto), median=scores["median"].clip(upper=upto))
    order = _order(scores, colours)
    return (
        ggplot(rows)
        + geom_vline(xintercept=stop, color=SLATE, linetype="dashed", size=0.6)
        + geom_segment(aes(x="lower", xend="upper", y="cell", yend="cell", color="CellType"), size=2.2, alpha=0.45)
        + geom_point(aes("median", "cell", color="CellType"), size=3.0)
        + geom_point(aes("eol", "cell"), color=INK, shape=23, fill="white", size=3.2, stroke=1.0)
        + scale_color_manual(values=colours)
        + scale_y_discrete(limits=order[::-1])
        + scale_x_continuous(limits=[stop - 60, upto], breaks=[500, 1000, 1500, 2000])
        + labs(x="Cycle", y="")
        + site_theme()
        + canvas(440, 300)
    )


def accuracy(summary: pd.DataFrame, value: str, ylabel: str, methods: dict[str, str], reference: float | None = None):  # noqa: ANN201
    """One measure of the predictions of each method against the cycle at which the test stopped.

    Args:
        summary: Rows with ``stop``, ``method`` and the column ``value``.
        value: Name of the column to draw.
        ylabel: Label of the vertical axis.
        methods: Colour of each method; the first is drawn solid and the others dashed.
        reference: A level to mark with a dotted line, such as the nominal coverage.

    Returns:
        The figure, with its key in one row below the axis.
    """
    names = list(methods)
    plot = ggplot(summary, aes("stop", value, color="method", linetype="method"))
    if reference is not None:
        plot += geom_hline(yintercept=reference, color=SLATE, linetype="dotted", size=0.5)
    return (
        plot
        + geom_line(size=1.2)
        + geom_point(size=2.6)
        + scale_color_manual(values=methods)
        + scale_linetype_manual(values={name: "solid" if index == 0 else "dashed" for index, name in enumerate(names)})
        + scale_x_continuous(breaks=sorted(summary["stop"].unique()))
        + labs(x="Cycle at which the test stops", y=ylabel)
        + site_theme()
        + theme(legend_position="bottom", legend_title=element_blank(), axis_title_y=element_text(size=10))
        + canvas(330, 300)
    )
