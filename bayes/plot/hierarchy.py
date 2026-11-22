"""Figures of the hierarchical battery study, in the style of the blog: one message each, direct labels, no legend.

Every function takes tidy data frames and returns a Lets-Plot figure; none reads a model or a file. The posterior
summaries that feed them are made in ``bayes.regression.hierarchical``.
"""

from __future__ import annotations

from lets_plot import (
    aes,
    element_blank,
    element_text,
    facet_grid,
    facet_wrap,
    geom_density,
    geom_hline,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_segment,
    geom_text,
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

from .theme import BLUE, GREY, INK, ORANGE, SITE_FONT, SLATE, canvas, columns, site_theme

STRATEGIES = {
    "none": "No pooling",
    "complete": "Complete pooling",
    "partial": "Partial pooling",
    "control": "Complete pooling, no diagnostics",
}


def tint(colour: str, amount: float = 0.85) -> str:
    """Mix a colour with white, for bands and fills.

    Args:
        colour: Colour as ``#RRGGBB``.
        amount: Share of white, from 0 (the colour) to 1 (white).

    Returns:
        The mixed colour as ``#RRGGBB``.
    """
    rgb = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{int(value):02X}" for value in rgb + (255 - rgb) * amount)


def palette(names: list[str]) -> dict[str, str]:
    """One colour per chemistry: blue for the first, orange for the second, then slate and grey.

    Args:
        names: Names in a fixed order.

    Returns:
        A mapping from name to colour.

    Raises:
        ValueError: If there are more than four names, since colour does not identify more groups reliably.
    """
    if len(names) > 4:
        raise ValueError("colour identifies at most four groups")
    return dict(zip(names, [BLUE, ORANGE, SLATE, GREY], strict=False))


def label(frame: pd.DataFrame, colour: str = INK, hjust: float = 0.0, size: float = 5.2):  # noqa: ANN201
    """Direct labels with a white halo, from a frame with columns ``x``, ``y`` and ``label``.

    Args:
        frame: Positions and texts of the labels, with the facet columns when the figure has facets.
        colour: Colour of the text.
        hjust: Horizontal justification, 0 for left and 1 for right.
        size: Text size.

    Returns:
        A Lets-Plot text layer.
    """
    return geom_text(
        aes("x", "y", label="label"),
        data=frame,
        color=colour,
        hjust=hjust,
        size=size,
        family=SITE_FONT,
        halo_width=2,
        halo_color="white",
        inherit_aes=False,
    )


def spread(values: list[float], gap: float) -> list[float]:
    """Move values apart so that neighbours are at least ``gap`` apart, keeping their order, for labels at line ends.

    Args:
        values: Positions, in any order.
        gap: Smallest distance between two neighbours.

    Returns:
        The adjusted positions, in the order of the input.
    """
    order = np.argsort(values)
    placed: list[float] = []
    for index in order:
        position = values[index]
        placed.append(max(position, placed[-1] + gap) if placed else position)
    result = np.empty(len(values))
    result[order] = placed
    return result.tolist()


def curves(frame: pd.DataFrame, held_out: list[str], colours: dict[str, str], threshold: float = 80.0):  # noqa: ANN201
    """State of health against cycle for every cell, with the held-out cells dashed.

    Args:
        frame: Rows with ``cell``, ``CellType``, ``cycle`` and ``soh``.
        held_out: Cells kept back for testing.
        colours: Colour of each chemistry.
        threshold: End-of-life threshold in percent.

    Returns:
        The figure.
    """
    drawn = frame.assign(held=np.where(frame["cell"].isin(held_out), "Held out", "Training"))
    ends = drawn.sort_values("cycle").groupby("CellType").tail(1).assign(label=lambda rows: rows["CellType"])
    return (
        ggplot(drawn, aes("cycle", "soh", group="cell"))
        + geom_hline(yintercept=threshold, color=SLATE, linetype="dotted", size=0.5, inherit_aes=False)
        + geom_line(aes(color="CellType", linetype="held"), size=0.9)
        + scale_color_manual(values=colours)
        + scale_linetype_manual(values={"Training": "solid", "Held out": "dashed"})
        + label(
            pd.DataFrame({"x": [60.0], "y": [threshold + 3.0], "label": [f"End of life, {threshold:.0f}%"]}),
            SLATE,
            size=5.0,
        )
        + label(ends.assign(x=ends["cycle"] + 40, y=ends["soh"]), size=5.4)
        + scale_x_continuous(limits=[0, drawn["cycle"].max() + 280])
        + labs(x="Cycle", y="State of health (%)")
        + site_theme()
        + canvas(780, 340)
    )


def prior(observed: np.ndarray, drawn: np.ndarray):  # noqa: ANN201
    """Density of the observed state of health against the prior predictive density.

    Args:
        observed: Observed values in percent.
        drawn: Prior predictive draws in percent.

    Returns:
        The figure.
    """
    comparison = pd.DataFrame(
        {
            "soh": np.concatenate([observed, drawn]),
            "source": ["Observed"] * len(observed) + ["Prior predictive"] * len(drawn),
        }
    )
    peaks = []
    for source, colour in (("Observed", BLUE), ("Prior predictive", SLATE)):
        heights, edges = np.histogram(comparison.loc[comparison["source"] == source, "soh"], bins=40, density=True)
        top = int(np.argmax(heights))
        peaks.append(
            {
                "x": float((edges[top] + edges[top + 1]) / 2),
                "y": float(heights[top]) * 1.06,
                "label": source,
                "colour": colour,
            }
        )
    peaks = pd.DataFrame(peaks)
    return (
        ggplot(comparison, aes("soh", color="source", fill="source"))
        + geom_density(size=1.0, alpha=0.18, trim=True)
        + scale_color_manual(values={"Observed": BLUE, "Prior predictive": SLATE})
        + scale_fill_manual(values={"Observed": BLUE, "Prior predictive": SLATE})
        + label(peaks.iloc[[0]], BLUE, hjust=0.5, size=5.4)
        + label(peaks.iloc[[1]], SLATE, hjust=0.5, size=5.4)
        + scale_x_continuous(breaks=[0, 25, 50, 75, 100])
        + labs(x="State of health (%)", y="Density")
        + site_theme()
        + canvas(720, 330)
    )


def chemistry(intervals: pd.DataFrame, colours: dict[str, str]):  # noqa: ANN201
    """Posterior mean and interval of the chemistry parameters, one panel per parameter.

    Args:
        intervals: Rows with ``quantity``, ``chemistry``, ``mean``, ``lower`` and ``upper``.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    names = list(colours)
    frame = intervals.assign(row=intervals["chemistry"].map({name: index + 1 for index, name in enumerate(names)}))
    return (
        ggplot(frame)
        + geom_segment(aes(x="lower", xend="upper", y="row", yend="row", color="chemistry"), size=1.8)
        + geom_point(aes("mean", "row", color="chemistry"), shape=21, fill="white", size=4.5, stroke=1.7)
        + scale_color_manual(values=colours)
        + label(
            frame.assign(x=frame["mean"], y=frame["row"] + 0.33, label=frame["mean"].map("{:.2f}".format)), hjust=0.5
        )
        + facet_wrap("quantity", ncol=columns(3), scales="free_x")
        + scale_y_continuous(breaks=list(range(1, len(names) + 1)), labels=names, limits=[0.5, len(names) + 0.7])
        + labs(x="Posterior mean and 90% interval", y="")
        + site_theme()
        + canvas(840, 260, 3, 3)
    )


def shrinkage(estimates: pd.DataFrame, means: pd.DataFrame, colours: dict[str, str], cells: list[str]):  # noqa: ANN201
    """Per-cell estimates without pooling and with partial pooling, joined by a segment, against the chemistry mean.

    Args:
        estimates: Rows with ``parameter``, ``method`` (``"No pooling"`` or ``"Partial pooling"``), ``cell``,
            ``chemistry`` and ``mean``.
        means: Rows with ``parameter``, ``chemistry``, ``x0``, ``x1`` and ``value``: the span and level of a chemistry.
        colours: Colour of each chemistry.
        cells: Cells in the order of the horizontal axis.

    Returns:
        The figure.
    """
    frame = estimates.assign(position=estimates["cell"].map({cell: index + 1 for index, cell in enumerate(cells)}))
    frame["fill"] = np.where(frame["method"] == "Partial pooling", frame["chemistry"], "none")
    wide = frame.pivot_table(index=["parameter", "cell", "position"], columns="method", values="mean").reset_index()
    return (
        ggplot()
        + geom_segment(
            aes(x="x0", xend="x1", y="value", yend="value"), data=means, color=SLATE, linetype="dashed", size=0.5
        )
        + geom_segment(
            aes(x="position", xend="position", y="No pooling", yend="Partial pooling"), data=wide, color=SLATE, size=0.5
        )
        + geom_point(
            aes("position", "mean", color="chemistry", fill="fill"), data=frame, shape=21, size=3.4, stroke=1.3
        )
        + scale_color_manual(values=colours)
        + scale_fill_manual(values={**colours, "none": "white"})
        + facet_wrap("parameter", ncol=columns(2), scales="free_y")
        + scale_x_continuous(breaks=list(range(1, len(cells) + 1)), labels=cells)
        + labs(x="", y="")
        + site_theme()
        + theme(axis_text_x=element_text(angle=60, hjust=1, size=9))
        + canvas(840, 380, 2, 2)
    )


def predictions(frame: pd.DataFrame):  # noqa: ANN201
    """Median and 90% interval of each strategy against the observed state of health, one row per cell.

    Args:
        frame: Rows with ``cell``, ``method``, ``cycle``, ``observed``, ``median``, ``lower`` and ``upper``.

    Returns:
        The figure.
    """
    phone = columns(0, 1) == 1
    drawn = frame.assign(panel=frame["method"] + ", " + frame["cell"]) if phone else frame
    layout = facet_wrap("panel", ncol=1) if phone else facet_grid(x="method", y="cell")
    return (
        ggplot(drawn, aes("cycle"))
        + geom_ribbon(aes(ymin="lower", ymax="upper"), fill=tint(BLUE, 0.8), size=0)
        + geom_line(aes(y="median"), color=BLUE, size=1.0)
        + geom_point(aes(y="observed"), color=INK, shape=21, fill="white", size=1.6, stroke=0.7)
        + layout
        + scale_y_continuous(limits=[0, 110])
        + labs(x="Cycle", y="State of health (%)")
        + site_theme()
        + canvas(840, 460, frame["method"].nunique() * frame["cell"].nunique(), frame["method"].nunique(), 1)
    )


def loo(table: pd.DataFrame, best: str):  # noqa: ANN201
    """Loss in expected log predictive density against the best model, with a one-standard-error bar.

    Args:
        table: Rows with ``method``, ``elpd_diff`` and ``dse``, as returned by ``arviz.compare`` with the index reset.
        best: Name of the best model, drawn as a filled point.

    Returns:
        The figure.
    """
    frame = table.assign(row=np.arange(len(table), 0, -1))
    frame["low"], frame["high"] = frame["elpd_diff"] - frame["dse"], frame["elpd_diff"] + frame["dse"]
    return (
        ggplot(frame)
        + geom_vline(xintercept=0, color=SLATE, linetype="dotted", size=0.5)
        + geom_segment(aes(x="low", xend="high", y="row", yend="row"), color=SLATE, size=1.0)
        + geom_point(aes("elpd_diff", "row", fill="method"), shape=21, color=BLUE, size=4.0, stroke=1.4)
        + scale_fill_manual(values={name: (BLUE if name == best else "white") for name in frame["method"]})
        + label(frame.assign(x=frame["elpd_diff"], y=frame["row"] + 0.38, label=frame["method"]), hjust=0.5)
        + scale_y_continuous(breaks=[], limits=[0.4, len(frame) + 0.7])
        + labs(x="Loss in expected log predictive density against the best model (smaller is better)", y="")
        + site_theme()
        + theme(axis_line_y=element_blank(), axis_ticks_y=element_blank())
        + canvas(780, 240)
    )


def heldout(frame: pd.DataFrame):  # noqa: ANN201
    """Median and 90% interval against the observed state of health, one panel for each held-out cell.

    Args:
        frame: Rows with ``cell``, ``cycle``, ``observed``, ``median``, ``lower`` and ``upper``.

    Returns:
        The figure.
    """
    return (
        ggplot(frame, aes("cycle"))
        + geom_ribbon(aes(ymin="lower", ymax="upper"), fill=tint(BLUE, 0.8), size=0)
        + geom_line(aes(y="median"), color=BLUE, size=1.0)
        + geom_point(aes(y="observed"), color=INK, shape=21, fill="white", size=1.4, stroke=0.7)
        + facet_wrap("cell", ncol=columns(3, 2), scales="free_x")
        + scale_y_continuous(limits=[0, 110])
        + labs(x="Cycle", y="State of health (%)")
        + site_theme()
        + canvas(900, 400, frame["cell"].nunique(), 3, 2)
    )


def effects(table: pd.DataFrame):  # noqa: ANN201
    """Posterior mean and 90% interval of the effect of each standardised diagnostic.

    Args:
        table: Rows with ``feature``, ``mean``, ``lower`` and ``upper``.

    Returns:
        The figure.
    """
    ordered = table.sort_values("mean").reset_index(drop=True)
    return (
        ggplot(ordered, aes(y="feature"))
        + geom_vline(xintercept=0.0, color=SLATE, linetype="dotted", size=0.5)
        + geom_segment(aes(x="lower", xend="upper", yend="feature"), color=BLUE, size=1.3)
        + geom_point(aes(x="mean"), color=BLUE, shape=21, fill="white", size=3.4, stroke=1.4)
        + scale_y_discrete(limits=list(ordered["feature"]))
        + labs(x="Effect on the logit of the state of health, per standard deviation", y="")
        + site_theme()
        + canvas(780, 240)
    )
