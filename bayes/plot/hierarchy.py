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
    scale_size_manual,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_discrete,
    theme,
)
import numpy as np
import pandas as pd

from .distribution import plot_density
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
        + canvas(620, 300)
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
        + canvas(540, 300)
    )


def chemistry(  # noqa: ANN201
    draws: pd.DataFrame, colours: dict[str, str], quantity: str, size: tuple[int, int] = (330, 240)
):
    """Posterior density of one chemistry parameter, one curve for each chemistry.

    The overlap of the curves is what the reader needs: the further apart they lie, the more surely the chemistries
    differ in that parameter. Each parameter is a figure of its own so that it becomes a subfigure.

    Args:
        draws: Rows from ``hierarchical.chemistry_draws`` with ``quantity``, ``chemistry`` and ``value``.
        colours: Colour of each chemistry.
        quantity: The parameter to draw.
        size: Width and height of the figure in pixels.

    Returns:
        The figure.
    """
    rows = draws[draws["quantity"] == quantity].rename(columns={"chemistry": "distribution"})
    names = [name for name in colours if name in set(rows["distribution"])]
    plot = plot_density(
        rows,
        title=None,
        x_label=quantity,
        y_label="",
        colours=[colours[name] for name in names],
        line_size=1.6,
        theme=None,
    )
    return (
        plot
        + site_theme()
        + theme(
            legend_position="top",
            legend_title=element_blank(),
            axis_line_y=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_title_x=element_text(face="plain", size=11),
        )
        + canvas(*size)
    )


def shrinkage(  # noqa: ANN201
    estimates: pd.DataFrame,
    means: pd.DataFrame,
    colours: dict[str, str],
    cells: list[str],
    parameter: str | None = None,
):
    """Per-cell estimates without pooling and with partial pooling, joined by a segment, against the chemistry mean.

    Args:
        estimates: Rows with ``parameter``, ``method`` (``"No pooling"`` or ``"Partial pooling"``), ``cell``,
            ``chemistry`` and ``mean``.
        means: Rows with ``parameter``, ``chemistry``, ``x0``, ``x1`` and ``value``: the span and level of a chemistry.
        colours: Colour of each chemistry.
        cells: Cells in the order of the horizontal axis.
        parameter: The parameter to draw, so that each is a subfigure of its own; both in one figure when ``None``.

    Returns:
        The figure.
    """
    if parameter is not None:
        estimates, means = estimates[estimates["parameter"] == parameter], means[means["parameter"] == parameter]
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
        + scale_x_continuous(breaks=list(range(1, len(cells) + 1)), labels=cells)
        + labs(x="", y=parameter or "")
        + site_theme()
        + theme(axis_text_x=element_text(angle=60, hjust=1, size=9))
        + (
            canvas(780, 300)
            if parameter
            else facet_wrap("parameter", ncol=columns(2), scales="free_y") + canvas(840, 380, 2, 2)
        )
    )


def bands(  # noqa: ANN201
    frame: pd.DataFrame,
    colours: dict[str, str],
    seen: int = 0,
    threshold: float = 80.0,
    size: tuple[int, int] = (780, 340),
):
    """Median and 90% interval of one strategy for the held-out cells, with the observations and a key.

    The band and the median take the colour of the chemistry of the cell. A band that the observations leave is the
    picture of an over-confident prediction. The key above the plot names every kind of mark: the median, the interval,
    the observations, the end-of-life threshold and, when the model has seen cycles, the cycles seen. The cells are
    named at the ends of their lines.

    Args:
        frame: Rows of one strategy and one number of cycles seen, with ``cell``, ``CellType``, ``cycle``, ``soh``,
            ``median``, ``lower`` and ``upper``.
        colours: Colour of each chemistry.
        seen: Cycles of the cell that the model has seen; a dashed line marks them when positive.
        threshold: End-of-life threshold in percent.
        size: Width and height of the figure in pixels.

    Returns:
        The figure.
    """
    ends = frame.sort_values("cycle").groupby("cell").tail(1)
    ends = ends.assign(x=ends["cycle"] + 40, y=ends["median"], label=ends["cell"])
    reach = float(frame["cycle"].max())
    width = reach + 300
    plot = (
        ggplot(frame, aes("cycle"))
        + geom_hline(yintercept=threshold, color=SLATE, linetype="dotted", size=0.5)
        + geom_ribbon(aes(ymin="lower", ymax="upper", group="cell", fill="CellType"), size=0)
        + scale_fill_manual(values={name: tint(colour, 0.75) for name, colour in colours.items()})
        + geom_line(aes(y="median", group="cell", color="CellType"), size=1.2)
        + scale_color_manual(values=colours)
        + geom_point(aes(y="soh", group="cell"), color=INK, shape=21, fill="white", size=1.4, stroke=0.6)
        + label(ends, INK, size=5.4)
    )
    if seen:
        plot += geom_vline(xintercept=seen, color=SLATE, linetype="dashed", size=0.6)
    # the key: two rows above the plot, one entry for each kind of mark; the colour of a chemistry is named at the lines
    swatch = width * 0.07
    columns_at = [width * 0.02, width * 0.36, width * 0.70]
    first, second = 126.0, 117.0

    def entry(kind: str, text: str, x0: float, y: float) -> list:
        line = pd.DataFrame({"x0": [x0], "x1": [x0 + swatch], "y": [y]})
        spec = aes(x="x0", xend="x1", y="y", yend="y")
        if kind == "median":
            marks = [geom_segment(spec, data=line, color=SLATE, size=1.2, inherit_aes=False)]
        elif kind == "interval":
            marks = [geom_segment(spec, data=line, color=tint(SLATE, 0.7), size=6.0, inherit_aes=False)]
        elif kind == "point":
            point = pd.DataFrame({"x": [x0 + swatch / 2], "y": [y]})
            marks = [
                geom_point(
                    aes("x", "y"),
                    data=point,
                    color=INK,
                    shape=21,
                    fill="white",
                    size=1.8,
                    stroke=0.7,
                    inherit_aes=False,
                )
            ]
        else:
            marks = [geom_segment(spec, data=line, color=SLATE, linetype=kind, size=0.7, inherit_aes=False)]
        words = pd.DataFrame({"x": [x0 + swatch * 1.25], "y": [y], "label": [text]})
        return [*marks, label(words, INK, size=4.8)]

    rows = [
        ("median", "Median", columns_at[0], first),
        ("interval", "90% interval", columns_at[1], first),
        ("point", "Observation", columns_at[2], first),
        ("dotted", f"End of life, {threshold:.0f}%", columns_at[0], second),
    ]
    if seen:
        rows.append(("dashed", "Cycles seen", columns_at[1], second))
    for kind, text, x0, y in rows:
        for layer in entry(kind, text, x0, y):
            plot += layer
    return (
        plot
        + scale_x_continuous(limits=[0, width])
        + scale_y_continuous(limits=[0, 132], breaks=[0, 20, 40, 60, 80, 100])
        + labs(x="Cycle", y="State of health (%)")
        + site_theme()
        + canvas(*size)
    )


def priors(frame: pd.DataFrame, quantity: str, colours: dict[str, str]):  # noqa: ANN201
    """The prior of a parameter and its posterior, as densities on one axis, to show what the data taught the model.

    Args:
        frame: Rows with ``quantity``, ``source`` and ``value``, from ``hierarchical.prior_posterior``.
        quantity: The parameter to draw.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    rows = frame[frame["quantity"] == quantity]
    low, high = np.percentile(rows["value"], [0.5, 99.5])  # the long tails of a prior would flatten the figure
    rows = rows[rows["value"].between(low, high)]  # dropped here, so that the axis never has to clip draws
    shades = {"Prior": SLATE, "Posterior": BLUE, **colours}
    peaks = []
    for source in dict.fromkeys(rows["source"]):
        heights, edges = np.histogram(rows.loc[rows["source"] == source, "value"], bins=60, density=True)
        top = int(np.argmax(heights))
        peaks.append(
            {
                "x": float((edges[top] + edges[top + 1]) / 2),
                "y": float(heights[top]) * 1.07,
                "label": source,
                "source": source,
            }
        )
    peaks = pd.DataFrame(peaks)
    plot = ggplot(rows, aes("value", color="source", fill="source")) + geom_density(size=1.0, alpha=0.2, trim=True)
    for source in peaks["source"]:
        plot += label(peaks[peaks["source"] == source], INK, hjust=0.5, size=5.4)
    return (
        plot
        + scale_color_manual(values=shades)
        + scale_fill_manual(values=shades)
        + scale_x_continuous(limits=[float(rows["value"].min()), float(rows["value"].max())])
        + labs(x=quantity, y="")
        + theme(axis_line_y=element_blank(), axis_text_y=element_blank(), axis_ticks_y=element_blank())
        + site_theme()
        + canvas(440, 270)
    )


def sensitivity(draws: pd.DataFrame, chemistry: str, quantity: str = "Log rate (per 1,000 cycles)"):  # noqa: ANN201
    """The posterior of a chemistry parameter under several priors, as overlaid densities with a legend.

    The curves of one chemistry lie on top of one another when the prior does not decide the answer. The densities are
    drawn by ``bayes.plot.distribution.plot_density``, one panel for each chemistry so that each is a subfigure.

    Args:
        draws: Rows from ``hierarchical.sensitivity_draws`` with ``setting``, ``chemistry`` and ``value``.
        chemistry: The chemistry to draw.
        quantity: Name of the parameter, for the axis.

    Returns:
        The figure.
    """
    rows = draws[draws["chemistry"] == chemistry].rename(columns={"setting": "distribution"})
    shades = [SLATE, BLUE, tint(BLUE, 0.5)]  # the default prior is the subject; the other two recede
    plot = plot_density(
        rows,
        title=None,
        x_label=quantity,
        y_label="",
        colours=shades[: rows["distribution"].nunique()],
        line_size=1.6,
        theme=None,
    )
    return (
        plot
        + site_theme()
        + theme(
            legend_position="top",
            legend_title=element_blank(),
            axis_line_y=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_title_x=element_text(face="plain", size=11),
        )
        + canvas(440, 300)
    )


def value(frame: pd.DataFrame, floor: float = -1.0):  # noqa: ANN201
    """Relative economic value of each strategy against the cost-loss ratio, for one number of cycles seen.

    Args:
        frame: Rows of one number of cycles seen with ``strategy``, ``ratio`` and ``value``.
        floor: Lowest value drawn; a curve that falls below it is drawn at it and its label says where it goes.

    Returns:
        The figure.
    """
    shown = frame.dropna(subset=["value"]).assign(name=lambda rows: rows["strategy"].map(STRATEGIES))
    shown = shown.assign(drawn=shown["value"].clip(lower=floor))
    look = {
        "Partial pooling": (BLUE, 1.5, "solid"),
        "Complete pooling": (SLATE, 1.0, "solid"),
        "No pooling": (SLATE, 1.0, "dashed"),
        "Complete pooling, no diagnostics": (GREY, 1.4, "dotted"),
    }
    ends = shown.sort_values("ratio").groupby("name").tail(1)
    y = spread([max(float(v), floor) for v in ends["drawn"]], 0.14)
    names = ends.assign(
        x=ends["ratio"] + 0.02,
        y=y,
        label=ends["name"].map(lambda n: n.replace("Complete pooling, no diagnostics", "Control, no diagnostics")),
    )
    plot = ggplot(shown, aes("ratio", "drawn", group="name", color="name", linetype="name")) + geom_hline(
        yintercept=0.0, color=SLATE, linetype="dotted", size=0.5
    )
    plot += geom_line(aes(size="name"))
    return (
        plot
        + scale_color_manual(values={name: look[name][0] for name in look})
        + scale_linetype_manual(values={name: look[name][2] for name in look})
        + scale_size_manual(values={name: look[name][1] for name in look})
        + label(names, INK, size=5.2)
        + scale_x_continuous(limits=[0.0, 1.0 + 0.34], breaks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        + scale_y_continuous(limits=[floor - 0.05, 1.0])
        + labs(x="Cost-loss ratio of retiring a cell", y="Relative economic value")
        + site_theme()
        + canvas(440, 300)
    )


def loo(  # noqa: ANN201
    frame: pd.DataFrame,
    colours: dict[str, str],
    chemistry_of: dict[str, str],
    method: str,
    reference: str,
    size: tuple[int, int] = (440, 320),
):
    """Gain in expected log predictive density of one model over a reference, for each cell.

    Each bar is the sum, over the observations of one cell, of the leave-one-out score of ``method`` minus that of
    ``reference``. A bar to the left of zero means the model predicts the left-out cycles of that cell worse than the
    reference does; the total of the bars is the difference that ``arviz.compare`` reports.

    Args:
        frame: Rows from ``hierarchical.loo_by_cell`` with ``method``, ``cell`` and ``elpd``.
        colours: Colour of each chemistry.
        chemistry_of: Chemistry of each cell.
        method: The model to draw.
        reference: The model it is measured against.
        size: Width and height of the figure in pixels.

    Returns:
        The figure.
    """
    wide = frame.pivot(index="cell", columns="method", values="elpd")
    rows = (wide[method] - wide[reference]).rename("gain").reset_index()
    rows["chemistry"] = rows["cell"].map(chemistry_of)
    order = {name: index for index, name in enumerate(colours)}
    rows = rows.assign(rank=rows["chemistry"].map(order)).sort_values(["rank", "gain"], ascending=[True, False])
    rows["zero"] = 0.0
    rows["label"] = rows["gain"].map("{:+,.0f}".format)
    low, high = float(rows["gain"].min()), float(rows["gain"].max())
    span = max(high, 0.0) - min(low, 0.0)
    plot = (
        ggplot(rows)
        + geom_vline(xintercept=0, color=SLATE, size=0.5)
        + geom_segment(aes(x="zero", xend="gain", y="cell", yend="cell", color="chemistry"), size=6)
    )
    for side, hjust, keep in ((1.0, 0.0, rows["gain"] >= 0), (-1.0, 1.0, rows["gain"] < 0)):
        if keep.any():  # a text layer without rows cannot be drawn
            plot += geom_text(
                aes("gain", "cell", label="label"),
                size=5.2,
                color=SLATE,
                hjust=hjust,
                nudge_x=side * 0.03 * span,
                data=rows[keep],
            )
    return (
        plot
        + scale_color_manual(values=colours)
        + scale_y_discrete(limits=list(rows["cell"])[::-1], expand=[0.0, 0.8])
        + scale_x_continuous(limits=[min(low, 0.0) - 0.22 * span, max(high, 0.0) + 0.22 * span])
        + labs(x="Gain in expected log density over " + reference.lower(), y="")
        + site_theme()
        + theme(legend_position="none", axis_line_y=element_blank(), axis_ticks_y=element_blank())
        + canvas(*size)
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
