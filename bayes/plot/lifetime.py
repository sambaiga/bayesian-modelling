"""Figures of the machine-lifetime study: one message each, direct labels and the colours of the blog.

Every function takes tidy data frames and returns a Lets-Plot figure; none fits a model. The quantities that feed them
come from ``bayes.lifetime``.
"""

from __future__ import annotations

from lets_plot import (
    aes,
    element_text,
    facet_wrap,
    geom_line,
    geom_point,
    geom_segment,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_x_continuous,
    scale_y_continuous,
    scale_y_discrete,
    theme,
)
import numpy as np
import pandas as pd

from .hierarchy import label, spread, tint
from .theme import BLUE, INK, ORANGE, SLATE, canvas, columns, site_theme


def rates(grid: np.ndarray, prior: np.ndarray, likelihood: np.ndarray, posterior: np.ndarray):  # noqa: ANN201
    """Prior, likelihood and posterior density of the failure rate on one grid.

    Args:
        grid: Failure rates, per machine-year.
        prior: Prior density on the grid.
        likelihood: Likelihood on the grid, scaled to the height of the posterior so that it can share the axis.
        posterior: Posterior density on the grid.

    Returns:
        The figure.
    """
    frame = pd.DataFrame(
        {
            "rate": np.tile(grid, 3),
            "density": np.concatenate([prior, likelihood, posterior]),
            "curve": np.repeat(["Prior", "Likelihood", "Posterior"], len(grid)),
        }
    )
    colours = {"Prior": SLATE, "Likelihood": SLATE, "Posterior": BLUE}
    tops = frame.loc[frame.groupby("curve")["density"].idxmax()].set_index("curve")
    step = 0.03 * float(grid.max())
    placed = {"Prior": (-step, 1.0), "Posterior": (step, 0.0), "Likelihood": (0.0, 0.5)}
    names = [
        label(
            pd.DataFrame(
                {"x": [tops.loc[curve, "rate"] + shift], "y": [tops.loc[curve, "density"] * 1.03], "label": [curve]}
            ),
            INK,
            hjust=hjust,
            size=5.4,
        )
        for curve, (shift, hjust) in placed.items()
    ]
    return (
        ggplot(frame, aes("rate", "density", color="curve"))
        + geom_line(aes(linetype="curve"), size=1.3)
        + scale_color_manual(values=colours)
        + scale_linetype_manual(values={"Prior": "solid", "Likelihood": "dashed", "Posterior": "solid"})
        + names[0]
        + names[1]
        + names[2]
        + labs(x="Failure rate (per machine-year)", y="Density")
        + site_theme()
        + canvas(720, 340)
    )


def prior_check(shares: np.ndarray, observed: int):  # noqa: ANN201
    """Prior predictive share of each number of failures in the pilot, with the observed count highlighted.

    Args:
        shares: Share of prior predictive draws for each count, from zero upwards.
        observed: The number of failures in the pilot.

    Returns:
        The figure.
    """
    frame = pd.DataFrame(
        {
            "failures": np.arange(len(shares)),
            "share": shares,
            "kind": np.where(np.arange(len(shares)) == observed, "Observed", "Other"),
        }
    )
    frame["bottom"] = 0.0
    pointer = pd.DataFrame(
        {"x": [float(observed)], "y": [float(shares[observed]) + 0.04], "label": [f"Observed: {observed}"]}
    )
    return (
        ggplot(frame, aes(x="failures"))
        + geom_segment(aes(xend="failures", y="bottom", yend="share", color="kind"), size=7)
        + scale_color_manual(values={"Observed": ORANGE, "Other": tint(SLATE, 0.45)})
        + label(pointer, INK, hjust=0.5, size=5.4)
        + scale_x_continuous(breaks=list(range(len(shares))))
        + labs(x="Failures in the pilot", y="Share of prior predictive draws")
        + site_theme()
        + canvas(720, 320)
    )


def traces(draws: np.ndarray):  # noqa: ANN201
    """The draws of each chain against the iteration, to see whether the chains mix.

    Args:
        draws: Array of shape (chains, draws).

    Returns:
        The figure.
    """
    chains, length = draws.shape
    frame = pd.DataFrame(
        {
            "iteration": np.tile(np.arange(length), chains),
            "rate": draws.ravel(),
            "chain": np.repeat([f"Chain {index + 1}" for index in range(chains)], length),
        }
    )
    return (
        ggplot(frame, aes("iteration", "rate", color="chain"))
        + geom_line(size=0.5, alpha=0.7)
        + scale_color_manual(values=dict.fromkeys(frame["chain"].unique(), BLUE))
        + labs(x="Iteration", y="Failure rate (per machine-year)")
        + site_theme()
        + canvas(720, 300)
    )


def survival_curves(frame: pd.DataFrame, horizon: float, subject: str):  # noqa: ANN201
    """Probability that a machine survives each time, for each prior, with the contract period marked.

    Args:
        frame: Rows with ``years``, ``survival`` and ``prior``.
        horizon: Length of the contract in years, drawn as a vertical reference.
        subject: Name of the prior that the post is about, drawn in blue; the others are slate, the third dashed.

    Returns:
        The figure.
    """
    names = list(dict.fromkeys(frame["prior"]))
    colours = {name: BLUE if name == subject else SLATE for name in names}
    dashes = {name: "solid" if index < 2 else "dashed" for index, name in enumerate(names)}
    ends = frame[frame["years"] == frame["years"].max()].reset_index(drop=True)
    ends = ends.assign(x=frame["years"].max() + 0.2, y=spread(ends["survival"].tolist(), 0.07), label=ends["prior"])
    return (
        ggplot(frame, aes("years", "survival", color="prior"))
        + geom_vline(xintercept=horizon, color=SLATE, linetype="dotted", size=0.5)
        + geom_line(aes(linetype="prior"), size=1.3)
        + scale_color_manual(values=colours)
        + scale_linetype_manual(values=dashes)
        + label(ends, INK, size=5.2)
        + label(
            pd.DataFrame({"x": [horizon + 0.15], "y": [0.99], "label": [f"Contract: {horizon:g} years"]}),
            SLATE,
            size=5.2,
        )
        + scale_x_continuous(limits=[0, frame["years"].max() + 3.4])
        + scale_y_continuous(limits=[0, 1])
        + labs(x="Years in service", y="Probability of surviving")
        + site_theme()
        + canvas(720, 340)
    )


def sensitivity(table: pd.DataFrame, price: float):  # noqa: ANN201
    """Expected repair cost and the probability that the contract pays off, for each prior.

    Args:
        table: Rows with ``prior``, ``cost`` (expected repair cost without a contract) and ``probability``.
        price: Price of the contract, drawn as a reference in the cost panel.

    Returns:
        The figure.
    """
    long = table.melt(id_vars="prior", value_vars=["cost", "probability"], var_name="measure")
    long["measure"] = long["measure"].map(
        {"cost": "(a) Expected cost of repairs", "probability": "(b) Probability that repairs cost more"}
    )
    long["zero"] = 0.0
    reference = pd.DataFrame({"measure": ["(a) Expected cost of repairs"], "x": [price]})
    order = list(table["prior"])
    return (
        ggplot(long, aes("value", "prior"))
        + geom_vline(aes(xintercept="x"), data=reference, color=SLATE, linetype="dotted", size=0.5, inherit_aes=False)
        + geom_segment(aes(x="zero", xend="value", yend="prior"), color=tint(BLUE, 0.4), size=1.2)
        + geom_point(color=BLUE, shape=21, fill="white", size=3.6, stroke=1.4)
        + facet_wrap("measure", ncol=columns(2), scales="free_x")
        + scale_y_discrete(limits=order[::-1])
        + labs(x="", y="")
        + site_theme()
        + theme(axis_text_y=element_text(size=11))
        + canvas(840, 280, 2, 2)
    )
