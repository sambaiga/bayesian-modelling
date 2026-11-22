"""The change chart: how much each method differs from one reference, on one metric.

This is the chart of the defence slides and the book for comparing many methods against a reference. Horizontal bars
start at zero, which is the reference and carries its name above it; the change is printed at the end of each bar, so
no axis has to be read; the rows keep one order in every panel. A bar to the right of zero is worse than the reference
for a metric where lower is better. The subject is blue, other methods that improve on the reference are slate, and
methods that are worse than the reference are orange.
"""

from __future__ import annotations

from lets_plot import (
    aes,
    facet_wrap,
    geom_segment,
    geom_text,
    ggplot,
    ggsize,
    labs,
    scale_color_manual,
    scale_x_continuous,
    scale_y_discrete,
)
import numpy as np
import pandas as pd

from .theme import BLUE, INK, ORANGE, SLATE, site_theme

MINUS = "−"


def percent(value: float, unit: str = "%") -> str:
    """A percentage change with a signed integer and the minus sign of typography.

    Args:
        value: The change.
        unit: Text after the number, ``"%"`` for a change in percent or `` pts`` for percentage points.

    Returns:
        ``"0%"`` for a change that rounds to zero, otherwise the sign and the rounded value, for example ``"+22%"``.
    """
    rounded = int(np.round(value))
    if rounded == 0:
        return f"0{unit}"
    return f"{rounded:+d}{unit}".replace("-", MINUS)


def roles(frame: pd.DataFrame, subject: str | None) -> pd.Series:
    """The role of each bar: worse than the reference, the subject, or another method that improves on it.

    Args:
        frame: Rows with ``name`` and ``value`` (the change in percent).
        subject: Name of the subject method, or ``None``.

    Returns:
        ``"worse"`` where the change is positive, ``"subject"`` for the subject, otherwise ``"other"``.
    """
    role = np.where(frame["value"] > 0.5, "worse", np.where(frame["name"] == subject, "subject", "other"))
    return pd.Series(role, index=frame.index)


def change(  # noqa: ANN201
    frame: pd.DataFrame, reference: str, order: list[str], subject: str | None = None, ncol: int = 3, unit: str = "%"
):
    """Bars of the change in percent of each method against the reference, one panel for each condition.

    Args:
        frame: Rows with ``panel`` (the title of the panel), ``name`` (the method) and ``value`` (the change in
            percent; a negative value is better).
        reference: Name of the reference, written above the zero line.
        order: Method names from the top row to the bottom row, the same in every panel.
        subject: Name of the method that the figure is about, drawn in blue when it improves on the reference.
        ncol: Panels in a row.
        unit: Text after each value and each tick, ``"%"`` or `` pts``. A ``role`` column in ``frame`` (``"worse"``,
            ``"subject"`` or ``"other"``) replaces the colouring by the sign of the change.

    Returns:
        The figure.
    """
    drawn = frame.assign(zero=0.0)
    if "role" not in drawn:
        drawn["role"] = roles(frame, subject)
    drawn["text"] = drawn["value"].map(lambda value: percent(value, unit))
    sign = np.where(drawn["value"] > 0.5, 1.0, -1.0)
    span = float(drawn["value"].abs().max())
    low, high = min(float(drawn["value"].min()), 0.0), max(float(drawn["value"].max()), 0.0)
    room = 0.3 * span
    left = room if low < -0.05 * span else 0.1 * span
    line = pd.DataFrame({"panel": sorted(drawn["panel"].unique()), "x": 0.0, "y": order[0], "yend": order[-1]})
    drawn["end"] = drawn["value"] + sign * 0.03 * span
    drawn["side"] = np.where(drawn["value"] > 0.5, 0.0, 1.0)
    rows = ["​", *order]
    top = pd.DataFrame({"panel": sorted(drawn["panel"].unique())}).assign(name="​", x=0.0, label=reference)
    return (
        ggplot(drawn, aes("value", "name", color="role"))
        + geom_segment(aes(x="x", xend="x", y="y", yend="yend"), data=line, color=INK, size=0.9, inherit_aes=False)
        + geom_segment(aes(x="zero", xend="value", y="name", yend="name"), size=11)
        + scale_color_manual(values={"subject": BLUE, "other": SLATE, "worse": ORANGE})
        + geom_text(aes("end", "name", label="text"), data=drawn[drawn["side"] == 0.0], hjust=0, size=5.6, color=INK)
        + geom_text(aes("end", "name", label="text"), data=drawn[drawn["side"] == 1.0], hjust=1, size=5.6, color=INK)
        + geom_text(aes("x", "name", label="label"), data=top, size=5.4, color=SLATE, hjust=0.5)
        + facet_wrap("panel", ncol=ncol)
        + scale_y_discrete(limits=rows[::-1])
        + scale_x_continuous(limits=[low - left, high + room], format="{.0f}" + unit)
        + labs(x="", y="")
        + site_theme()
        + ggsize(900, 60 + 40 * len(rows))
    )
