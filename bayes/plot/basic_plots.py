from __future__ import annotations

from collections.abc import Sequence

from lets_plot import (
    aes,
    facet_wrap,
    geom_line,
    geom_point,
    geom_smooth,
    ggplot,
    ggsize,
    labs,
    layer_tooltips,
    scale_color_manual,
)
import numpy as np
import pandas as pd

from .theme import modern_theme, pro_colors


def line_plot(
    y: Sequence[float],
    x: Sequence[float] | None,
    *,
    title: str = "Charge Curve",
    y_label: str = "Voltage (V)",
    x_label: str | None,
    fig_size: tuple[int, int] = (300, 280),
    caption: str = "Bayes Modelling | Anthony Faustine @2025",
) -> ggplot:
    """Plot a clean charge/discharge curve with modern styling.

    Args:
        y: Y-values (voltage).
        x: Optional X-values (time or SOC). If None, uses normalized [0,1].
        title: Plot title.
        y_label: Y-axis label.
        x_label: X-axis label (e.g., "Time (h)" or "SOC").
        fig_size: Figure size as (width, height) in pixels.
        caption: Plot caption (e.g., author or tool credit).

    Returns:
        A styled Lets-Plot ggplot object.

    Examples:
        >>> p = plot_charge_curve(voltage, time, title="CC-CV Charge", x_label="Time (min)")
        >>> p
    """
    y_arr = np.asarray(y)
    if x is None:
        x_arr = np.linspace(0, 1, len(y_arr))
    else:
        x_arr = np.asarray(x)
        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same length")

    p = (
        ggplot()
        + geom_line(
            aes(x=x_arr, y=y_arr),
            color="#1f77b4",
            size=1.2,
            stat="identity",
            sampling=None,
            tooltips=layer_tooltips().disable_splitting(),
        )
        + labs(x=x_label or "Normalized Time", y=y_label, title=title, caption=caption)
        + modern_theme()
        + scale_color_manual(values=pro_colors)
        + ggsize(*fig_size)
    )

    return p


def scatter_plot(
    data: pd.DataFrame,
    x_col: str = "cycle",
    y_col: str = "Vc(Qc)_Mean",
    color_col: str = "failure_status",
    caption: str = "Bayes Modelling | Anthony Faustine @2025",
) -> ggplot:
    """Create a faceted scatter plot with LOESS trend per battery.

    Visualizes the relationship between two variables across multiple batteries,
    with color indicating failure status and a smoothed trend line.

    Args:
        data: Input DataFrame containing battery cycle data.
        x_col: Column name for the x-axis (e.g., cycle number).
            Defaults to "cycle".
        y_col: Column name for the y-axis (e.g., mean voltage at charge).
            Defaults to "Vc(Qc)_Mean".
        color_col: Column name for color grouping (e.g., failure status).
            Defaults to "failure_status".
        caption: Plot caption (e.g., author or tool credit).


    Returns:
        A Lets-Plot ggplot object with faceted scatter and LOESS smooth.
    """
    plot = (
        ggplot(data, aes(x=x_col, y=y_col, color=color_col))
        + geom_point(size=4)
        + geom_smooth(
            method="loess",
            size=1.0,
            color=pro_colors[2],
            seed=42,
        )
        + scale_color_manual(values=pro_colors)
        + facet_wrap("BatteryID", ncol=4)
        + modern_theme()
        + labs(
            title=f"{y_col} vs {x_col} per Battery",
            x=x_col,
            y=y_col,
            caption=caption,
            color="Status",
        )
    )
    return plot
