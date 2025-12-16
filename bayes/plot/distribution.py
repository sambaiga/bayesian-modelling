from __future__ import annotations

from typing import Literal

from lets_plot import (
    LetsPlot,
    aes,
    flavor_high_contrast_dark,
    flavor_solarized_light,
    geom_density,
    ggplot,
    ggsize,
    guide_legend,
    guides,
    labs,
    layer_tooltips,
    scale_color_manual,
    scale_fill_manual,
)
import pandas as pd

from .theme import modern_theme, pro_colors


def plot_density(
    df: pd.DataFrame,
    *,
    title: str = "Density Plot",
    subtitle: str | None = None,
    n_samples: int | None = None,
    x_col: str = "value",
    color_col: str = "distribution",
    x_label: str = "Value",
    y_label: str = "Density",
    alpha: float = 0.25,
    line_size: float = 1.2,
    adjust: float = 1.0,
    caption: str = "Bayes Modelling | Anthony Faustine @2025",
    theme: Literal["dark", "light"] | None = "light",
    fig_size: tuple[int, int] = (300, 280),
) -> ggplot:
    """Create an overlaid kernel density plot for multiple distributions from a long-format DataFrame.

    The function produces professional-looking density plots with filled areas, colored outlines,
    interactive tooltips, and a clean modern theme. It is ideal for comparing empirical density
    estimates across groups (e.g., different priors, posteriors, or sampled distributions).

    Args:
        df: DataFrame with at least two columns:
            - 'value': numeric samples
            - 'distribution': categorical labels for each group
        title: Main title of the plot.
        subtitle: Optional subtitle. If None, a default subtitle is generated showing the number
            of samples per distribution (when uniform).
        n_samples: Number of samples per distribution. Used in the default subtitle if provided.
            If None, the function attempts to infer it.
        x_col: column to be used along x-axis
        color_col: categorical column for colors
        x_label: Label for the x-axis.
        y_label: Label for the y-axis.
        colors: List of hex color strings matching the order of unique values in
            ``df['distribution']``. If None, a colorblind-friendly palette is used.
        alpha: Transparency of the filled areas (0 = fully transparent, 1 = opaque).
        line_size: Thickness of the density outline lines.
        adjust: Bandwidth adjustment for the kernel density estimate.
            Values >1 increase smoothing, <1 decrease smoothing.
        caption: Plot caption (e.g., author or tool credit).
        fig_size: Figure size as (width, height) in pixels.
        theme: Literal["dark", "light"] | None = "dark",
            - "dark": high-contrast dark background (default)
            - "light": clean light background
            - None: plain modern theme without additional flavor

    Returns:
        A Lets-Plot ``ggplot`` object ready for display or saving.

    Examples:
        >>> p = plot_density(
        ...     df=df,
        ...     title="Gamma Distribution Priors",
        ...     n_samples=1000,
        ...     caption="Bayes|Anthony @2025",
        ...     alpha=0.3,
        ...     adjust=1.2,
        ... )
        >>> p  # displays the plot in Jupyter
    """
    required_cols = {x_col, color_col}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"The following required column(s) are missing: {sorted(missing_cols)}. "
            f"Available columns: {sorted(df.columns)}"
        )
    distributions = df[color_col].unique()
    n_dist = len(distributions)
    colors = pro_colors[:n_dist]

    default_subtitle = "Empirical Density Estimates"

    p = (
        ggplot(df, aes(x=x_col, color=color_col))
        + geom_density(
            fill="white",
            size=line_size,
            alpha=alpha,
            adjust=adjust,
            tooltips=layer_tooltips().disable_splitting(),
        )
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
        + labs(
            title=title,
            subtitle=subtitle or default_subtitle,
            caption=caption,
            x=x_label,
            y=y_label,
            color="Distribution",
            fill="Distribution",
        )
        + modern_theme()
        + guides(fill=guide_legend(nrow=1), color=guide_legend(nrow=1))
        + ggsize(*fig_size)
    )
    if theme == "dark":
        p += flavor_high_contrast_dark()
    elif theme == "light":
        p += flavor_solarized_light()
    return p
