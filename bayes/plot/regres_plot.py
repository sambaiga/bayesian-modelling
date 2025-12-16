from __future__ import annotations

from lets_plot import (
    aes,
    facet_wrap,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    layer_tooltips,
    scale_color_manual,
    scale_fill_manual,
    scale_y_continuous,
)
import numpy as np
import pandas as pd

from .theme import modern_theme


def plot_hdi_regression(
    df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    pred_column: str = "pred_mean",
    hdi_low_column: str = "hdi_low",
    hdi_high_column: str = "hdi_high",
    group_column: str | None = None,
    x_label: str = "Independent Variable (X)",
    y_label: str = "Dependent Variable (Y)",
    title_prefix: str = "Posterior Predictive Regression",
    subtitle: str = "Prediction interval accounts for both parameter uncertainty and observation noise.",
    caption_text: str = "Bayesian Modelling| Anthony Faustine @ 2025",
    alpha: float = 0.05,
    steps: float | None = 0.4,
) -> ggplot:
    """General-purpose regression plot with observed data, posterior mean, and HDI ribbon.

    This function visualizes a regression line (posterior predictive mean) together with
    prediction uncertainty expressed as a Highest Density Interval (HDI) ribbon across
    the range of the independent variable. The plot can optionally be faceted by a
    grouping column (e.g., individual batteries or units).

    Args:
        df (pd.DataFrame): DataFrame containing the observed data, posterior predictive
            mean, and HDI boundaries.
        x_column (str): Name of the column representing the independent variable (X-axis).
        y_column (str): Name of the column representing the observed dependent variable
            (Y-axis).
        pred_column (str, optional): Name of the column containing the posterior predictive
            mean. Defaults to "pred_mean".
        hdi_low_column (str, optional): Name of the column containing the lower bound of
            the HDI. Defaults to "hdi_low".
        hdi_high_column (str, optional): Name of the column containing the upper bound of
            the HDI. Defaults to "hdi_high".
        group_column (str | None, optional): Name of the column used for faceting
            (e.g., "battery_id"). If provided, the plot is split into multiple panels.
            Defaults to None.
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        title_prefix (str): Main title prefix for the plot.
        subtitle (str): Subtitle for the plot, explaining the prediction interval.
        caption_text (str): Text displayed as the plot caption (bottom center).
        alpha (float, optional): Significance level for the HDI. The HDI displayed will be
            (1.0 - alpha) * 100%. Must be strictly between 0 and 1.
            Defaults to 0.05 (95% HDI).
        steps (float | None, optional): Step size for y-axis ticks. If None, automatic
            ticks are used.

    Returns:
        ggplot: A lets-plot ggplot object ready for display.

    Raises:
        ValueError: If `alpha` is not strictly between 0 and 1.
        KeyError: If required columns (`x_column`, `y_column`, etc.) are missing from the
            DataFrame.
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be strictly between 0 and 1.")

    required_cols = [x_column, y_column, pred_column, hdi_low_column, hdi_high_column]
    if group_column:
        required_cols.append(group_column)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {', '.join(missing_cols)}")

    hdi_prob = int((1.0 - alpha) * 100)
    hdi_label = f"{hdi_prob}% HDI"

    max_val = df[[y_column, hdi_high_column]].max().max()
    y_max = np.ceil(max_val / 0.2) * 0.2

    color_palette = {
        "Observed Data": "#00B294",
        "Posterior Mean": "#0078D4",
        hdi_label: "#FF8C00",
    }

    df_obs = df.assign(Layer="Observed Data")
    df_mean = df.assign(Layer="Posterior Mean")
    df_hdi = df.assign(Layer=hdi_label)

    plot = ggplot(data=df) + aes(x=x_column)

    plot += geom_ribbon(
        data=df_hdi,
        mapping=aes(
            ymin=hdi_low_column,
            ymax=hdi_high_column,
            fill="Layer",
        ),
        alpha=0.25,
        tooltips=layer_tooltips()
        .title(hdi_label)
        .line(f"lower | @{hdi_low_column}")
        .line(f"upper | @{hdi_high_column}"),
    )

    plot += geom_line(
        data=df_mean,
        mapping=aes(y=pred_column, color="Layer"),
        size=1.3,
        line_type="dashed",
        tooltips=layer_tooltips().title("Posterior Mean").line(f"value | @{pred_column}"),
    )

    plot += geom_point(
        data=df_obs,
        mapping=aes(y=y_column, color="Layer"),
        size=0.9,
        alpha=0.6,
        tooltips=layer_tooltips().title("Observed Data").line(f"value | @{y_column}"),
    )

    if group_column:
        plot += facet_wrap(group_column, ncol=2, scales="free_y")

    plot += scale_color_manual(
        values={
            "Observed Data": color_palette["Observed Data"],
            "Posterior Mean": color_palette["Posterior Mean"],
        },
        name="Legend",
        breaks=["Observed Data", "Posterior Mean"],
    )

    plot += scale_fill_manual(
        values={hdi_label: color_palette[hdi_label]},
        name="Legend",
        breaks=[hdi_label],
    )

    plot += labs(
        title=title_prefix,
        subtitle=subtitle,
        caption=caption_text,
        x=x_label,
        y=y_label,
    )

    if steps is not None:
        plot += scale_y_continuous(
            limits=(0.0, y_max),
            breaks=np.arange(0, y_max + 0.1, steps),
            expand=(0.02, 0.02),
        )

    plot += modern_theme(legend_pos="right")

    return plot
