import contextlib
import logging
from typing import Any, Literal

import arviz as az
from cycler import cycler
import matplotlib as mpl
from matplotlib_inline.backend_inline import set_matplotlib_formats

from bayes.utils.logging import log as logger


def golden_height(width):
    return width * 0.618


def configure_matlib_style(
    latex: bool = False,
    style: list[str] | None = None,
    font_size: int = 10,
    legend_position: Literal["top", "bottom", "left", "right", "none"] = "top",
    fig_width: float | None = None,
    fig_height: float | None = None,
    dpi: int = 480,
    color_cycle: list[str] | None = None,
    custom_rc: dict[str, Any] | None = None,
) -> None:
    """Configure scientific plotting styles with small fonts and publication-quality settings.

    Applies matplotlib configuration with optimized defaults for scientific plotting,
    including font sizes, figure dimensions, and color cycles. Supports LaTeX rendering
    and custom style specifications.

    Args:
        latex: Enable LaTeX text rendering. Defaults to False.
        style: Custom matplotlib style specifications. If None, uses ["science", "no-latex"]
            when latex is False, else ["science"]. Defaults to None.
        font_size: Base font size in points. Defaults to 10.
        small_font_size: Small font size for ticks and legends. Defaults to 8.
        fig_width: Figure width in inches. If None, uses matplotlib's default. Defaults to None.
        fig_height: Figure height in inches. If None, uses matplotlib's default. Defaults to None.
        dpi: Output resolution for saved figures. Defaults to 480.
        color_cycle: Custom color cycle using hex codes. If None, uses predefined colors.
            Defaults to None.
        custom_rc: Additional matplotlib rcParams to override defaults. Defaults to None.
        legend_position: position of the legend.

    Raises:
        ImportError: If required styles (e.g., science) are not available
        OSError: If LaTeX dependencies are missing when latex=True
    """
    # Default color cycle
    default_colors = [
        "#107591",
        "#00c0bf",
        "#f69a48",
        "#fdcd49",
        "#8da798",
        "#a19368",
        "#525252",
        "#a6761d",
        "#7035b7",
        "#cf166e",
    ]
    font_family = ["Inter", "Segoe UI", "Roboto", "Helvetica Neue", "Helvetica", "Arial", "sans-serif"]
    title_size = int(font_size * 1.5)
    small_font_size = int(font_size * 0.8)
    if fig_width is not None:
        fig_height = golden_height(fig_width) or fig_height
    # Base configuration
    plot_config = {
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "figure.figsize": (
            fig_width or mpl.rcParams["figure.figsize"][0],
            fig_height or mpl.rcParams["figure.figsize"][1],
        ),
        # Font settings
        "font.size": font_size,
        "font.family": font_family,
        "axes.labelsize": font_size,
        "axes.titlesize": title_size,
        "axes.edgecolor": "#555555",
        "axes.facecolor": "white",
        "axes.labelcolor": "#333333",
        "axes.titlecolor": "#333333",
        "xtick.labelsize": small_font_size,
        "ytick.labelsize": small_font_size,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": small_font_size,
        # Line and axis settings
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "lines.markeredgewidth": 0.8,
        # Layout and spacing
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.autolayout": True,
        # Legend customization
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.labelspacing": 0.3,
        # Color cycle
        "axes.prop_cycle": cycler(color=color_cycle or default_colors),
    }

    # LaTeX configuration
    if latex:
        plot_config.update(
            {"text.usetex": True, "font.family": "serif", "text.latex.preamble": r"\usepackage{amsmath,amssymb}"}
        )

    # Apply custom rcParams
    if custom_rc:
        plot_config.update(custom_rc)

    # Apply style and settings
    try:
        if style is None:
            style = ["science", "no-latex"] if not latex else ["science"]

        mpl.style.use(style)
        mpl.rcParams.update(plot_config)

        # Set output formats for IPython
        set_matplotlib_formats("svg", "pdf", "png")

    except (ImportError, OSError) as e:
        logger.info(f"Style configuration warning: {str(e)}")
        mpl.rcParams.update(plot_config)


# Context manager version for temporary styling
@contextlib.contextmanager
def plotting_context(**kwargs):
    original_rc = mpl.rcParams.copy()
    configure_matlib_style(**kwargs)
    try:
        yield
    finally:
        mpl.rcParams.update(original_rc)
