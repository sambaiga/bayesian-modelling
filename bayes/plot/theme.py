from contextlib import contextmanager
import math
from typing import Literal

from lets_plot import element_blank, element_line, element_rect, element_text, ggsize, margin, theme

pro_colors_old = [
    "#0078D4",  # Bright blue (primary accent)
    "#00B294",  # Teal (success/accent)
    "#FF8C00",  # Orange (highlight/accent)
    "#D83B01",  # Red-orange (alert/accent)
    "#5C2D91",  # Purple (secondary accent)
    "#107C10",  # Green (confirmation)
    "#605E5C",  # Neutral gray (text or border)
    "#E1DFDD",  # Light gray (background grid)
]

pro_colors = [
    "#0078D4",  # blue
    "#00B294",  # teal
    "#E69F00",  # ColorBrewer orange (much more distinct from red)
    "#D55E00",  # ColorBrewer vermilion
    "#CC79A7",  # muted pink-purple
    "#009E73",  # better green
    "#605E5C",
    "#E1DFDD",
]

tableau10 = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # lime
    "#17becf",  # cyan
]


def modern_theme(
    show_x_axis: bool = True,
    font_size: int = 10,
    line_width: float = 1.0,
    x_axis_angle: int = 0,
    legend_pos: Literal["top", "bottom", "left", "right"] = "top",
) -> theme:
    """Create a custom theme for Lets-Plot visualizations.

    Args:
        show_x_axis: Whether to display the x-axis elements.
        font_size: Base font size for text elements.
        line_width: Base line width for line elements.
        x_axis_angle: Angle for x-axis text labels.
        legend_pos: Position of the legend ("top", "bottom", "left", "right").

    Returns:
        A Lets-Plot theme object with customized styles.
    """
    font_family = "Inter, 'Segoe UI', Roboto, 'Helvetica Neue', Helvetica, Arial, sans-serif"
    title_size = int(font_size * 1.6)
    subtitle_size = int(font_size * 1.25)
    axis_title_size = int(font_size * 1.15)

    base_theme = theme(
        legend_position=legend_pos if legend_pos != "none" else "none",
        legend_background=element_blank(),
        legend_key=element_blank(),
        legend_spacing=5,
        plot_background=element_blank(),
        panel_background=element_blank(),
        panel_grid_minor=element_blank(),
        panel_grid_major_y=element_blank(),
        panel_grid_major_x=element_blank(),
        axis_line=element_line(
            size=line_width * 0.9,
            color="#555555",
        ),
        axis_ticks=element_line(
            size=line_width * 0.9,
            color="#555555",
        ),
        # Global text base
        text=element_text(
            family=font_family,
            size=font_size,
            color="#333333",
        ),
        axis_text_x=element_text(
            angle=x_axis_angle,
            hjust=0.5,
            margin=[10, 0, 0, 0],
        ),
        axis_text_y=element_text(
            margin=[0, 0, 0, 10],
        ),
        axis_title_x=element_text(
            size=axis_title_size,
            face="bold",
            margin=[10, 0, 0, 0],
        ),
        axis_title_y=element_text(
            size=axis_title_size,
            face="bold",
            margin=[0, 10, 0, 0],
        ),
        plot_title=element_text(
            size=title_size,
            face="bold",
            hjust=0.5,
            margin=[0, 0, 12, 0],
        ),
        plot_subtitle=element_text(
            size=subtitle_size,
            hjust=0.5,
            margin=[0, 0, 20, 0],
        ),
        plot_caption=element_text(
            size=font_size * 0.9,
            hjust=1,
            color="#666666",
            margin=[15, 0, 0, 0],
        ),
        legend_title=element_text(
            size=axis_title_size,
            face="bold",
        ),
        legend_text=element_text(
            size=font_size,
        ),
    )

    if not show_x_axis:
        base_theme += theme(
            axis_title_x=element_blank(),
            axis_text_x=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line_x=element_blank(),
        )

    return base_theme


# Colours by role: subject blue, highlight orange, reference slate, context grey.
BLUE, ORANGE, SLATE, INK, GREY = "#0369A1", "#E26029", "#64748B", "#1E293B", "#CED4DA"
SITE_FONT = "'Libre Franklin', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def site_theme(font_size: float = 11) -> theme:
    """Theme of the figures of the blog: thin dark axes on the left and bottom, no grid, no legend, Libre Franklin.

    Args:
        font_size: Size of the axis and strip text.

    Returns:
        A Lets-Plot theme with a white plot background, so that a figure reads in a dark page as well.
    """
    return theme(
        legend_position="none",
        plot_background=element_rect(fill="white", color="white"),
        panel_background=element_blank(),
        panel_border=element_blank(),
        panel_grid=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(face="bold", size=font_size + 1, hjust=0),
        axis_line=element_line(size=0.5, color=INK),
        axis_ticks=element_line(size=0.5, color=INK),
        axis_ticks_length=4,
        text=element_text(family=SITE_FONT, size=font_size, color=INK),
        plot_margin=[8, 18, 4, 4],
    )


PHONE_WIDTH = 420
_LAYOUT = {"phone": False}


@contextmanager
def phone_layout():  # noqa: ANN201
    """Build figures for a phone inside the block: narrow, with the panels stacked.

    A figure is drawn twice, once for the page and once for a phone, and the page shows the one that fits. A narrow
    canvas keeps the text at a readable size when the figure is scaled to the width of the screen.
    """
    _LAYOUT["phone"] = True
    try:
        yield
    finally:
        _LAYOUT["phone"] = False


def columns(desktop: int, phone: int = 1) -> int:
    """Panels in a row: ``desktop`` normally, ``phone`` inside :func:`phone_layout`.

    Args:
        desktop: Panels in a row on a wide screen.
        phone: Panels in a row on a phone.

    Returns:
        The number of columns of panels to draw.
    """
    return phone if _LAYOUT["phone"] else desktop


def canvas(width: int, height: int, panels: int = 1, ncol: int = 1, phone_ncol: int = 1):  # noqa: ANN201
    """The size of a figure: as given, or narrow with the panels re-stacked inside :func:`phone_layout`.

    On a phone the canvas is ``PHONE_WIDTH`` wide and each panel keeps its aspect ratio, a little taller so that its
    text stays readable.

    Args:
        width: Width in pixels on a wide screen.
        height: Height in pixels on a wide screen, with about 60 pixels of margin and axis.
        panels: Number of panels.
        ncol: Columns of panels on a wide screen.
        phone_ncol: Columns of panels on a phone.

    Returns:
        A Lets-Plot size.
    """
    if not _LAYOUT["phone"]:
        return ggsize(width, height)
    rows_wide = math.ceil(panels / ncol)
    rows_phone = math.ceil(panels / phone_ncol)
    per_panel = (height - 60) / rows_wide
    ratio = (PHONE_WIDTH / phone_ncol) / (width / ncol)
    return ggsize(PHONE_WIDTH, int(60 + rows_phone * per_panel * ratio * 1.15))
