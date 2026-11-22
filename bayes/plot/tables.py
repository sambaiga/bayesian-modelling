"""Great-tables style of the book, for the tables that stay tables: a specification or a summary of estimates.

A table of metrics is drawn as a figure (``bayes.plot.tiles``); a table is kept when the reader needs the exact words
or symbols, such as the priors of a model. Every table goes through :func:`themed_gt`, which is the style of the umeme
book on the colours of the site.
"""

from __future__ import annotations

from great_tables import GT, loc, style
import pandas as pd

BLUE = "#0369A1"
BLUE_TINT = "#EAF3FA"
INK = "#1A202C"
MUTED = "#6C757D"
RULE = "#DEE2E6"
FONTS = ["Libre Franklin", "Segoe UI", "Roboto", "Helvetica Neue", "sans-serif"]
HEADING_FONT = "Jost"


def themed_gt(table: GT | pd.DataFrame, width: str = "100%", striped: bool = True) -> GT:
    """Apply the table style of the book to a ``GT`` object or a data frame.

    Call it last, after the header, the column labels and the source notes. A blue rule above and below the table, a
    blue header row with white bold text in the heading font, thin grey rules between rows and a muted tint on every
    second row, so that a table reads as part of the same page as the figures.

    Args:
        table: A ``GT`` object, or a data frame to wrap.
        width: CSS width of the table.
        striped: Whether to tint every second row; needs the number of rows, so it applies to a data frame only.

    Returns:
        The styled ``GT`` object.
    """
    rows = None
    if isinstance(table, pd.DataFrame):
        rows = len(table)
        table = GT(table)
    table = (
        table.tab_options(
            table_width=width,
            container_width=width,
            table_font_names=FONTS,
            table_font_size="13px",
            table_font_color=INK,
            heading_background_color="#FFFFFF",
            heading_title_font_size="16px",
            heading_title_font_weight="bold",
            heading_subtitle_font_size="13px",
            column_labels_background_color=BLUE,
            column_labels_font_weight="bold",
            column_labels_font_size="13px",
            column_labels_padding="9px",
            data_row_padding="7px",
            table_body_border_bottom_color=RULE,
            table_body_border_bottom_width="1px",
            table_body_border_bottom_style="solid",
            table_border_top_color=BLUE,
            table_border_top_width="2px",
            table_border_top_style="solid",
            table_border_bottom_color=BLUE,
            table_border_bottom_width="1px",
            table_border_bottom_style="solid",
            source_notes_font_size="11px",
        )
        .tab_style(style=style.text(color="#FFFFFF", font=HEADING_FONT), locations=loc.column_labels())
        .tab_style(style=style.text(color=MUTED, style="italic"), locations=loc.source_notes())
    )
    if striped and rows is not None and rows > 1:
        table = table.tab_style(style=style.fill(color=BLUE_TINT), locations=loc.body(rows=list(range(1, rows, 2))))
    return table
