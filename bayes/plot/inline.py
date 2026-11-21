"""Inline SVG output of Lets-Plot figures, for pages that must not load a script."""

from __future__ import annotations

import re

from IPython.display import HTML, display


def inline_svg(plot) -> str:  # noqa: ANN001
    """Render a Lets-Plot figure as an SVG that scales with the width of its container.

    Args:
        plot: A Lets-Plot figure.

    Returns:
        An HTML string with the inline SVG inside a ``letsplot`` container; the SVG has a ``viewBox`` and fills the
        width of its container.

    Raises:
        ValueError: If the SVG has no width and height to turn into a ``viewBox``.
    """
    svg = plot.to_svg()
    match = re.search(r'width="([\d.]+)" height="([\d.]+)"', svg)
    if match is None:
        raise ValueError("the SVG of the figure has no width and height")
    width, height = (float(value) for value in match.groups())
    svg = svg.replace(match.group(0), f'viewBox="0 0 {width:g} {height:g}" width="100%"', 1)
    return f'<div class="letsplot">{svg}</div>'


def show(plot) -> None:  # noqa: ANN001
    """Display a Lets-Plot figure as inline SVG.

    Args:
        plot: A Lets-Plot figure.
    """
    display(HTML(inline_svg(plot)))
