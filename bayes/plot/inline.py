"""Inline SVG output of Lets-Plot figures, for pages that must not load a script."""

from __future__ import annotations

import io
import re

from IPython.display import HTML, display

from .theme import phone_layout


def inline_svg(plot, kind: str = "") -> str:  # noqa: ANN001
    """Render a Lets-Plot figure as an SVG that scales with the width of its container.

    Args:
        plot: A Lets-Plot figure.
        kind: ``"desktop"`` or ``"phone"`` for a figure that has both variants, otherwise empty.

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
    suffix = f" letsplot-{kind}" if kind else ""
    return f'<div class="letsplot{suffix}">{svg}</div>'


def show(figure) -> None:  # noqa: ANN001
    """Display a Lets-Plot figure as inline SVG, with a narrow variant for phones when given a builder.

    A figure that is passed as a function is built twice, once for the page and once inside ``phone_layout``, and the
    page shows the variant that fits the width of the screen. A figure that is passed as a plot is shown once.

    Args:
        figure: A Lets-Plot figure, a function without arguments that returns one, or a matplotlib concept diagram.
    """
    if hasattr(figure, "savefig"):  # a matplotlib concept diagram
        display(HTML(diagram_svg(figure)))
        return
    if not callable(figure):
        display(HTML(inline_svg(figure)))
        return
    wide = inline_svg(figure(), "desktop")
    with phone_layout():
        narrow = inline_svg(figure(), "phone")
    display(HTML(wide + narrow))


def diagram_svg(figure, max_width: int = 560) -> str:  # noqa: ANN001
    """Render a matplotlib concept diagram as an SVG that scales with the width of its container.

    Args:
        figure: A matplotlib figure from ``bayes.plot.concepts``.
        max_width: Widest display of the diagram in pixels; it is centred and narrower on a phone.

    Returns:
        An HTML string with the inline SVG in a centred box; the SVG has a ``viewBox`` and fills the box.

    Raises:
        ValueError: If the SVG has no width and height to turn into a ``viewBox``.
    """
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg")
    svg = buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    match = re.search(r'width="([\d.]+)pt" height="([\d.]+)pt"', svg)
    if match is None:
        raise ValueError("the SVG of the diagram has no width and height")
    width, height = (float(value) for value in match.groups())
    svg = svg.replace(match.group(0), f'viewBox="0 0 {width:g} {height:g}" width="100%"', 1)
    return f'<div class="concept" style="max-width: {max_width}px; margin: 0 auto">{svg}</div>'
