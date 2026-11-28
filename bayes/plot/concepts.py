"""Concept diagrams of the series, drawn with matplotlib and Bootstrap Icons in the manner of the book's diagrams.

Each function draws one idea: the three levels of pooling of the cells of a fleet, and the decision of an operator who
must retire a cell or leave it in service. A diagram is authored at its final size, in millimetres, so that its type is
never rescaled: the canvas is 120 mm wide and the page shows it at most 560 pixels wide. Colour follows the roles
of the site: blue for the first chemistry and for the subject, orange for the second chemistry and for what to notice,
slate for the fleet, the links and the text. Text and icons are written as outlines, so the SVG needs no font.
Every function returns a closed matplotlib figure; none reads a model or a file.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import matplotlib.pyplot as plt
import numpy as np

from .icons import ASSETS, icon, iconfont
from .theme import BLUE, INK, ORANGE, SLATE

ORANGE_TEXT = "#B3471B"  # orange darkened to read as text

mpl.rcParams.update({"svg.fonttype": "path", "mathtext.fontset": "cm"})

TEXT = FontProperties(fname=str(ASSETS / "LibreFranklin-Regular.ttf"))
MEDIUM = FontProperties(fname=str(ASSETS / "LibreFranklin-Medium.ttf"))
NOTE, LABEL = 9.0, 10.0  # points; the diagrams are 120 mm wide and shown smaller, so the type is a little large
WIDTH = 120.0
WALL, PALE_BLUE, HAIRLINE = "#F8FAFC", "#EAF3FA", "#CBD5E1"
WHITE = "#FFFFFF"
STROKE = 0.85


def tint(colour: str, amount: float) -> str:
    """Mix a colour with white; 0 keeps the colour and 1 gives white.

    Args:
        colour: Colour as ``#RRGGBB``.
        amount: Share of white.

    Returns:
        The mixed colour as ``#RRGGBB``.
    """
    rgb = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)])
    return "#" + "".join(f"{int(value):02X}" for value in rgb + (255 - rgb) * amount)


def _canvas(height: float) -> tuple[plt.Figure, plt.Axes]:
    """A figure ``WIDTH`` by ``height`` millimetres whose data coordinates are millimetres."""
    fig = plt.figure(figsize=(WIDTH / 25.4, height / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def _text(
    ax: plt.Axes, xy: tuple[float, float], text: str, colour: str = INK, size: float = NOTE, font: Any = TEXT, **kw: Any
) -> None:
    ax.text(*xy, text, fontproperties=font, fontsize=size, color=colour, **kw)


def _glyph(ax: plt.Axes, xy: tuple[float, float], name: str, size: float, colour: str, zorder: int = 6) -> None:
    ax.text(*xy, icon(name), fontproperties=iconfont(size), color=colour, ha="center", va="center", zorder=zorder)


def _card(
    ax: plt.Axes, left: float, bottom: float, width: float, height: float, face: str = WALL, edge: str = HAIRLINE
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (left, bottom),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=1.4",
            facecolor=face,
            edgecolor=edge,
            linewidth=STROKE * 0.7,
            zorder=1,
        )
    )


def _line(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    colour: str = SLATE,
    width: float = 0.5,
    dashed: bool = False,
) -> None:
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=colour,
        linewidth=STROKE * width,
        linestyle=(0, (3, 2)) if dashed else "solid",
        zorder=2,
        solid_capstyle="round",
    )


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], colour: str = SLATE) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=6,
            color=colour,
            linewidth=STROKE * 0.9,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def _badge(
    ax: plt.Axes,
    xy: tuple[float, float],
    name: str,
    colour: str,
    radius: float = 3.4,
    size: float = 8.5,
    ring: bool = False,
) -> None:
    """An icon in a round badge: filled in the colour, or a white badge with a ring."""
    ax.add_patch(
        Circle(xy, radius, facecolor=WHITE if ring else colour, edgecolor=colour, linewidth=STROKE * 1.2, zorder=5)
    )
    _glyph(ax, xy, name, size, colour if ring else WHITE)


SQUEEZE = 0.7  # the fan from the top card to the cells is drawn at this share of its first height, for a phone


def up(y: float) -> float:
    """A height above the row of cells (20 mm), compressed by ``SQUEEZE`` so that a panel is about as tall as wide."""
    return 20.0 + (y - 20.0) * SQUEEZE


TOP = up(62.0)  # the bottom of the top card
PANEL, PANEL_HEIGHT = 66.0, 66.0  # the three pooling panels are shown side by side and stacked on a phone
SPACING, GAP = 4.5, 4.0


def _layout(chemistry_of: dict[str, str], held_out: list[str]) -> list[dict[str, Any]]:
    """The 13 cells along the bottom: chemistry by chemistry, with a gap between them and the held-out cell last."""
    names = list(dict.fromkeys(chemistry_of.values()))
    cells: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        members = [cell for cell, chemistry in chemistry_of.items() if chemistry == name]
        members.sort(key=lambda cell: cell in held_out)
        cells.extend({"cell": cell, "chemistry": name, "held": cell in held_out, "group": index} for cell in members)
    total = len(cells) * SPACING + (len(names) - 1) * GAP
    start = (PANEL - total) / 2 + SPACING / 2
    for position, cell in enumerate(cells):
        cell["x"] = start + position * SPACING + cell["group"] * GAP
    return cells


def _batteries(ax: plt.Axes, cells: list[dict[str, Any]], colours: dict[str, str], y: float) -> None:
    """The cells as battery icons in the colour of their chemistry; a held-out cell has a dashed ring."""
    first_held = True
    for cell in cells:
        colour = colours[cell["chemistry"]]
        _glyph(ax, (cell["x"], y), "battery-half", 9.0, colour)
        if cell["held"]:
            ax.add_patch(
                Circle(
                    (cell["x"], y),
                    2.7,
                    facecolor="none",
                    edgecolor=colour,
                    linewidth=STROKE * 0.9,
                    linestyle=(0, (2, 1.6)),
                    zorder=4,
                )
            )
            if first_held:
                _text(ax, (cell["x"], y - 5.2), "Held out", SLATE, NOTE - 1.6, ha="center", va="center")
                first_held = False


def _note(ax: plt.Axes, text: str) -> None:
    """Two short lines under a panel."""
    _text(ax, (PANEL / 2, 4.0), text, SLATE, NOTE - 1.4, ha="center", va="center", linespacing=1.4)


def _panel(height: float = PANEL_HEIGHT) -> tuple[plt.Figure, plt.Axes]:
    """A pooling panel of ``PANEL`` by ``height`` millimetres in data coordinates of millimetres."""
    fig = plt.figure(figsize=(PANEL / 25.4, height / 25.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, PANEL)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def complete(chemistry_of: dict[str, str], held_out: list[str], colours: dict[str, str]) -> plt.Figure:
    """Complete pooling: one curve for the whole fleet, so every cell, and any new cell, receives it.

    Args:
        chemistry_of: Chemistry of each cell, in the order to be drawn.
        held_out: Cells kept back for testing, drawn with a dashed ring.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    fig, ax = _panel()
    cells = _layout(chemistry_of, held_out)
    middle = PANEL / 2
    for cell in cells:
        _line(ax, (middle, TOP), (cell["x"], 20.0), width=0.45)
    _card(ax, middle - 20, TOP, 40.0, 13.0, face=PALE_BLUE, edge=tint(BLUE, 0.55))
    _glyph(ax, (middle - 14.5, TOP + 6.5), "boxes", 10.0, SLATE)
    _glyph(ax, (middle - 8.0, TOP + 6.5), "graph-down", 10.0, BLUE)
    _text(ax, (middle - 3.5, TOP + 6.5), "One curve", INK, NOTE - 1.0, MEDIUM, ha="left", va="center")
    _batteries(ax, cells, colours, 15.0)
    _note(ax, "The same curve for every cell,\nand for a new cell")
    return plt.close(fig) or fig


def none(chemistry_of: dict[str, str], held_out: list[str], colours: dict[str, str]) -> plt.Figure:
    """No pooling: every cell has a curve of its own, and a new cell has nothing but the prior.

    Args:
        chemistry_of: Chemistry of each cell, in the order to be drawn.
        held_out: Cells kept back for testing, drawn with a dashed ring.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    fig, ax = _panel()
    cells = _layout(chemistry_of, held_out)
    level, radius = up(55.0), 2.0
    for cell in cells:
        colour = colours[cell["chemistry"]]
        if cell["held"]:
            _badge(ax, (cell["x"], level), "question-lg", SLATE, radius=radius, size=6.0, ring=True)
            ax.add_patch(
                Circle(
                    (cell["x"], level),
                    radius,
                    facecolor="none",
                    edgecolor=SLATE,
                    linewidth=STROKE * 0.9,
                    linestyle=(0, (2, 1.6)),
                    zorder=6,
                )
            )
        else:
            _line(ax, (cell["x"], level - radius), (cell["x"], 20.0), width=0.45)
            _badge(ax, (cell["x"], level), "graph-down", colour, radius=radius, size=6.0)
    held = next(cell for cell in cells if cell["held"])
    _text(ax, (held["x"], level + 5.0), "No curve", SLATE, NOTE - 1.6, ha="center", va="center")
    _batteries(ax, cells, colours, 15.0)
    _note(ax, "A curve from each record;\na new cell has none")
    return plt.close(fig) or fig


def partial(chemistry_of: dict[str, str], held_out: list[str], colours: dict[str, str]) -> plt.Figure:
    """Partial pooling: the fleet sets what chemistries are like, and each chemistry what its cells are like.

    A held-out cell is joined to its chemistry by a dashed line: a new cell starts from the chemistry it belongs to.

    Args:
        chemistry_of: Chemistry of each cell, in the order to be drawn.
        held_out: Cells kept back for testing, drawn with a dashed ring.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    fig, ax = _panel()
    cells = _layout(chemistry_of, held_out)
    names = list(dict.fromkeys(chemistry_of.values()))
    centres = {name: float(np.mean([c["x"] for c in cells if c["chemistry"] == name])) for name in names}
    middle = PANEL / 2
    for name in names:
        colour = colours[name]
        _line(ax, (middle, TOP), (centres[name], up(45.5)), SLATE, 0.7)
        for cell in cells:
            if cell["chemistry"] == name:
                _line(
                    ax,
                    (centres[name], up(42.5)),
                    (cell["x"], 20.0),
                    colour if cell["held"] else SLATE,
                    0.7 if cell["held"] else 0.45,
                    dashed=cell["held"],
                )
    _card(ax, middle - 20, TOP, 40.0, 13.0, face=PALE_BLUE, edge=tint(BLUE, 0.55))
    _glyph(ax, (middle - 14.5, TOP + 6.5), "boxes", 10.0, SLATE)
    _text(ax, (middle - 9.0, TOP + 6.5), "Fleet", INK, NOTE - 1.0, MEDIUM, ha="left", va="center")
    for name in names:
        _badge(ax, (centres[name], up(44.0)), "layers-fill", colours[name], radius=3.0, size=7.5)
        side = -1 if name == names[0] else 1
        _text(
            ax,
            (centres[name] + side * 4.5, up(44.0)),
            name,
            INK,
            NOTE - 1.4,
            MEDIUM,
            ha="right" if side < 0 else "left",
            va="center",
        )
    _batteries(ax, cells, colours, 15.0)
    _note(ax, "Cells stay near their chemistry;\na new cell starts there")
    return plt.close(fig) or fig


def decision() -> plt.Figure:
    """The decision of an operator: three steps, then the probability axis split at the threshold ``alpha``.

    The cost ``C`` of retiring and the loss ``L`` of a cell left in service while below the threshold enter only through
    their ratio ``alpha``; retiring is the cheaper choice when the probability exceeds it.

    Returns:
        The figure.
    """
    fig, ax = _canvas(66.0)
    top, card_width, card_height = 58.0, 33.0, 21.0
    steps = [
        ("clipboard2-pulse", "Inspect", "measure the cell"),
        ("bar-chart-line", "Predict", "probability p that\nthe cell is below 80%"),
        ("signpost-split", "Compare", r"$p$ with $\alpha = C / L$"),
    ]
    gap = (WIDTH - 4.0 - 3 * card_width) / 2
    for index, (name, first, second) in enumerate(steps):
        left = 2.0 + index * (card_width + gap)
        _card(ax, left, top - card_height, card_width, card_height)
        _glyph(ax, (left + 6.5, top - 7.5), name, 14.0, BLUE)
        _text(ax, (left + 12.5, top - 7.5), first, INK, NOTE, MEDIUM, ha="left", va="center")
        _text(ax, (left + 3.0, top - 15.0), second, SLATE, NOTE - 1.4, ha="left", va="center", linespacing=1.4)
        if index < 2:
            _arrow(
                ax,
                (left + card_width + 0.6, top - card_height / 2),
                (left + card_width + gap - 0.6, top - card_height / 2),
            )
    left, right, base, band = 6.0, 114.0, 12.0, 14.0
    cut = left + 0.32 * (right - left)  # an illustrative alpha
    ax.add_patch(Rectangle((left, base), cut - left, band, facecolor=tint(SLATE, 0.88), edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((cut, base), right - cut, band, facecolor=tint(BLUE, 0.84), edgecolor="none", zorder=1))
    _line(ax, (cut, base), (cut, base + band + 3.0), colour=ORANGE, width=1.6, dashed=True)
    _text(ax, (cut, base + band + 5.0), r"$\alpha = C / L$", ORANGE_TEXT, NOTE, MEDIUM, ha="center", va="center")
    _glyph(ax, (left + 5.0, base + band / 2), "exclamation-triangle", 12.0, SLATE)
    _text(ax, (left + 10.0, base + band / 2 + 2.6), "Leave in service", INK, NOTE - 0.6, MEDIUM, ha="left", va="center")
    _text(ax, (left + 10.0, base + band / 2 - 2.8), "risk the loss L", SLATE, NOTE - 1.4, ha="left", va="center")
    _glyph(ax, (cut + 6.0, base + band / 2), "arrow-repeat", 12.0, BLUE)
    _text(ax, (cut + 11.5, base + band / 2 + 2.6), "Retire the cell", INK, NOTE - 0.6, MEDIUM, ha="left", va="center")
    _text(ax, (cut + 11.5, base + band / 2 - 2.8), "pay the cost C", SLATE, NOTE - 1.4, ha="left", va="center")
    for position, label in ((left, "0"), (right, "1")):
        _text(ax, (position, base - 3.2), label, SLATE, NOTE - 1.0, ha="center", va="center")
    _text(
        ax,
        ((left + right) / 2, base - 3.2),
        "Probability p that the cell is below 80%",
        SLATE,
        NOTE - 1.0,
        ha="center",
        va="center",
    )
    return plt.close(fig) or fig
