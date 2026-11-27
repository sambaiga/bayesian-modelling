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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
import matplotlib.pyplot as plt
import numpy as np

from .icons import ASSETS, icon, iconfont
from .theme import BLUE, INK, ORANGE, SLATE

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


def _layout(chemistry_of: dict[str, str], held_out: list[str]) -> list[dict[str, Any]]:
    """The 13 cells along the bottom: chemistry by chemistry, with a gap between them and the held-out cell last."""
    names = list(dict.fromkeys(chemistry_of.values()))
    cells: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        members = [cell for cell, chemistry in chemistry_of.items() if chemistry == name]
        members.sort(key=lambda cell: cell in held_out)
        cells.extend({"cell": cell, "chemistry": name, "held": cell in held_out, "group": index} for cell in members)
    spacing = 7.4
    total = len(cells) * spacing + (len(names) - 1) * 6.0
    start = (WIDTH - total) / 2 + spacing / 2
    for position, cell in enumerate(cells):
        cell["x"] = start + position * spacing + cell["group"] * 6.0
    return cells


def _batteries(
    ax: plt.Axes, cells: list[dict[str, Any]], colours: dict[str, str], y: float, label: bool = True
) -> None:
    """The cells as battery icons in the colour of their chemistry; a held-out cell has a dashed ring."""
    first_held = True
    for cell in cells:
        colour = colours[cell["chemistry"]]
        _glyph(ax, (cell["x"], y), "battery-half", 17.0, colour)
        if cell["held"]:
            ax.add_patch(
                Circle(
                    (cell["x"], y),
                    4.4,
                    facecolor="none",
                    edgecolor=colour,
                    linewidth=STROKE * 1.0,
                    linestyle=(0, (2, 1.6)),
                    zorder=4,
                )
            )
            if label and first_held:
                _text(ax, (cell["x"], y - 7.0), "Held out", SLATE, NOTE - 0.6, ha="center", va="center")
                first_held = False


def complete(chemistry_of: dict[str, str], held_out: list[str], colours: dict[str, str]) -> plt.Figure:
    """Complete pooling: one curve for the whole fleet, so every cell, and any new cell, receives it.

    Args:
        chemistry_of: Chemistry of each cell, in the order to be drawn.
        held_out: Cells kept back for testing, drawn with a dashed ring.
        colours: Colour of each chemistry.

    Returns:
        The figure.
    """
    fig, ax = _canvas(66.0)
    cells = _layout(chemistry_of, held_out)
    centre = WIDTH / 2
    for cell in cells:
        _line(ax, (centre, 49.0), (cell["x"], 19.0))
    _card(ax, centre - 21, 49.0, 42.0, 12.0, face=PALE_BLUE, edge=tint(BLUE, 0.55))
    _glyph(ax, (centre - 15, 55.0), "boxes", 13.0, SLATE)
    _glyph(ax, (centre - 7.5, 55.0), "graph-down", 12.0, BLUE)
    _text(ax, (centre - 2.5, 55.0), "One curve", INK, NOTE, MEDIUM, ha="left", va="center")
    _batteries(ax, cells, colours, 14.0)
    _text(ax, (3, 1.8), "Every cell, and a new cell, receives the same curve", SLATE, NOTE - 0.4, va="center")
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
    fig, ax = _canvas(66.0)
    cells = _layout(chemistry_of, held_out)
    for cell in cells:
        colour = colours[cell["chemistry"]]
        if cell["held"]:
            _badge(ax, (cell["x"], 40.0), "question-lg", SLATE, ring=True)
            ax.add_patch(
                Circle(
                    (cell["x"], 40.0),
                    3.4,
                    facecolor="none",
                    edgecolor=SLATE,
                    linewidth=STROKE * 1.0,
                    linestyle=(0, (2, 1.6)),
                    zorder=6,
                )
            )
        else:
            _line(ax, (cell["x"], 36.6), (cell["x"], 19.0))
            _badge(ax, (cell["x"], 40.0), "graph-down", colour)
    _text(ax, (3, 59.0), "A curve for each cell, learned from its own record", INK, NOTE, MEDIUM, va="center")
    held = next(cell for cell in cells if cell["held"])
    _text(ax, (held["x"] - 1.5, 47.5), "No curve", SLATE, NOTE - 0.6, ha="center", va="center")
    _batteries(ax, cells, colours, 14.0)
    _text(ax, (3, 1.8), "A new cell has no curve, only the prior", SLATE, NOTE - 0.4, va="center")
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
    fig, ax = _canvas(84.0)
    cells = _layout(chemistry_of, held_out)
    names = list(dict.fromkeys(chemistry_of.values()))
    centres = {name: float(np.mean([c["x"] for c in cells if c["chemistry"] == name])) for name in names}
    middle = WIDTH / 2
    for name in names:
        colour = colours[name]
        _line(ax, (middle, 66.0), (centres[name], 50.0), SLATE, 0.8)
        for cell in cells:
            if cell["chemistry"] == name:
                _line(
                    ax,
                    (centres[name], 46.0),
                    (cell["x"], 19.0),
                    colour if cell["held"] else SLATE,
                    0.8 if cell["held"] else 0.5,
                    dashed=cell["held"],
                )
    _card(ax, middle - 25, 66.0, 50.0, 12.0, face=PALE_BLUE, edge=tint(BLUE, 0.55))
    _glyph(ax, (middle - 19, 72.0), "boxes", 13.0, SLATE)
    _text(ax, (middle - 13, 72.0), "Fleet: what chemistries are like", INK, NOTE - 0.4, MEDIUM, ha="left", va="center")
    for name in names:
        colour = colours[name]
        _badge(ax, (centres[name], 48.0), "layers-fill", colour, radius=4.2, size=10.0)
        side = -1 if name == names[0] else 1
        _text(
            ax,
            (centres[name] + side * 5.6, 48.0),
            f"{name}: level,\nrate, amplitude",
            INK,
            NOTE - 0.6,
            ha="right" if side < 0 else "left",
            va="center",
            linespacing=1.1,
        )
    _batteries(ax, cells, colours, 14.0)
    _text(
        ax, (3, 1.8), "Cells stay close to their chemistry, and a new cell starts there", SLATE, NOTE - 0.4, va="center"
    )
    return plt.close(fig) or fig


def decision() -> plt.Figure:
    """The decision of an operator: inspect a cell, read the probability that it is below 80%, retire it or leave it.

    The cost ``C`` of retiring and the loss ``L`` of a cell left in service while below the threshold enter only through
    their ratio ``alpha``; retiring is the cheaper choice when the probability exceeds it.

    Returns:
        The figure.
    """
    fig, ax = _canvas(84.0)
    y, half_width, half_height = 51.0, 14.0, 17.0
    steps = [("clipboard2-pulse", "Inspect", "the cell", 2.0), ("bar-chart-line", "Model gives", "probability p", 29.0)]
    for name, first, second, left in steps:
        _card(ax, left, y - 14.0, 24.0, 28.0)
        _glyph(ax, (left + 12.0, y + 6.0), name, 20.0, BLUE)
        _text(ax, (left + 12.0, y - 3.5), first, INK, NOTE - 0.6, MEDIUM, ha="center", va="center")
        _text(ax, (left + 12.0, y - 9.5), second, SLATE, NOTE - 1.2, ha="center", va="center")
    _arrow(ax, (26.5, y), (28.5, y))
    cx = 68.0  # the decision: a diamond
    _arrow(ax, (53.5, y), (cx - half_width - 0.3, y))
    ax.add_patch(
        Polygon(
            [(cx - half_width, y), (cx, y + half_height), (cx + half_width, y), (cx, y - half_height)],
            closed=True,
            facecolor=PALE_BLUE,
            edgecolor=tint(BLUE, 0.4),
            linewidth=STROKE * 0.9,
            zorder=2,
        )
    )
    _glyph(ax, (cx, y + 6.0), "signpost-split", 15.0, BLUE)
    _text(ax, (cx, y - 4.0), r"$p > \alpha$ ?", INK, NOTE, MEDIUM, ha="center", va="center")
    outcomes = [
        (y + 22.0, "arrow-repeat", BLUE, "Retire", "pay C", "yes"),
        (y - 22.0, "exclamation-triangle-fill", ORANGE, "Leave", "risk L", "no"),
    ]
    for level, name, colour, first, second, answer in outcomes:
        toward = 1.0 if level > y else -1.0
        _card(ax, 90.0, level - 9.5, 28.0, 19.0, face=tint(colour, 0.9), edge=tint(colour, 0.5))
        _glyph(ax, (96.0, level + 0.5), name, 15.0, colour)
        _text(ax, (101.5, level + 3.0), first, INK, NOTE - 0.6, MEDIUM, ha="left", va="center")
        _text(ax, (101.5, level - 3.5), second, SLATE, NOTE - 1.2, ha="left", va="center")
        _line(ax, (cx, y + toward * half_height), (cx, level), width=0.9)
        _arrow(ax, (cx, level), (89.7, level))
        _text(ax, (cx + 1.8, level - toward * 4.5), answer, colour, NOTE - 0.4, MEDIUM, ha="left", va="center")
    _card(ax, 2.0, 2.0, 116.0, 13.0, face=PALE_BLUE, edge=tint(BLUE, 0.5))
    _text(
        ax,
        (60.0, 8.5),
        r"Retire when $p > \alpha = C / L$, the cost over the loss of a cell below 80%",
        INK,
        NOTE - 0.8,
        ha="center",
        va="center",
    )
    return plt.close(fig) or fig
