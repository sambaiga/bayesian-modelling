"""Bootstrap Icons as matplotlib text glyphs, for the concept diagrams of the series.

Bootstrap Icons (https://icons.getbootstrap.com/, MIT licensed) is the icon set of the site and of the book. The font is
vendored in ``assets/bootstrap-icons.ttf`` (see ``assets/BOOTSTRAP_ICONS_NOTICE.txt``) and a glyph is looked up by the
name of the icon through the character map of the font, so no table of codepoints is kept by hand.
"""

from __future__ import annotations

from functools import cache
import logging
from pathlib import Path
import warnings

from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

__all__ = ["ASSETS", "FONT", "icon", "iconfont"]

ASSETS = Path(__file__).parent / "assets"
FONT = ASSETS / "bootstrap-icons.ttf"

font_manager.fontManager.addfont(str(FONT))

# matplotlib measures the line height of every text with the letters "lp", which an icon font does not have. The drawing
# does not use those letters, so the warning that it raises for each icon is silenced, and only this message.
warnings.filterwarnings("ignore", message=r"Glyph \d+ \(.*\) missing from font\(s\) bootstrap-icons")


@cache
def _codepoints() -> dict[str, int]:
    """Codepoint of each icon of the font, by name."""
    from fontTools.ttLib import TTFont

    logging.getLogger("fontTools").setLevel(logging.ERROR)  # the font has two padding bytes that fontTools reports
    with TTFont(str(FONT)) as font:
        cmap = font.getBestCmap()
    if cmap is None:
        raise RuntimeError(f"{FONT.name} has no character map")
    return {name: code for code, name in cmap.items()}


def icon(name: str) -> str:
    """The character that draws the Bootstrap icon ``name`` in :func:`iconfont`.

    Args:
        name: Icon name as on icons.getbootstrap.com, for example ``"battery-half"``.

    Returns:
        A one-character string.

    Raises:
        KeyError: If the font has no icon of that name.

    Examples:
        >>> len(icon("battery-half"))
        1
    """
    try:
        return chr(_codepoints()[name])
    except KeyError:
        raise KeyError(f"no Bootstrap icon named {name!r}") from None


def iconfont(size: float = 24.0) -> FontProperties:
    """Font properties that draw :func:`icon` glyphs at ``size`` points.

    Args:
        size: Size in points.

    Returns:
        The font properties.
    """
    return FontProperties(fname=str(FONT), size=size)
