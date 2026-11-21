"""Loading and splitting of the CALCE cycling features used by the hierarchical battery study."""

from __future__ import annotations

from pathlib import Path
import urllib.request

import pandas as pd

from bayes.regression.hierarchical import state_of_health

FIGSHARE_URL = "https://ndownloader.figshare.com/files/59415941"


def load(path: Path, features: list[str], every: int = 8, url: str = FIGSHARE_URL) -> pd.DataFrame:
    """Read the cycle features of the cells, download them when absent, and keep every ``every``-th cycle of each cell.

    Neighbouring cycles of one cell are almost duplicates, so thinning loses little and shortens the sampling.

    Args:
        path: File of the features; downloaded from ``url`` when it does not exist.
        features: Diagnostic columns to keep.
        every: Keep one cycle in ``every`` for each cell.
        url: Source of the file.

    Returns:
        Rows with ``cell`` (the identifier without its prefix), ``CellType``, ``cycle``, ``soh``, ``y`` (the scaled
        state of health) and the features, sorted by cell and cycle.

    Raises:
        ValueError: If ``every`` is less than one.
    """
    if every < 1:
        raise ValueError("every must be at least one")
    if not path.exists():
        urllib.request.urlretrieve(url, path)  # noqa: S310
    raw = pd.read_parquet(path, columns=["cycle", "BatteryID", "CellType", "soh", *features])
    raw["cell"] = raw["BatteryID"].str.replace("CALCE_", "", regex=False)
    raw = raw.sort_values(["cell", "cycle"]).drop(columns="BatteryID")
    thinned = raw[raw.groupby("cell").cumcount() % every == 0].copy()
    thinned["y"] = state_of_health(thinned["soh"])
    return thinned.reset_index(drop=True)


def split(frame: pd.DataFrame, held_out: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate the cells kept back for testing from the training cells.

    Args:
        frame: Rows of all cells, with a ``cell`` column.
        held_out: Cells to keep back.

    Returns:
        The training rows and the held-out rows.

    Raises:
        ValueError: If a held-out cell is not in the frame.
    """
    missing = set(held_out) - set(frame["cell"])
    if missing:
        raise ValueError(f"cells not in the data: {sorted(missing)}")
    mask = frame["cell"].isin(held_out)
    return frame[~mask].reset_index(drop=True), frame[mask].reset_index(drop=True)
