"""Predict the end of life of the cells that are still running when a test is stopped, with two methods.

A manufacturer stops the test of a batch at a chosen cycle. The cells that have crossed the threshold have a known
end of life, and the cells that have not are the ones to predict. Two methods use what the test recorded up to the stop:

* the Weibull model of the lifetimes, which sees only the cycle of each crossing and the cells that have not crossed
  as censored records, and so knows nothing of a running cell beyond its chemistry and that it is still in service;
* the power-law path of the loss, which sees every measurement up to the stop and predicts the cycle at which the path
  of each running cell crosses.

Both give draws of the end of life of each running cell, conditioned on the cell being in service at the stop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bayes import degradation, survival
from bayes.regression import hierarchical as hier


def weibull_predictions(
    table: pd.DataFrame, stop: float, cache: Path, config: dict, seed: int = 0, draws: int = 4000
) -> pd.DataFrame:
    """Draws of the end of life of the running cells from the Weibull model fitted to the records at a stop.

    Args:
        table: The output of :func:`bayes.survival.end_of_life`.
        stop: The cycle at which the test stops.
        cache: Folder of the stored fits.
        config: ``draws``, ``tune`` and ``chains`` of the sampler.
        seed: Seed of the sampler and of the conditional draws.
        draws: Number of predictive draws for each cell.

    Returns:
        A frame with one column for each running cell and ``draws`` rows, all above ``stop``; empty of columns when no
        cell is running.
    """
    records = survival.truncate(table, stop)
    idata = hier.sample(
        survival.weibull_model(records), cache / f"weibull-{int(stop)}.nc", seed=seed + int(stop), **config
    )
    alpha = idata.posterior["alpha"].stack(s=("chain", "draw")).to_numpy()
    beta = idata.posterior["beta_chem"].stack(s=("chain", "draw"))
    generator = np.random.default_rng(seed)
    pick = generator.integers(0, alpha.size, size=draws)
    running = records.loc[~records["event"] & (records["eol"] > stop)]
    columns = {}
    for cell, chemistry in zip(running["cell"], running["CellType"], strict=True):
        scale = beta.sel(chemistry=chemistry).to_numpy()[pick] * survival.CYCLE_SCALE
        columns[cell] = survival.weibull_conditional_draws(alpha[pick], scale, stop, generator)
    return pd.DataFrame(columns)


def path_predictions(
    frame: pd.DataFrame, table: pd.DataFrame, stop: float, cache: Path, config: dict, seed: int = 0, loss: float = 20.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draws of the end of life of the running cells from the path model fitted to the measurements up to a stop.

    The draws at or below the stop are dropped, because the cell is known to have been in service then; the share
    dropped is reported, since a large share means that the path disagrees with what was seen.

    Args:
        frame: Rows with ``cell``, ``CellType``, ``cycle`` and ``soh`` for every cell.
        table: The output of :func:`bayes.survival.end_of_life`.
        stop: The cycle at which the test stops.
        cache: Folder of the stored fits.
        config: ``draws``, ``tune`` and ``chains`` of the sampler.
        seed: Seed of the sampler.
        loss: The loss in points of state of health that defines the end of life.

    Returns:
        The draws, one column for each running cell, and a frame with ``cell`` and ``dropped``, the share of draws at or
        below the stop.
    """
    seen = frame[frame["cycle"] <= stop]
    idata = hier.sample(degradation.path_model(seen), cache / f"path-{int(stop)}.nc", seed=seed + int(stop), **config)
    crossings = degradation.cell_crossings(idata, loss)
    running = table.loc[table["eol"] > stop, "cell"].tolist()
    columns, dropped = {}, []
    for cell in running:
        values = crossings[cell].to_numpy()
        kept = values[values > stop]
        dropped.append({"cell": cell, "dropped": 1.0 - kept.size / values.size})
        columns[cell] = kept
    longest = max((len(values) for values in columns.values()), default=0)
    padded = {
        cell: np.pad(values, (0, longest - len(values)), constant_values=np.nan) for cell, values in columns.items()
    }
    return pd.DataFrame(padded), pd.DataFrame(dropped, columns=["cell", "dropped"])


def score_predictions(
    predictions: pd.DataFrame, table: pd.DataFrame, level: float = 0.9, tau: float = 0.1
) -> pd.DataFrame:
    """Score the draws of each running cell against its observed end of life.

    Args:
        predictions: Draws, one column for each cell; ``NaN`` marks padding.
        table: The output of :func:`bayes.survival.end_of_life`.
        level: Coverage of the central interval.
        tau: Quantile of the rated life.

    Returns:
        One row for each cell with the chemistry, the observed ``eol`` and the scores of
        :func:`bayes.survival.interval_and_pinball`.
    """
    actual = table.set_index("cell")
    rows = []
    for cell in predictions.columns:
        draws = predictions[cell].dropna().to_numpy()
        scores = survival.interval_and_pinball(draws, float(actual.loc[cell, "eol"]), level, tau)
        rows.append(
            {"cell": cell, "CellType": actual.loc[cell, "CellType"], "eol": float(actual.loc[cell, "eol"]), **scores}
        )
    return pd.DataFrame(rows)
