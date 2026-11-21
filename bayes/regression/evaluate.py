"""Tests of the pooling strategies on cells that the fitted model has not seen, with a control and a common window.

A new cell is shown the first ``seen`` cycles of its life, and every strategy is scored on the same later window, so
that the scores of different ``seen`` are comparable. An embargo of ``gap`` cycles separates the cycles that a model
has seen from the cycles that it is scored on, because neighbouring cycles of one cell are strongly correlated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bayes.metrics.interval import interval_coverage, interval_score
from bayes.regression import hierarchical as hier

# Each strategy is a level of pooling and whether the diagnostics enter the model. The control has no diagnostics.
STRATEGY = {
    "none": ("none", True),
    "complete": ("complete", True),
    "partial": ("partial", True),
    "control": ("complete", False),
}
ALPHA = 0.1


def bands(draws: np.ndarray, alpha: float = ALPHA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower bound, median and upper bound of the central interval of predictive draws.

    Args:
        draws: Array of shape (draws, rows).
        alpha: Significance level; the interval has coverage ``1 - alpha``.

    Returns:
        The lower bound, the median and the upper bound, each of length ``rows``.
    """
    lower, median, upper = np.quantile(draws, [alpha / 2, 0.5, 1 - alpha / 2], axis=0)
    return lower, median, upper


def predict_cells(
    idata,  # noqa: ANN001
    strategy: str,
    rows: pd.DataFrame,
    scaler: StandardScaler,
    features: list[str],
    seed: int,
    known: bool,
    draws: int = 1000,
) -> pd.DataFrame:
    """Median and 90% interval of the state of health at the cycles of each cell in ``rows``.

    Args:
        idata: Inference data of the fitted model of the strategy.
        strategy: A key of ``STRATEGY``.
        rows: Rows of one or more cells, with ``cell``, ``CellType``, ``cycle``, ``soh`` and the features.
        scaler: The scaler that the model was built with.
        features: Names of the diagnostic columns.
        seed: Seed of the draws; a cell uses ``seed`` plus its position, so no prediction depends on call order.
        known: Whether the cells were part of the fitted data.
        draws: Number of posterior draws.

    Returns:
        The rows with the added columns ``median``, ``lower`` and ``upper``.
    """
    pooling, diagnostics = STRATEGY[strategy]
    parts = []
    for position, (_, cell) in enumerate(rows.groupby("cell", sort=True)):
        predicted = hier.predict(
            idata, pooling, cell, scaler, features, cell["CellType"].iloc[0], seed + position, known, diagnostics, draws
        )
        lower, median, upper = bands(predicted)
        parts.append(cell.assign(median=median, lower=lower, upper=upper))
    return pd.concat(parts, ignore_index=True)


def score(frame: pd.DataFrame, alpha: float = ALPHA) -> dict[str, float]:
    """Error of the median and the coverage, width and interval score of the intervals, in points of state of health.

    Args:
        frame: Rows with ``soh``, ``median``, ``lower`` and ``upper``.
        alpha: Significance level of the intervals.

    Returns:
        ``mae``, ``rmse``, ``coverage``, ``width``, ``interval`` and the number of observations ``n``.
    """
    observed = frame["soh"].to_numpy()
    error = frame["median"].to_numpy() - observed
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "coverage": float(interval_coverage(observed, frame["lower"].to_numpy(), frame["upper"].to_numpy())),
        "width": float(np.mean(frame["upper"] - frame["lower"])),
        "interval": interval_score(observed, frame["lower"].to_numpy(), frame["upper"].to_numpy(), alpha),
        "n": int(len(frame)),
    }


def cold_start(
    train: pd.DataFrame,
    held_out: pd.DataFrame,
    scaler: StandardScaler,
    features: list[str],
    seen: tuple[int, ...],
    gap: int,
    main: dict[str, int],
    refit: dict[str, int],
    cache: Path,
    draws: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every strategy on the held-out cells, after the model has seen 0, then more, early cycles of them.

    With ``seen = 0`` the main fits on the training cells are used. For larger values the strategy is refitted on the
    training cells plus the cycles of the held-out cells up to ``seen``. Every setting is scored on the cycles after
    ``max(seen) + gap``, the same window throughout.

    Args:
        train: Rows of the training cells, with ``cell``, ``CellType``, ``cycle``, ``y``, ``soh`` and the features.
        held_out: Rows of the held-out cells.
        scaler: Scaler fitted on the training cells.
        features: Names of the diagnostic columns.
        seen: Numbers of cycles shown to the model; it starts with 0.
        gap: Embargo in cycles between the last cycle seen and the first cycle scored.
        main: ``draws``, ``tune`` and ``chains`` of the fits on the training cells.
        refit: The same for the refits.
        cache: Folder of the stored fits.
        draws: Posterior draws used for each prediction.

    Returns:
        The scores (one row per strategy and ``seen``), the sampling diagnostics of every fit, and the predictions of
        each strategy for the whole life of the held-out cells when nothing of them has been seen.

    Raises:
        ValueError: If ``seen`` does not start with 0.
    """
    if not seen or seen[0] != 0:
        raise ValueError("seen must start with 0")
    window = held_out["cycle"] > max(seen) + gap
    scores, diagnostics, curves = [], [], []
    for index, strategy in enumerate(STRATEGY):
        pooling, with_diagnostics = STRATEGY[strategy]
        for number in seen:
            frame = (
                train if number == 0 else pd.concat([train, held_out[held_out["cycle"] <= number]], ignore_index=True)
            )
            model = hier.hierarchical_model(frame, scaler, features, pooling, with_diagnostics)
            settings, name = (main, f"train-{strategy}") if number == 0 else (refit, f"cold{number}-{strategy}")
            seed = 2026 + index if number == 0 else 3000 + 10 * number + index
            idata = hier.sample(
                model, cache / f"{name}.nc", seed=seed, log_likelihood=number == 0 and strategy != "control", **settings
            )
            diagnostics.append({"strategy": strategy, "cycles_seen": number, **hier.convergence(idata)})
            later = predict_cells(
                idata, strategy, held_out[window], scaler, features, 100 * index + number, number > 0, draws
            )
            scores.append({"strategy": strategy, "cycles_seen": number, **score(later)})
            if number == 0:
                curves.append(
                    predict_cells(idata, strategy, held_out, scaler, features, 100 * index, False, draws).assign(
                        strategy=strategy
                    )
                )
    return pd.DataFrame(scores), pd.DataFrame(diagnostics), pd.concat(curves, ignore_index=True)
