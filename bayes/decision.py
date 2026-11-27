"""The cost-loss decision on the probability that a cell is at its end of life, and the value of a forecast to it.

An operator inspects a cell and either retires it, at a cost ``C``, or leaves it in service. A cell left in service
that is below the end-of-life threshold causes a loss ``L``, and a cell above it costs nothing. With the ratio
``alpha = C / L`` and a probability ``p`` that the cell is below the threshold, retiring is cheaper than leaving the
cell when ``p > alpha`` (Murphy, 1977). Expenses are in units of ``L``. The value of a forecast is the share of the
possible saving that it achieves (Richardson, 2000): 0 for a user who has only the long-run rate of the event, and 1
for one who knows the outcome.

The module is plain numpy and pandas; the probabilities come from ``bayes.regression.evaluate``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check(probability: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arrays of equal length with probabilities in [0, 1] and outcomes of 0 or 1, as float and bool."""
    probability = np.asarray(probability, dtype=float)
    outcome = np.asarray(outcome).astype(bool)
    if probability.shape != outcome.shape or probability.ndim != 1:
        raise ValueError("probability and outcome must be one-dimensional and of the same length")
    if probability.size == 0:
        raise ValueError("at least one observation is needed")
    if np.any((probability < 0.0) | (probability > 1.0)) or np.any(np.isnan(probability)):
        raise ValueError("probabilities must lie in [0, 1]")
    return probability, outcome


def expense(probability: np.ndarray, outcome: np.ndarray, ratio: float) -> float:
    """Mean expense per inspection, in units of the loss, of retiring a cell when its probability exceeds the ratio.

    Args:
        probability: Probability that each cell is below the threshold.
        outcome: Whether each cell is below the threshold.
        ratio: Cost-loss ratio ``C / L``, between 0 and 1.

    Returns:
        The mean of ``ratio`` for each cell that is retired and ``1`` for each cell below the threshold that is left.

    Raises:
        ValueError: If the inputs differ in length, are empty, or hold a probability outside [0, 1].
    """
    probability, outcome = _check(probability, outcome)
    retire = probability > ratio
    return float(np.mean(np.where(retire, ratio, outcome.astype(float))))


def value(probability: np.ndarray, outcome: np.ndarray, ratio: float) -> float:
    """Relative economic value of the probabilities for one cost-loss ratio.

    Args:
        probability: Probability that each cell is below the threshold.
        outcome: Whether each cell is below the threshold.
        ratio: Cost-loss ratio, between 0 and 1.

    Returns:
        ``(E_rate - E) / (E_rate - E_perfect)``, where ``E`` is the expense of the probabilities, ``E_rate`` that of the
        better of always and never retiring, ``min(ratio, s)`` for an event rate ``s``, and ``E_perfect = ratio * s``.
        It is ``nan`` when the event never or always occurs, or when the ratio is 0 or 1, since there is then no
        saving to share.

    Raises:
        ValueError: If the inputs differ in length, are empty, or hold a probability outside [0, 1].
    """
    probability, outcome = _check(probability, outcome)
    rate = float(outcome.mean())
    best_without = min(ratio, rate)
    possible = best_without - ratio * rate
    if rate in (0.0, 1.0) or abs(possible) < 1e-12:
        return float("nan")
    return (best_without - expense(probability, outcome, ratio)) / possible


def value_curve(probability: np.ndarray, outcome: np.ndarray, ratios: np.ndarray) -> pd.DataFrame:
    """Expense and relative economic value over a grid of cost-loss ratios.

    Args:
        probability: Probability that each cell is below the threshold.
        outcome: Whether each cell is below the threshold.
        ratios: Cost-loss ratios, each between 0 and 1.

    Returns:
        Rows with ``ratio``, ``expense`` and ``value``.
    """
    ratios = np.asarray(ratios, dtype=float)
    return pd.DataFrame(
        {
            "ratio": ratios,
            "expense": [expense(probability, outcome, ratio) for ratio in ratios],
            "value": [value(probability, outcome, ratio) for ratio in ratios],
        }
    )


def value_frame(windows: pd.DataFrame, ratios: np.ndarray, threshold: float) -> pd.DataFrame:
    """The value curve of every strategy for each number of cycles seen.

    Args:
        windows: Scored cycles with ``strategy``, ``cycles_seen``, ``soh`` and ``p_below``.
        ratios: Cost-loss ratios, each between 0 and 1.
        threshold: End-of-life state of health in percent; a cycle with a state of health below it is an event.

    Returns:
        Rows with ``strategy``, ``cycles_seen``, ``ratio``, ``expense`` and ``value``.
    """
    parts = []
    for (strategy, seen), rows in windows.groupby(["strategy", "cycles_seen"], sort=False):
        curve = value_curve(rows["p_below"].to_numpy(), (rows["soh"] < threshold).to_numpy(), ratios)
        parts.append(curve.assign(strategy=strategy, cycles_seen=seen))
    return pd.concat(parts, ignore_index=True)
