"""Time to end of life: one record per cell, a test that stops early, and the Weibull survival function.

The end of life of a cell is the first cycle at which its state of health falls below the threshold. A cell
contributes one record to a survival model: the cycle of that first crossing if the test saw it, and otherwise the last
cycle that the test ran, with the record marked as censored. Stopping a test at a chosen cycle therefore turns the
cells that have not yet crossed into censored records, which is what a manufacturer has when it must decide before the
slow cells fail.

The Weibull distribution of a lifetime ``T`` with shape ``alpha`` and scale ``beta`` has the survival function
``S(t) = exp(-(t / beta) ** alpha)``. Its mean remaining life at an age ``t`` is the integral of ``S`` from ``t`` to
infinity divided by ``S(t)``, which is an upper incomplete gamma function and not ``beta * exp((t / beta) ** alpha)``
minus ``t``: the factor ``Gamma(1 + 1 / alpha)`` is part of the answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import special


def end_of_life(frame: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
    """The first cycle below the threshold for every cell, with the first and last cycle of its record.

    Args:
        frame: Rows with ``cell``, ``CellType``, ``cycle`` and ``soh`` (state of health in percent).
        threshold: End-of-life state of health in percent; a cycle with a state of health below it is a crossing.

    Returns:
        One row for each cell with ``cell``, ``CellType``, ``first``, ``last`` and ``eol``, where ``eol`` is ``NaN``
        for a cell that never crosses, sorted by chemistry and then by ``eol``.

    Raises:
        ValueError: If a cell is already below the threshold in its first record, so that its crossing is not seen.
    """
    ordered = frame.sort_values(["cell", "cycle"])
    grouped = ordered.groupby("cell", sort=False)
    table = pd.DataFrame(
        {
            "CellType": grouped["CellType"].first(),
            "first": grouped["cycle"].min(),
            "last": grouped["cycle"].max(),
            "eol": ordered[ordered["soh"] < threshold].groupby("cell")["cycle"].min(),
        }
    ).rename_axis("cell")
    starts = grouped["soh"].first()
    below = starts.index[starts < threshold].tolist()
    if below:
        raise ValueError(f"cells already below the threshold in their first record: {below}")
    return table.reset_index().sort_values(["CellType", "eol"], ignore_index=True)


def truncate(table: pd.DataFrame, at: float) -> pd.DataFrame:
    """The records of a test that stops at a cycle: the crossing if it was seen, otherwise a censored record.

    Args:
        table: The output of :func:`end_of_life`.
        at: The cycle at which the test stops.

    Returns:
        A copy of ``table`` with ``time``, the cycle of the crossing or the cycle at which the cell stopped being
        watched, and ``event``, true when the crossing was seen. A cell that has not crossed is watched until the
        test stops or its record ends, whichever comes first.
    """
    seen = table["eol"].notna() & (table["eol"] <= at)
    time = np.where(seen, table["eol"], np.minimum(at, table["last"]))
    return table.assign(time=time.astype(float), event=seen.to_numpy())


def weibull_survival(alpha: np.ndarray | float, beta: np.ndarray | float, age: np.ndarray | float) -> np.ndarray:
    """Probability that a cell is still in service at an age, ``exp(-(age / beta) ** alpha)``.

    Args:
        alpha: Shape of the Weibull distribution, positive.
        beta: Scale of the Weibull distribution in cycles, positive.
        age: Age in cycles, not negative.

    Returns:
        The survival probability, broadcast over the arguments.

    Raises:
        ValueError: If the shape or the scale is not positive or an age is negative.
    """
    alpha, beta, age = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(beta, float), np.asarray(age, float))
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("the shape and the scale must be positive")
    if np.any(age < 0):
        raise ValueError("an age cannot be negative")
    return np.exp(-((age / beta) ** alpha))


def weibull_remaining_life(alpha: np.ndarray | float, beta: np.ndarray | float, age: np.ndarray | float) -> np.ndarray:
    """Mean remaining life, in cycles, of a cell that is still in service at an age.

    The mean of ``T - age`` given ``T > age`` is ``beta * exp(x) * Gamma(1 + 1 / alpha, x) - age``, where
    ``x = (age / beta) ** alpha`` and ``Gamma(s, x)`` is the upper incomplete gamma function.

    Args:
        alpha: Shape of the Weibull distribution, positive.
        beta: Scale of the Weibull distribution in cycles, positive.
        age: Age in cycles, not negative.

    Returns:
        The mean remaining life, broadcast over the arguments; at age 0 it is the mean life ``beta * Gamma(1 + 1 /
        alpha)``.

    Raises:
        ValueError: If the shape or the scale is not positive or an age is negative.
    """
    alpha, beta, age = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(beta, float), np.asarray(age, float))
    if np.any(alpha <= 0) or np.any(beta <= 0):
        raise ValueError("the shape and the scale must be positive")
    if np.any(age < 0):
        raise ValueError("an age cannot be negative")
    s = 1.0 + 1.0 / alpha
    x = (age / beta) ** alpha
    # exp(x) * Gamma(s, x) = Gamma(s) * exp(x) * Q(s, x); the product is stable through the log of the regularised Q
    with np.errstate(divide="ignore"):
        log_tail = special.gammaln(s) + x + np.log(special.gammaincc(s, x))
    return beta * np.exp(log_tail) - age
