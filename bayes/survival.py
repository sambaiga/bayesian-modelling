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
import pymc as pm
from scipy import special

CYCLE_SCALE = 1000.0  # cycles in one unit of the Weibull model, which keeps its parameters near one


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


def weibull_model(table: pd.DataFrame, scale: float = CYCLE_SCALE) -> pm.Model:
    """Weibull model of end of life with one record per cell, a chemistry effect on the scale and censoring.

    The lifetime of a cell of chemistry ``c`` is Weibull with a shape ``alpha`` shared by the chemistries and a scale
    whose logarithm is ``mu + (c - 1 / 2) * delta``, so that ``delta`` is the difference between the two chemistries.
    A record that is censored enters through the survival function at its time and a record that is not through the
    density. Two chemistries cannot support an estimate of a between-chemistry spread, so the difference is a single
    parameter with a prior that keeps it modest. Cycles are counted in units of ``scale``.

    Args:
        table: The output of :func:`truncate`, with ``CellType``, ``time`` and ``event``, and exactly two chemistries.
        scale: Number of cycles in one unit of the model, which keeps the parameters near one.

    Returns:
        The PyMC model, with the records attached.

    Raises:
        ValueError: If there are not exactly two chemistries.
    """
    chemistry_index, chemistries = pd.factorize(table["CellType"])
    if len(chemistries) != 2:
        raise ValueError("the model compares exactly two chemistries")
    time = table["time"].to_numpy(float) / scale
    upper = np.where(table["event"].to_numpy(bool), np.inf, time)  # a record at its upper bound is censored
    with pm.Model(coords={"chemistry": chemistries, "cell": table["cell"].to_numpy()}) as model:
        alpha = pm.LogNormal("alpha", np.log(2.0), 0.5)
        mu = pm.Normal("mu", np.log(0.8), 0.7)
        delta = pm.Normal("delta", 0.0, 0.5)
        beta_chem = pm.Deterministic("beta_chem", pm.math.exp(mu + (np.arange(2) - 0.5) * delta), dims="chemistry")
        pm.Censored(
            "lifetime",
            pm.Weibull.dist(alpha=alpha, beta=beta_chem[chemistry_index]),
            lower=None,
            upper=upper,
            observed=time,
            dims="cell",
        )
    return model


def weibull_conditional_draws(
    alpha: np.ndarray, beta: np.ndarray, age: float, generator: np.random.Generator
) -> np.ndarray:
    """Lifetimes of a cell that is known to be in service at an age, by the inverse of the conditional distribution.

    Given ``T > age`` the survival function is ``S(t) / S(age)``; solving ``S(t) / S(age) = 1 - u`` for a uniform ``u``
    gives ``t = beta * ((age / beta) ** alpha - log(1 - u)) ** (1 / alpha)``.

    Args:
        alpha: Posterior draws of the shape.
        beta: Posterior draws of the scale, with the same shape as ``alpha``.
        age: Age in cycles at which the cell is known to be in service, in the units of ``beta``.
        generator: Source of the uniform draws.

    Returns:
        One lifetime for each draw, all above ``age``.

    Raises:
        ValueError: If the shape or the scale is not positive or the age is negative.
    """
    alpha, beta = np.broadcast_arrays(np.asarray(alpha, float), np.asarray(beta, float))
    if np.any(alpha <= 0) or np.any(beta <= 0) or age < 0:
        raise ValueError("the shape and the scale must be positive and the age not negative")
    u = generator.uniform(size=alpha.shape)
    return beta * ((age / beta) ** alpha - np.log1p(-u)) ** (1.0 / alpha)


def interval_and_pinball(draws: np.ndarray, actual: float, level: float = 0.9, tau: float = 0.1) -> dict[str, float]:
    """Score the predictive draws of one end of life: the interval, its score and the loss of a rated life.

    The rated life is the ``tau`` quantile of the draws, the cycle that the manufacturer promises: a cell that fails
    earlier costs the loss and a promise below the life wastes what the cell could have delivered, which at a ratio
    ``tau`` of the costs makes the quantile the best promise. Its loss is the pinball loss.

    Args:
        draws: Predictive draws of the end of life, in cycles.
        actual: The end of life that was observed.
        level: Coverage of the central interval.
        tau: Quantile of the rated life, between 0 and 1.

    Returns:
        ``median`` and ``error`` (median minus actual), ``lower`` and ``upper`` of the central interval, ``covered``
        (1 when it contains the actual value), ``interval`` (width plus ``2 / (1 - level)`` times the distance by which
        the actual value falls outside it), ``rated`` (the ``tau`` quantile) and ``pinball``.

    Raises:
        ValueError: If there are no draws or ``level`` or ``tau`` is outside (0, 1).
    """
    draws = np.asarray(draws, float)
    if draws.size == 0:
        raise ValueError("at least one draw is needed")
    if not (0.0 < level < 1.0 and 0.0 < tau < 1.0):
        raise ValueError("level and tau must lie in (0, 1)")
    margin = (1.0 - level) / 2.0
    lower, median, upper, rated = np.quantile(draws, [margin, 0.5, 1.0 - margin, tau])
    outside = max(lower - actual, 0.0) + max(actual - upper, 0.0)
    gap = actual - rated
    return {
        "median": float(median),
        "error": float(median - actual),
        "lower": float(lower),
        "upper": float(upper),
        "covered": float(lower <= actual <= upper),
        "interval": float(upper - lower + 2.0 / (1.0 - level) * outside),
        "rated": float(rated),
        "pinball": float(max(tau * gap, (tau - 1.0) * gap)),
    }


def weibull_prior_lifetimes(draws: int, generator: np.random.Generator) -> np.ndarray:
    """Lifetimes in cycles that the priors of :func:`weibull_model` imply, before any cell is seen.

    A prior predictive check draws the parameters from their priors, then a lifetime from the Weibull distribution that
    they define, for a chemistry chosen at random (Gabry, Simpson, Vehtari, Betancourt and Gelman, 2019).

    Args:
        draws: Number of lifetimes.
        generator: Source of the random draws.

    Returns:
        The lifetimes in cycles.
    """
    alpha = generator.lognormal(np.log(2.0), 0.5, draws)
    mu = generator.normal(np.log(0.8), 0.7, draws)
    delta = generator.normal(0.0, 0.5, draws)
    chemistry = generator.integers(0, 2, draws)
    beta = np.exp(mu + (chemistry - 0.5) * delta)
    return CYCLE_SCALE * beta * generator.exponential(size=draws) ** (1.0 / alpha)


def weibull_survival_bands(
    idata, cycles: np.ndarray, level: float = 0.9, draws: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Posterior median and central band of the survival curve of each chemistry.

    Args:
        idata: A fit of :func:`weibull_model`.
        cycles: Cycles at which to evaluate the curve.
        level: Coverage of the band.
        draws: Number of posterior draws used.
        seed: Seed of the choice of draws.

    Returns:
        Rows with ``chemistry``, ``cycle``, ``median``, ``lower`` and ``upper``.
    """
    alpha = idata.posterior["alpha"].stack(s=("chain", "draw")).to_numpy()
    beta = idata.posterior["beta_chem"].stack(s=("chain", "draw"))
    pick = np.random.default_rng(seed).choice(alpha.size, size=min(draws, alpha.size), replace=False)
    margin = (1.0 - level) / 2.0
    rows = []
    for chemistry in beta["chemistry"].to_numpy():
        scale = beta.sel(chemistry=chemistry).to_numpy()[pick] * CYCLE_SCALE
        curves = weibull_survival(alpha[pick][:, None], scale[:, None], np.asarray(cycles, float)[None, :])
        lower, median, upper = np.quantile(curves, [margin, 0.5, 1.0 - margin], axis=0)
        rows.append(
            pd.DataFrame({"chemistry": chemistry, "cycle": cycles, "median": median, "lower": lower, "upper": upper})
        )
    return pd.concat(rows, ignore_index=True)
