"""Tests of the pooling strategies on cells that the fitted model has not seen, with a control and a common window.

A new cell is shown the first ``seen`` cycles of its life, and every strategy is scored on the same later window, so
that the scores of different ``seen`` are comparable. An embargo of ``gap`` cycles separates the cycles that a model
has seen from the cycles that it is scored on, because neighbouring cycles of one cell are strongly correlated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

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
END_OF_LIFE = 80.0  # state of health, in percent, below which a cell is at the end of its useful life


class ColdStart(NamedTuple):
    """The results of :func:`cold_start`.

    Attributes:
        scores: One row per strategy and number of cycles seen.
        diagnostics: The sampling diagnostics of every fit.
        curves: The predictions of every strategy for the whole life of the held-out cells, for each number of cycles
            seen, with ``strategy`` and ``cycles_seen``.
        windows: The scored cycles of the held-out cells with the probability ``p_below`` that the state of health is
            below the end-of-life threshold, for each strategy and number of cycles seen.
    """

    scores: pd.DataFrame
    diagnostics: pd.DataFrame
    curves: pd.DataFrame
    windows: pd.DataFrame


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
    threshold: float | None = None,
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
        threshold: A state of health in percent; when given, ``p_below`` holds the share of the predictive draws below
            it, the probability that the cell is at its end of life.

    Returns:
        The rows with the added columns ``median``, ``lower`` and ``upper``, and ``p_below`` when ``threshold`` is
        given.
    """
    pooling, diagnostics = STRATEGY[strategy]
    parts = []
    for position, (_, cell) in enumerate(rows.groupby("cell", sort=True)):
        predicted = hier.predict(
            idata, pooling, cell, scaler, features, cell["CellType"].iloc[0], seed + position, known, diagnostics, draws
        )
        lower, median, upper = bands(predicted)
        part = cell.assign(median=median, lower=lower, upper=upper)
        if threshold is not None:
            part = part.assign(p_below=(predicted < threshold).mean(axis=0))
        parts.append(part)
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
    main: dict[str, Any],
    refit: dict[str, Any],
    cache: Path,
    draws: int = 1000,
) -> ColdStart:
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
        The scores (one row per strategy and ``seen``), the sampling diagnostics of every fit, the predictions of each
        strategy for the whole life of the held-out cells for every ``seen``, and the scored cycles with the
        probability that the state of health is below ``END_OF_LIFE``.

    Raises:
        ValueError: If ``seen`` does not start with 0.
    """
    if not seen or seen[0] != 0:
        raise ValueError("seen must start with 0")
    window = held_out["cycle"] > max(seen) + gap
    scores, diagnostics, curves, windows = [], [], [], []
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
                idata,
                strategy,
                held_out[window],
                scaler,
                features,
                100 * index + number,
                number > 0,
                draws,
                END_OF_LIFE,
            )
            scores.append({"strategy": strategy, "cycles_seen": number, **score(later)})
            windows.append(later.assign(strategy=strategy, cycles_seen=number))
            curves.append(
                predict_cells(
                    idata, strategy, held_out, scaler, features, 100 * index + number, number > 0, draws
                ).assign(strategy=strategy, cycles_seen=number)
            )
    return ColdStart(
        pd.DataFrame(scores),
        pd.DataFrame(diagnostics),
        pd.concat(curves, ignore_index=True),
        pd.concat(windows, ignore_index=True),
    )


def phi_sensitivity(
    train: pd.DataFrame,
    held_out: pd.DataFrame,
    scaler: StandardScaler,
    features: list[str],
    seen: tuple[int, ...],
    gap: int,
    refit: dict[str, Any],
    cache: Path,
    phi_priors: dict[str, tuple[float, float]],
    trained: dict[str, Any],
    draws: int = 1000,
) -> ColdStart:
    """Score partial pooling on the held-out cells under other priors for the precision of the Beta likelihood.

    The same protocol as :func:`cold_start`, for the partial-pooling strategy only: with ``seen = 0`` the fit on the
    training cells is taken from ``trained``, and for larger values it is refitted with the cycles of the held-out
    cells up to ``seen``. The precision sets the width of every predictive interval, so a prior that decides it also
    decides coverage, the interval score and the probabilities that the retirement rule reads.

    Args:
        train: Rows of the training cells.
        held_out: Rows of the held-out cells.
        scaler: Scaler fitted on the training cells.
        features: Names of the diagnostic columns.
        seen: Numbers of cycles shown to the model; it starts with 0.
        gap: Embargo in cycles between the last cycle seen and the first cycle scored.
        refit: ``draws``, ``tune`` and ``chains`` of the refits.
        cache: Folder of the stored fits.
        phi_priors: Mean and standard deviation of the logarithm of the precision, for each setting.
        trained: The fit on the training cells under each setting, keyed like ``phi_priors``.
        draws: Posterior draws used for each prediction.

    Returns:
        A :class:`ColdStart` whose ``strategy`` column holds the name of the setting.

    Raises:
        ValueError: If ``seen`` does not start with 0.
    """
    if not seen or seen[0] != 0:
        raise ValueError("seen must start with 0")
    window = held_out["cycle"] > max(seen) + gap
    scores, diagnostics, windows = [], [], []
    for index, (setting, phi_prior) in enumerate(phi_priors.items()):
        for number in seen:
            if number == 0:
                idata = trained[setting]
            else:
                frame = pd.concat([train, held_out[held_out["cycle"] <= number]], ignore_index=True)
                model = hier.hierarchical_model(frame, scaler, features, "partial", phi_prior=phi_prior)
                idata = hier.sample(
                    model,
                    cache / f"phi-{setting.lower()}-{number}.nc",
                    seed=7000 + 100 * index + number,
                    **refit,
                )
            diagnostics.append({"strategy": setting, "cycles_seen": number, **hier.convergence(idata)})
            later = predict_cells(
                idata,
                "partial",
                held_out[window],
                scaler,
                features,
                500 * index + number,
                number > 0,
                draws,
                END_OF_LIFE,
            )
            scores.append({"strategy": setting, "cycles_seen": number, **score(later)})
            windows.append(later.assign(strategy=setting, cycles_seen=number))
    return ColdStart(
        pd.DataFrame(scores), pd.DataFrame(diagnostics), pd.DataFrame(), pd.concat(windows, ignore_index=True)
    )


def leave_one_out(
    frame: pd.DataFrame,
    features: list[str],
    strategy: str,
    cache: Path,
    settings: dict[str, Any],
    seed: int = 2026,
    draws: int = 1000,
    alpha: float = ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict each cell from a model fitted to all the other cells, and score the whole life of the cell.

    Every fold fits its own scaler on its training cells, so nothing of the held-out cell reaches the model. A fold is
    stored in ``cache`` under the strategy and the cell, so a repeat run reads the fits back.

    Args:
        frame: Rows of all cells, with ``cell``, ``CellType``, ``cycle``, ``soh``, ``y`` and the features.
        features: Names of the diagnostic columns.
        strategy: A key of ``STRATEGY``.
        cache: Folder of the stored fits.
        settings: ``draws``, ``tune`` and ``chains`` of each fit.
        seed: Seed of the first fit; a fold uses ``seed`` plus its position.
        draws: Predictive draws per cell.
        alpha: Significance level of the intervals.

    Returns:
        The scores, one row per cell, and the predictions of every cell, with ``median``, ``lower`` and ``upper``.

    Raises:
        ValueError: If there are fewer than two cells or ``strategy`` is unknown.
    """
    if strategy not in STRATEGY:
        raise ValueError(f"strategy must be one of {list(STRATEGY)}, got {strategy!r}")
    cells = sorted(frame["cell"].unique())
    if len(cells) < 2:
        raise ValueError("leave-one-out needs at least two cells")
    pooling, diagnostics = STRATEGY[strategy]
    scores, curves = [], []
    for position, cell in enumerate(cells):
        train, test = frame[frame["cell"] != cell], frame[frame["cell"] == cell]
        scaler = StandardScaler().fit(train[features])
        model = hier.hierarchical_model(train, scaler, features, pooling, diagnostics)
        idata = hier.sample(model, cache / f"loo-{strategy}-{cell}.nc", seed=seed + position, **settings)
        predicted = predict_cells(idata, strategy, test, scaler, features, seed + 100 * position, False, draws)
        scores.append({"cell": cell, "strategy": strategy, **score(predicted, alpha)})
        curves.append(predicted.assign(strategy=strategy))
    return pd.DataFrame(scores), pd.concat(curves, ignore_index=True)


def percent_change(
    scores: pd.DataFrame,
    measures: dict[str, str],
    strategy: str,
    reference: str,
    key: str = "cell",
) -> pd.DataFrame:
    """Change in percent of one strategy against a reference strategy, for each row key and measure.

    Args:
        scores: Rows with ``strategy``, the ``key`` column and the measures, such as the result of ``leave_one_out``.
        measures: Titles of the panels, by measure column, for example ``{"mae": "(a) Error of the median"}``.
        strategy: The strategy that is compared.
        reference: The strategy that is the reference, whose value of each measure is the zero of the change.
        key: Column that identifies a row of the comparison, such as the held-out cell.

    Returns:
        Rows with ``panel``, ``name`` (the key) and ``value`` (``100 * (strategy / reference - 1)``).

    Raises:
        ValueError: If a strategy has no rows, or a reference value is not positive.
    """
    parts = []
    for name in (strategy, reference):
        if name not in set(scores["strategy"]):
            raise ValueError(f"no rows for the strategy {name!r}")
    compared = scores[scores["strategy"] == strategy].set_index(key)
    base = scores[scores["strategy"] == reference].set_index(key)
    for measure, title in measures.items():
        if (base[measure] <= 0).any():
            raise ValueError(f"the reference value of {measure!r} must be positive")
        change = 100.0 * (compared[measure] / base[measure] - 1.0)
        parts.append(pd.DataFrame({"panel": title, "name": change.index, "value": change.to_numpy()}))
    return pd.concat(parts, ignore_index=True)


def change_by_strategy(
    scores: pd.DataFrame,
    measure: str,
    reference: str,
    strategies: list[str],
    labels: dict[str, str],
    panel: str = "cycles_seen",
) -> pd.DataFrame:
    """Change in percent of each strategy against a reference strategy, for each panel.

    Args:
        scores: Rows with ``strategy``, the ``panel`` column and the measure, such as the result of ``cold_start``.
        measure: The measure column, for which lower is better.
        reference: The strategy that is the reference, whose value is the zero of the change.
        strategies: The strategies that are compared with it.
        labels: Name of each strategy in the figure.
        panel: Column whose values are the panels, such as the cycles of a new cell that the model has seen.

    Returns:
        Rows with ``panel`` (the title, with a letter and the value), ``name`` and ``value`` in percent.

    Raises:
        ValueError: If a reference value is not positive.
    """
    parts = []
    for index, (value, rows) in enumerate(scores.groupby(panel, sort=True)):
        base = rows.loc[rows["strategy"] == reference, measure].iloc[0]
        if base <= 0:
            raise ValueError(f"the reference value of {measure!r} must be positive")
        for name in strategies:
            now = rows.loc[rows["strategy"] == name, measure].iloc[0]
            parts.append(
                {
                    "panel": f"({'abcdefgh'[index]}) {value:g} cycles seen",
                    "name": labels[name],
                    "value": 100.0 * (now / base - 1.0),
                }
            )
    return pd.DataFrame(parts)


def coverage_gap(
    scores: pd.DataFrame,
    nominal: float,
    strategies: list[str],
    labels: dict[str, str],
    subject: str,
    tolerance: float = 10.0,
    panel: str = "cycles_seen",
) -> pd.DataFrame:
    """Coverage of each strategy minus the nominal coverage, in percentage points, with a role for each bar.

    A strategy that falls short of the nominal coverage by more than ``tolerance`` points is over-confident and is
    marked ``"worse"``. A strategy that exceeds it has intervals that are too wide, which the interval score
    penalises, and is not marked. The subject strategy is marked ``"subject"`` otherwise.

    Args:
        scores: Rows with ``strategy``, the ``panel`` column and ``coverage``.
        nominal: Nominal coverage as a share, for example 0.9.
        strategies: Strategies to show, in the order of the rows.
        labels: Name of each strategy in the figure.
        subject: The strategy that the figure is about.
        tolerance: Shortfall in points beyond which a bar is marked worse.
        panel: Column whose values are the panels.

    Returns:
        Rows with ``panel``, ``name``, ``value`` (points) and ``role``.
    """
    parts = []
    for index, (value, rows) in enumerate(scores.groupby(panel, sort=True)):
        for name in strategies:
            gap = 100.0 * (rows.loc[rows["strategy"] == name, "coverage"].iloc[0] - nominal)
            role = "worse" if gap < -tolerance else ("subject" if name == subject else "other")
            parts.append(
                {
                    "panel": f"({'abcdefgh'[index]}) {value:g} cycles seen",
                    "name": labels[name],
                    "value": gap,
                    "role": role,
                }
            )
    return pd.DataFrame(parts)
