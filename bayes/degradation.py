"""A power-law path of capacity loss for each cell, and the cycle at which the path crosses the end-of-life threshold.

The loss of a cell, 100 minus its state of health in percent, grows as ``exp(a) * (cycle / 500) ** b``. Here
``exp(a)`` is the loss after 500 cycles and ``b`` is the shape of the decline, linear at 1 and accelerating above it.
The path has no ceiling and no floor, so it reaches any loss, and the cycle at which it reaches the end-of-life loss
has a closed form, ``500 * (loss / exp(a)) ** (1 / b)``. The reference of 500 cycles lies inside the data, which keeps
the level and the shape from trading off against each other in the posterior; a reference of 1,000 cycles made the
sampler slow and divergent. That is what a degradation analysis needs and what the
saturating curve of the earlier model cannot give, since that curve levels off unless a diagnostic that is not known in
advance keeps it falling (Lu and Meeker, 1993).

The parameters of a cell come from its chemistry, with a spread between cells that each chemistry estimates, and the
cell offsets are sampled non-centred. The chemistries have priors of their own and no common parent, since a spread
between two chemistries cannot be estimated from two groups and, tried, made the sampler diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm

CYCLE_UNIT = 500.0  # the reference cycle: exp(a) is the loss after this many cycles


def path_model(frame: pd.DataFrame, cell: str = "cell", chemistry: str = "CellType") -> pm.Model:
    """Hierarchical power-law path of the loss of state of health, with Gaussian noise.

    Args:
        frame: Rows with the cell and chemistry columns, ``cycle`` and ``soh`` in percent.
        cell: Name of the cell column.
        chemistry: Name of the chemistry column.

    Returns:
        The PyMC model, with the observed loss attached. It stores ``a_cell`` and ``b_cell`` for every cell.
    """
    cell_index, cells = pd.factorize(frame[cell])
    chemistry_of_cell = frame.drop_duplicates(cell).set_index(cell).loc[cells, chemistry]
    chemistry_index, chemistries = pd.factorize(chemistry_of_cell)
    coords = {"cell": cells, "chemistry": chemistries, "obs": np.arange(len(frame))}
    with pm.Model(coords=coords) as model:
        cycle = pm.Data("cycle", frame["cycle"].to_numpy(float) / CYCLE_UNIT, dims="obs")
        a_chem = pm.Normal("a_chem", np.log(10.0), 1.0, dims="chemistry")
        logb_chem = pm.Normal("logb_chem", np.log(1.5), 0.5, dims="chemistry")
        sigma_cell_a = pm.HalfNormal("sigma_cell_a", 0.5, dims="chemistry")
        sigma_cell_b = pm.HalfNormal("sigma_cell_b", 0.3, dims="chemistry")
        z_a = pm.Normal("z_a", 0.0, 1.0, dims="cell")
        z_b = pm.Normal("z_b", 0.0, 1.0, dims="cell")
        a_cell = pm.Deterministic("a_cell", a_chem[chemistry_index] + z_a * sigma_cell_a[chemistry_index], dims="cell")
        b_cell = pm.Deterministic(
            "b_cell", pm.math.exp(logb_chem[chemistry_index] + z_b * sigma_cell_b[chemistry_index]), dims="cell"
        )
        noise = pm.HalfNormal("noise", 2.0)
        mean = pm.math.exp(a_cell[cell_index]) * cycle ** b_cell[cell_index]
        pm.Normal("loss", mean, noise, observed=100.0 - frame["soh"].to_numpy(float), dims="obs")
    return model


def crossing_cycle(a: np.ndarray, b: np.ndarray, loss: float = 20.0, unit: float = CYCLE_UNIT) -> np.ndarray:
    """Cycle at which the path ``exp(a) * (cycle / unit) ** b`` reaches a loss.

    Args:
        a: Logarithm of the loss after ``unit`` cycles.
        b: Shape of the path, positive.
        loss: The loss in points of state of health that defines the end of life.
        unit: The reference cycle of the path.

    Returns:
        The cycle of the crossing, broadcast over ``a`` and ``b``.

    Raises:
        ValueError: If ``b`` is not positive or the loss is not positive.
    """
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    if np.any(b <= 0) or loss <= 0:
        raise ValueError("the shape and the loss must be positive")
    return unit * (loss / np.exp(a)) ** (1.0 / b)


def cell_crossings(idata, loss: float = 20.0) -> pd.DataFrame:
    """Posterior draws of the crossing cycle of every cell, one column for each cell.

    Args:
        idata: A fit of :func:`path_model` with ``a_cell`` and ``b_cell`` in its posterior.
        loss: The loss that defines the end of life, in points of state of health.

    Returns:
        A frame with one row for each posterior draw and one column for each cell.
    """
    a = idata.posterior["a_cell"].stack(draw_all=("chain", "draw")).transpose("draw_all", "cell")
    b = idata.posterior["b_cell"].stack(draw_all=("chain", "draw")).transpose("draw_all", "cell")
    return pd.DataFrame(crossing_cycle(a.to_numpy(), b.to_numpy(), loss), columns=a["cell"].to_numpy())


def path_bands(idata, cycles: np.ndarray, level: float = 0.9, draws: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Posterior median and central band of the loss path of every cell.

    Args:
        idata: A fit of :func:`path_model`.
        cycles: Cycles at which to evaluate the paths.
        level: Coverage of the band.
        draws: Number of posterior draws used.
        seed: Seed of the choice of draws.

    Returns:
        Rows with ``cell``, ``cycle``, ``median``, ``lower`` and ``upper``, the loss in points of state of health.
    """
    a = idata.posterior["a_cell"].stack(s=("chain", "draw")).transpose("s", "cell")
    b = idata.posterior["b_cell"].stack(s=("chain", "draw")).transpose("s", "cell")
    pick = np.random.default_rng(seed).choice(a.sizes["s"], size=min(draws, a.sizes["s"]), replace=False)
    margin = (1.0 - level) / 2.0
    grid = np.asarray(cycles, float) / CYCLE_UNIT
    rows = []
    for index, cell in enumerate(a["cell"].to_numpy()):
        paths = np.exp(a.to_numpy()[pick, index])[:, None] * grid[None, :] ** b.to_numpy()[pick, index][:, None]
        lower, median, upper = np.quantile(paths, [margin, 0.5, 1.0 - margin], axis=0)
        rows.append(pd.DataFrame({"cell": cell, "cycle": cycles, "median": median, "lower": lower, "upper": upper}))
    return pd.concat(rows, ignore_index=True)
