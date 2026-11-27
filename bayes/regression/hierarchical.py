"""Hierarchical Beta regression of battery state of health: cells nested in chemistries, at three levels of pooling.

The module builds the model, samples it with a cache, summarises the posterior and predicts the state of health of a
cell that was, or was not, part of the fitted data. It holds no plotting and no scoring; see ``bayes.plot.hierarchy``
and ``bayes.metrics.interval``.
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

POOLING = ("partial", "complete", "none")
CEILING = 1.1  # state of health is divided by this, so that it stays inside the open unit interval
CYCLE_UNIT = 1000.0  # cycles are counted in thousands, which keeps the rate near one


def state_of_health(soh: pd.Series | np.ndarray, ceiling: float = CEILING) -> np.ndarray:
    """Scale a state of health in percent to the open unit interval that a Beta likelihood requires.

    Args:
        soh: State of health in percent.
        ceiling: Value by which the fraction is divided; a state of health above 100% then stays below one.

    Returns:
        The scaled values, clipped to 0.001 and 0.999.
    """
    return np.clip(np.asarray(soh, dtype=float) / 100.0 / ceiling, 1e-3, 1 - 1e-3)


def hierarchical_model(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    features: list[str],
    pooling: str = "partial",
    diagnostics: bool = True,
    cell: str = "cell",
    chemistry: str = "CellType",
    phi_prior: tuple[float, float] = (float(np.log(100.0)), 1.0),
) -> pm.Model:
    """Beta regression of the state of health on the cycle and on diagnostics, at one of three levels of pooling.

    The mean follows a saturating exponential in the cycle count, with an initial level, an amplitude and a rate, plus a
    linear effect of the standardised diagnostics, all on the logit scale. The variants differ only in how the cell
    parameters are tied together: ``"complete"`` gives every cell the same parameters, ``"none"`` gives each cell
    its own
    with no shared hyperparameters, and ``"partial"`` draws the parameters of a cell from its chemistry and those of a
    chemistry from the fleet. Cell offsets are sampled non-centred, which avoids a funnel in the posterior.

    Args:
        frame: Rows with the cell and chemistry columns, ``cycle``, ``y`` (the scaled state of health) and ``features``.
        scaler: Scaler fitted on the training cells, so that every frame is standardised in the same way.
        features: Names of the diagnostic columns.
        pooling: ``"partial"``, ``"complete"`` or ``"none"``.
        diagnostics: Whether the diagnostics enter the model; ``False`` is the control without them.
        cell: Name of the cell column.
        chemistry: Name of the chemistry column.
        phi_prior: Mean and standard deviation of the logarithm of the precision ``phi``, which has a log-normal prior.
            The default is centred on 100, a noise of about five points of state of health at mid-life.

    Returns:
        The PyMC model, with the observations of ``y`` attached.

    Raises:
        ValueError: If ``pooling`` is not one of ``POOLING``.
    """
    if pooling not in POOLING:
        raise ValueError(f"pooling must be one of {POOLING}, got {pooling!r}")
    cell_index, cells = pd.factorize(frame[cell])
    chemistry_of_cell = frame.drop_duplicates(cell).set_index(cell).loc[cells, chemistry]
    chemistry_index, chemistries = pd.factorize(chemistry_of_cell)
    coords = {"cell": cells, "chemistry": chemistries, "feature": features, "obs": np.arange(len(frame))}
    with pm.Model(coords=coords) as model:
        cycle = pm.Data("cycle", frame["cycle"].to_numpy() / CYCLE_UNIT, dims="obs")
        inputs = pm.Data("diagnostics", scaler.transform(frame[features]), dims=("obs", "feature"))
        phi = pm.LogNormal("phi", phi_prior[0], phi_prior[1])

        if pooling == "partial":
            mu_alpha = pm.Normal("mu_alpha", 2.0, 1.0)
            sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)
            mu_lambda = pm.Normal("mu_lambda", np.log(5.0), 0.5)
            sigma_lambda = pm.HalfNormal("sigma_lambda", 0.5)
            amplitude_scale = pm.HalfNormal("amplitude_scale", 1.0)
            alpha_chem = pm.Normal("alpha_chem", mu_alpha, sigma_alpha, dims="chemistry")
            loglambda_chem = pm.Normal("loglambda_chem", mu_lambda, sigma_lambda, dims="chemistry")
            amplitude_chem = pm.HalfNormal("amplitude_chem", amplitude_scale, dims="chemistry")
            sigma_cell_alpha = pm.HalfNormal("sigma_cell_alpha", 0.5, dims="chemistry")
            sigma_cell_lambda = pm.HalfNormal("sigma_cell_lambda", 0.5, dims="chemistry")
            offset_alpha = pm.Normal("z_alpha", 0.0, 1.0, dims="cell")
            offset_lambda = pm.Normal("z_lambda", 0.0, 1.0, dims="cell")
            alpha_cell = pm.Deterministic(
                "alpha_cell",
                alpha_chem[chemistry_index] + offset_alpha * sigma_cell_alpha[chemistry_index],
                dims="cell",
            )
            rate_cell = pm.Deterministic(
                "lambda_cell",
                pm.math.exp(loglambda_chem[chemistry_index] + offset_lambda * sigma_cell_lambda[chemistry_index]),
                dims="cell",
            )
            amplitude_cell = pm.Deterministic("amplitude_cell", amplitude_chem[chemistry_index], dims="cell")
            alpha, rate, amplitude = alpha_cell[cell_index], rate_cell[cell_index], amplitude_cell[cell_index]
        elif pooling == "complete":
            alpha = pm.Normal("alpha", 2.0, 1.0)
            rate = pm.math.exp(pm.Normal("loglambda", np.log(5.0), 0.5))
            amplitude = pm.HalfNormal("amplitude", 1.0)
        else:
            alpha_cell = pm.Normal("alpha_cell", 2.0, 1.0, dims="cell")
            rate_cell = pm.Deterministic(
                "lambda_cell", pm.math.exp(pm.Normal("loglambda_cell", np.log(5.0), 0.5, dims="cell")), dims="cell"
            )
            amplitude_cell = pm.HalfNormal("amplitude_cell", 1.0, dims="cell")
            alpha, rate, amplitude = alpha_cell[cell_index], rate_cell[cell_index], amplitude_cell[cell_index]

        eta = alpha - amplitude * (1 - pm.math.exp(-rate * cycle))
        if diagnostics:
            beta = pm.Normal("beta", 0.0, 0.2, dims="feature")
            eta = eta + pm.math.dot(inputs, beta)
        mean = pm.math.invlogit(eta)
        pm.Beta("y", mean * phi, (1 - mean) * phi, observed=frame["y"].to_numpy(), dims="obs")
    return model


def sample(
    model: pm.Model,
    path: Path,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    log_likelihood: bool = False,
) -> az.InferenceData:
    """Sample the posterior of a model, or read the stored samples when the same fit has been run before.

    Args:
        model: The model to sample.
        path: File of the stored samples; read when it exists and written after sampling when it does not.
        draws: Draws per chain after tuning.
        tune: Tuning draws per chain.
        chains: Number of chains.
        seed: Seed of the sampler, recorded so that a run can be repeated.
        log_likelihood: Whether to store the log likelihood, which leave-one-out cross-validation needs.

    Returns:
        The inference data.
    """
    if path.exists():
        return az.from_netcdf(path)
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=min(chains, 2),
            target_accept=0.9,
            random_seed=seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": log_likelihood},
        )
    idata.to_netcdf(path)
    return idata


def convergence(idata: az.InferenceData) -> dict[str, float]:
    """Divergences, the worst R-hat and the smallest effective sample sizes of a fit.

    Args:
        idata: Inference data of a fitted model.

    Returns:
        Divergences, maximum R-hat and minimum bulk and tail effective sample sizes.
    """
    table = az.summary(idata, kind="diagnostics")
    return {
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "r_hat": float(table["r_hat"].max()),
        "ess_bulk": float(table["ess_bulk"].min()),
        "ess_tail": float(table["ess_tail"].min()),
    }


def cell_parameters(
    idata: az.InferenceData, pooling: str, count: int, generator: np.random.Generator, chemistry: str, known: str | None
) -> dict[str, np.ndarray]:
    """Draws of the curve parameters of one cell, from its own posterior or from what the model says of a new cell.

    A fitted cell has its own draws. A new cell has none, so its parameters come from the chemistry level for partial
    pooling (with fresh offsets), from the common parameters for complete pooling, and from the prior for no pooling.

    Args:
        idata: Inference data of a model fitted by ``hierarchical_model``.
        pooling: ``"partial"``, ``"complete"`` or ``"none"``.
        count: Number of draws.
        generator: Source of the random numbers.
        chemistry: Chemistry of the cell, used for a new cell under partial pooling.
        known: Name of the cell when it is part of the fitted data, otherwise ``None``.

    Returns:
        Arrays ``alpha``, ``rate``, ``amplitude``, ``phi`` and, with diagnostics, ``beta``, one row per draw.
    """
    posterior = idata.posterior.stack(sample=("chain", "draw"))
    chosen = generator.choice(posterior.sizes["sample"], size=min(count, posterior.sizes["sample"]), replace=False)
    posterior = posterior.isel(sample=chosen)
    size = posterior.sizes["sample"]
    common = {"phi": posterior["phi"].to_numpy()}
    if "beta" in posterior:
        common["beta"] = posterior["beta"].transpose("sample", "feature").to_numpy()
    if pooling == "complete":
        return {
            **common,
            "alpha": posterior["alpha"].to_numpy(),
            "rate": np.exp(posterior["loglambda"].to_numpy()),
            "amplitude": posterior["amplitude"].to_numpy(),
        }
    if known is not None:
        return {
            **common,
            "alpha": posterior["alpha_cell"].sel(cell=known).to_numpy(),
            "rate": posterior["lambda_cell"].sel(cell=known).to_numpy(),
            "amplitude": posterior["amplitude_cell"].sel(cell=known).to_numpy(),
        }
    if pooling == "partial":
        level = posterior.sel(chemistry=chemistry)
        alpha = level["alpha_chem"].to_numpy() + level["sigma_cell_alpha"].to_numpy() * generator.standard_normal(size)
        rate = np.exp(
            level["loglambda_chem"].to_numpy() + level["sigma_cell_lambda"].to_numpy() * generator.standard_normal(size)
        )
        return {**common, "alpha": alpha, "rate": rate, "amplitude": level["amplitude_chem"].to_numpy()}
    return {
        **common,
        "alpha": 2.0 + generator.standard_normal(size),
        "rate": np.exp(np.log(5.0) + 0.5 * generator.standard_normal(size)),
        "amplitude": np.abs(generator.standard_normal(size)),
    }


def predict(
    idata: az.InferenceData,
    pooling: str,
    rows: pd.DataFrame,
    scaler: StandardScaler,
    features: list[str],
    chemistry: str,
    seed: int,
    known: bool = False,
    diagnostics: bool = True,
    draws: int = 1000,
    cell: str = "cell",
) -> np.ndarray:
    """Posterior predictive draws of the state of health (in percent) of one cell at the cycles of ``rows``.

    Args:
        idata: Inference data of a model fitted by ``hierarchical_model``.
        pooling: ``"partial"``, ``"complete"`` or ``"none"``.
        rows: Rows of one cell, with the cell column, ``cycle`` and the diagnostics.
        scaler: The scaler that the model was built with.
        features: Names of the diagnostic columns.
        chemistry: Chemistry of the cell.
        seed: Seed of the draws, so that a prediction does not depend on the order of the calls.
        known: Whether the cell was part of the fitted data.
        diagnostics: Whether the model uses the diagnostics.
        draws: Number of posterior draws used.
        cell: Name of the cell column.

    Returns:
        Array of shape (draws, rows), including the noise of the Beta likelihood.
    """
    generator = np.random.default_rng(seed)
    pars = cell_parameters(idata, pooling, draws, generator, chemistry, rows[cell].iloc[0] if known else None)
    cycle = rows["cycle"].to_numpy() / CYCLE_UNIT
    eta = pars["alpha"][:, None] - pars["amplitude"][:, None] * (1 - np.exp(-pars["rate"][:, None] * cycle[None, :]))
    if diagnostics:
        eta = eta + pars["beta"] @ scaler.transform(rows[features]).T
    mean = expit(eta)
    precision = pars["phi"][:, None]
    return generator.beta(mean * precision, (1 - mean) * precision) * CEILING * 100.0


def chemistry_intervals(idata: az.InferenceData, probability: float = 0.9) -> pd.DataFrame:
    """Posterior mean and central interval of the three chemistry parameters of a partial-pooling fit.

    Args:
        idata: Inference data of a partial-pooling model.
        probability: Probability mass of the interval.

    Returns:
        One row per quantity and chemistry, with ``mean``, ``lower`` and ``upper``.
    """
    tail = 50.0 * (1.0 - probability)
    names = {
        "alpha_chem": "Initial level (logit)",
        "loglambda_chem": "Log rate (per 1,000 cycles)",
        "amplitude_chem": "Amplitude (logit units)",
    }
    rows = []
    for variable, quantity in names.items():
        stacked = idata.posterior[variable].stack(sample=("chain", "draw"))
        for chemistry in stacked["chemistry"].to_numpy():
            values = stacked.sel(chemistry=chemistry).to_numpy()
            lower, upper = np.percentile(values, [tail, 100.0 - tail])
            rows.append(
                {"quantity": quantity, "chemistry": chemistry, "mean": values.mean(), "lower": lower, "upper": upper}
            )
    return pd.DataFrame(rows)


def cell_estimates(idata: az.InferenceData, variable: str, log: bool = False) -> pd.DataFrame:
    """Posterior mean of a per-cell parameter.

    Args:
        idata: Inference data of a model with a per-cell parameter.
        variable: Name of the parameter, for example ``"alpha_cell"``.
        log: Whether to return the logarithm of the parameter.

    Returns:
        One row per cell, with ``cell`` and ``mean``.
    """
    draws = idata.posterior[variable]
    if log:
        draws = np.log(draws)
    mean = draws.mean(("chain", "draw"))
    return pd.DataFrame({"cell": mean["cell"].to_numpy(), "mean": mean.to_numpy()})


def shrinkage_frames(
    none: az.InferenceData, partial: az.InferenceData, chemistry_of: pd.Series, cells: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-cell estimates of the initial level and log rate without and with partial pooling, and chemistry means.

    Args:
        none: Inference data of the no-pooling model.
        partial: Inference data of the partial-pooling model.
        chemistry_of: Chemistry of each training cell, indexed by cell.
        cells: Cells in the order of the horizontal axis of the figure.

    Returns:
        The estimates, one row per parameter, method and cell, and the chemistry means with the span of their cells.
    """
    parameters = {
        "Initial level (logit)": ("alpha_cell", False, "alpha_chem"),
        "Log rate (per 1,000 cycles)": ("lambda_cell", True, "loglambda_chem"),
    }
    estimates, means = [], []
    for title, (variable, log, level) in parameters.items():
        for method, idata in (("No pooling", none), ("Partial pooling", partial)):
            estimates.append(cell_estimates(idata, variable, log).assign(parameter=title, method=method))
        chemistry_mean = partial.posterior[level].mean(("chain", "draw"))
        for chemistry in chemistry_mean["chemistry"].to_numpy():
            members = [cells.index(cell) + 1 for cell in chemistry_of[chemistry_of == chemistry].index if cell in cells]
            means.append(
                {
                    "parameter": title,
                    "chemistry": chemistry,
                    "x0": min(members) - 0.4,
                    "x1": max(members) + 0.4,
                    "value": float(chemistry_mean.sel(chemistry=chemistry)),
                }
            )
    frame = pd.concat(estimates, ignore_index=True)
    frame["chemistry"] = frame["cell"].map(chemistry_of)
    return frame, pd.DataFrame(means)


PRIOR_LEVELS = {
    "alpha_chem": "Initial level (logit)",
    "loglambda_chem": "Log rate (per 1,000 cycles)",
    "amplitude_chem": "Amplitude (logit units)",
}


def prior_posterior(
    prior: az.InferenceData, posterior: az.InferenceData, draws: int = 4000, seed: int = 0
) -> pd.DataFrame:
    """Draws of each chemistry parameter and of the precision from the prior and from the posterior, to be compared.

    The prior of a chemistry parameter is the same for every chemistry, so its draws are pooled; the posterior has one
    set of draws per chemistry. The precision has one posterior, shown on the logarithmic scale because it lies far
    from its prior.

    Args:
        prior: Prior draws of a partial-pooling model, from ``pymc.sample_prior_predictive``.
        posterior: Inference data of the partial-pooling fit.
        draws: Draws kept for each source, so that the frame stays small.
        seed: Seed of the thinning.

    Returns:
        Rows with ``quantity``, ``source`` (``"Prior"``, a chemistry, or ``"Posterior"`` for the precision) and
        ``value``.
    """
    generator = np.random.default_rng(seed)

    def thin(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values).ravel()
        return generator.choice(values, size=min(draws, values.size), replace=False)

    rows = []
    for variable, quantity in PRIOR_LEVELS.items():
        rows.append(pd.DataFrame({"quantity": quantity, "source": "Prior", "value": thin(prior.prior[variable])}))
        for chemistry in posterior.posterior[variable]["chemistry"].to_numpy():
            values = thin(posterior.posterior[variable].sel(chemistry=chemistry))
            rows.append(pd.DataFrame({"quantity": quantity, "source": str(chemistry), "value": values}))
    precision = "Precision (log10)"
    rows.append(pd.DataFrame({"quantity": precision, "source": "Prior", "value": np.log10(thin(prior.prior["phi"]))}))
    rows.append(
        pd.DataFrame(
            {"quantity": precision, "source": "Posterior", "value": np.log10(thin(posterior.posterior["phi"]))}
        )
    )
    return pd.concat(rows, ignore_index=True)


def prior_sensitivity(fits: dict[str, az.InferenceData], probability: float = 0.9) -> pd.DataFrame:
    """Chemistry parameters of fits that differ only in a prior, with the probability that one rate exceeds the other.

    Args:
        fits: Partial-pooling fits keyed by the name of the prior setting, in the order to be shown.
        probability: Probability mass of the interval.

    Returns:
        The rows of ``chemistry_intervals`` for every setting, with a ``setting`` column, and for each setting the
        posterior mean of the precision in ``precision``.
    """
    frames = []
    for setting, idata in fits.items():
        precision = float(idata.posterior["phi"].mean())
        frames.append(chemistry_intervals(idata, probability).assign(setting=setting, precision=precision))
    return pd.concat(frames, ignore_index=True)


def loo_by_cell(fits: dict[str, az.InferenceData], cells: np.ndarray) -> pd.DataFrame:
    """Expected log predictive density of each cell under leave-one-out cross-validation, for every model.

    Leave-one-out cross-validation scores each observation by the log density the model gives it when that single
    observation is left out of the fit (Vehtari, Gelman and Gabry, 2017). Summing the scores of the observations of
    one cell shows which cells a model predicts well *at the cycles it has not been shown*, which is a statement about
    cells that are in the data, and not about a cell the model has never seen.

    Args:
        fits: Fits with the log likelihood stored, keyed by model name.
        cells: Cell of each observation, in the order of the observations of the fits.

    Returns:
        Rows with ``method``, ``cell``, ``elpd`` (the summed pointwise score) and ``bad`` (observations with a Pareto
        shape above 0.7, for which the estimate is unreliable).

    Raises:
        ValueError: If the length of ``cells`` is not the number of observations of a fit.
    """
    cells = np.asarray(cells)
    rows = []
    for method, idata in fits.items():
        result = az.loo(idata, pointwise=True)
        scores = np.asarray(result.loo_i)
        if scores.shape[0] != cells.shape[0]:
            raise ValueError("there must be one cell for each observation")
        shapes = np.asarray(result.pareto_k)
        frame = pd.DataFrame({"cell": cells, "elpd": scores, "bad": shapes > 0.7})
        summed = frame.groupby("cell", sort=False).agg(elpd=("elpd", "sum"), bad=("bad", "sum")).reset_index()
        rows.append(summed.assign(method=method))
    return pd.concat(rows, ignore_index=True)


def chemistry_draws(idata: az.InferenceData, draws: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Posterior draws of the three chemistry parameters of a partial-pooling fit, to be drawn as densities.

    Args:
        idata: Inference data of a partial-pooling model.
        draws: Draws kept for each parameter and chemistry.
        seed: Seed of the thinning.

    Returns:
        Rows with ``quantity``, ``chemistry`` and ``value``.
    """
    frames = [
        sensitivity_draws({"posterior": idata}, variable, draws, seed).drop(columns="setting").assign(quantity=quantity)
        for variable, quantity in PRIOR_LEVELS.items()
    ]
    return pd.concat(frames, ignore_index=True)


def sensitivity_draws(
    fits: dict[str, az.InferenceData], variable: str = "loglambda_chem", draws: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Posterior draws of a chemistry parameter under each setting, to be drawn over one another.

    Args:
        fits: Partial-pooling fits keyed by the name of the prior setting, in the order to be shown.
        variable: Name of the chemistry parameter, for example ``"loglambda_chem"``.
        draws: Draws kept for each setting and chemistry.
        seed: Seed of the thinning.

    Returns:
        Rows with ``setting``, ``chemistry`` and ``value``.
    """
    generator = np.random.default_rng(seed)
    rows = []
    for setting, idata in fits.items():
        stacked = idata.posterior[variable].stack(sample=("chain", "draw"))
        for chemistry in stacked["chemistry"].to_numpy():
            values = stacked.sel(chemistry=chemistry).to_numpy()
            kept = generator.choice(values, size=min(draws, values.size), replace=False)
            rows.append(pd.DataFrame({"setting": setting, "chemistry": str(chemistry), "value": kept}))
    return pd.concat(rows, ignore_index=True)
