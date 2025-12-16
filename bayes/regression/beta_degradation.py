import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.preprocessing import StandardScaler


def beta_regression_model(
    data: pd.DataFrame,
    features: list[str],
    target: str = "capacity",
    scaler: StandardScaler | None = None,
    lower_bound: float = 0.2,
    upper_bound: float = 1.3,
    eps: float = 1e-8,
) -> tuple[pm.Model, StandardScaler]:
    """Beta regression model for bounded battery capacity data using PyMC.

    Capacity (SoH) is scaled to the (0, 1) interval for the Beta distribution.

    Args:
        data: DataFrame containing 'capacity', 'cycle', and feature columns.
        features: List of column names used as predictors (X variables).
        target: Name of the capacity column.
        scaler: Pre-fitted StandardScaler object, or None to fit a new one.
        lower_bound: Physical lower bound for capacity (for scaling).
        upper_bound: Physical upper bound for capacity (for scaling).
        eps: Small value to avoid boundary issues in Beta distribution.

    Returns:
        A tuple containing the PyMC model and the fitted/provided StandardScaler.
    """
    # 1. Prepare Features (X)
    if scaler is None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(data[features])
    else:
        x_scaled = scaler.transform(data[features])

    # 2. Prepare Targets (Y)
    y = data[target].values.astype(np.float64)
    cycles = data["cycle"].values.astype(np.float64)
    n_features = len(features)

    # Transform y to (0,1) interval and clip to avoid boundaries (0 or 1)
    y_scaled = (y - lower_bound) / (upper_bound - lower_bound)
    y_scaled = np.clip(y_scaled, eps, 1 - eps)

    with pm.Model() as model:
        # Data Containers
        x_data = pm.Data("x_data", x_scaled)
        cycle_data = pm.Data("cycle_data", cycles)
        y_data = pm.Data("y_data", y_scaled)

        # Priors
        initial_logit_capacity_mean = -np.log(1 - 1e-6)
        intercept = pm.Normal("intercept", mu=initial_logit_capacity_mean, sigma=0.5)
        lambda_rate = pm.Lognormal("lambda_rate", mu=np.log(0.005), sigma=0.5)
        beta = pm.Normal("beta", mu=0, sigma=0.2, shape=n_features)
        phi = pm.Gamma("phi", alpha=100, beta=2.0)
        degr_amp = pm.HalfNormal("degr_amp", sigma=0.1)

        # Linear predictor (eta) on logit scale
        degradation = pm.math.exp(-lambda_rate * cycle_data)
        degradation_term = -degr_amp * (1 - degradation)
        logit_mu = intercept + degradation_term + pm.math.dot(x_data, beta)

        # Convert to probability scale (0,1)
        mu_scaled = pm.Deterministic("mu_scaled", pm.math.invlogit(logit_mu))

        # Beta likelihood
        alpha = mu_scaled * phi
        beta_shape = (1 - mu_scaled) * phi
        pm.Beta("y_obs", alpha=alpha, beta=beta_shape, observed=y_data)

        # Transform mu back to original scale
        mu_original = pm.Deterministic("mu_original", mu_scaled * (upper_bound - lower_bound) + lower_bound)

        pm.Deterministic("capacity_pred", mu_original)
        pm.Deterministic("feature_effects", beta)

    return model, scaler


def _rescale(arr: np.ndarray, upper_bound: float, lower_bound: float) -> np.ndarray:
    return arr * (upper_bound - lower_bound) + lower_bound


def get_posterior_predictions(
    idata,
    model,
    scaler,
    data: "pd.DataFrame",
    features: list[str],
    *,
    lower_bound: float = 0.2,
    upper_bound: float = 1.3,
    alpha: float = 0.1,
) -> "pd.DataFrame":
    """Generate posterior predictions with HDI for new data using a fitted PyMC model.

    The target variable is assumed to have been scaled to [0, 1] during training,
    so predictions are rescaled back to the original bounds.

    Args:
        idata: InferenceData object containing posterior samples.
        model: The PyMC model used for fitting.
        scaler: Fitted scaler (e.g., MinMaxScaler) used on training features.
        data: DataFrame containing new observations with feature columns and 'cycle'.
        features: List of feature column names used in the model.
        lower_bound: Original lower bound of the target variable (default: 0.2).
        upper_bound: Original upper bound of the target variable (default: 1.3).
        alpha: Significance level. The HDI will be (1 - alpha) * 100%.
               Example: alpha=0.05 → 95% HDI (default).

    Returns:
        Copy of input DataFrame with added prediction columns:
        - pred_mean
        - pred_median
        - hdi_low (2.5% HDI)
        - hdi_high (97.5% HDI)
    """
    x_new = scaler.transform(data[features])
    cycles_new = data["cycle"].values
    n_new = len(x_new)
    y_dummy_scaled = np.full(n_new, 0.5)

    pred_df = data.copy()
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 (exclusive)")

    hdi_prob = 1.0 - alpha

    with model:
        pm.set_data(
            {
                "x_data": x_new,
                "cycle_data": cycles_new,
                "y_data": y_dummy_scaled,
            }
        )
        post_pred = pm.sample_posterior_predictive(
            idata,
            var_names=["y_obs"],
            random_seed=42,
            predictions=True,
        )

    # Extract predictions and compute 1-alpha% HDI
    y_pred_scaled = post_pred.predictions["y_obs"]
    hdi_scaled = az.hdi(y_pred_scaled, hdi_prob=hdi_prob)["y_obs"].values

    pred_scaled = y_pred_scaled.stack(sample=("chain", "draw")).values
    pred = _rescale(pred_scaled.T, upper_bound, lower_bound)

    hdi = _rescale(hdi_scaled, upper_bound, lower_bound)
    hdi_low, hdi_high = hdi[:, 0], hdi[:, 1]

    pred_mean = pred.mean(axis=0)
    pred_median = np.percentile(pred, 50, axis=0)

    pred_df["pred_mean"] = pred_mean
    pred_df["pred_median"] = pred_median
    pred_df["hdi_low"] = hdi_low
    pred_df["hdi_high"] = hdi_high

    return pred_df
