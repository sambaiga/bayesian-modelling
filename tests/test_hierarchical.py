import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from sklearn.preprocessing import StandardScaler
import xarray as xr

from bayes.regression import hierarchical as hier

FEATURES = ["a", "b"]


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    generator = np.random.default_rng(0)
    rows = []
    for chemistry, level, rate in (("X", 2.5, 1.0), ("Y", 1.2, 3.0)):
        for index in range(3):
            cycle = np.linspace(10, 900, 25)
            eta = level + 0.1 * index - 0.8 * (1 - np.exp(-rate * cycle / 1000.0))
            soh = 100 * hier.CEILING / (1 + np.exp(-eta)) * (1 + 0.01 * generator.standard_normal(25))
            rows.append(
                pd.DataFrame(
                    {
                        "cell": f"{chemistry}{index}",
                        "CellType": chemistry,
                        "cycle": cycle,
                        "soh": soh,
                        "a": generator.standard_normal(25),
                        "b": generator.standard_normal(25),
                    }
                )
            )
    data = pd.concat(rows, ignore_index=True)
    data["y"] = hier.state_of_health(data["soh"])
    return data


@pytest.fixture(scope="module")
def scaler(frame: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(frame[FEATURES])


def posterior(**variables) -> az.InferenceData:
    """Inference data with chain and draw dimensions, for the tests of the posterior summaries."""
    data = {}
    for name, (dims, values) in variables.items():
        values = np.asarray(values, dtype=float)
        data[name] = (("chain", "draw", *dims), np.broadcast_to(values, (2, 50, *values.shape)).copy())
    coords = {"chemistry": ["X", "Y"], "cell": ["X0", "Y0"], "feature": FEATURES}
    return az.InferenceData(posterior=xr.Dataset(data, coords=coords))


def test_state_of_health_stays_inside_the_open_unit_interval():
    scaled = hier.state_of_health(np.array([0.0, 50.0, 100.0, 104.6, 120.0]))
    assert (scaled > 0).all() and (scaled < 1).all()
    assert scaled[3] == pytest.approx(1.046 / hier.CEILING)


def test_the_three_levels_of_pooling_have_their_own_parameters(frame, scaler):
    names = {
        pooling: {v.name for v in hier.hierarchical_model(frame, scaler, FEATURES, pooling).free_RVs}
        for pooling in hier.POOLING
    }
    assert {"alpha_chem", "z_alpha", "sigma_cell_alpha"} <= names["partial"]
    assert {"alpha", "loglambda", "amplitude"} <= names["complete"] and "alpha_cell" not in names["complete"]
    assert {"alpha_cell", "loglambda_cell"} <= names["none"] and "alpha_chem" not in names["none"]


def test_cell_and_chemistry_dimensions_follow_the_data(frame, scaler):
    model = hier.hierarchical_model(frame, scaler, FEATURES, "partial")
    assert list(model.coords["cell"]) == ["X0", "X1", "X2", "Y0", "Y1", "Y2"]
    assert list(model.coords["chemistry"]) == ["X", "Y"]


def test_the_control_without_diagnostics_has_no_coefficients(frame, scaler):
    model = hier.hierarchical_model(frame, scaler, FEATURES, "complete", diagnostics=False)
    assert "beta" not in {v.name for v in model.free_RVs}


def test_an_unknown_level_of_pooling_is_rejected(frame, scaler):
    with pytest.raises(ValueError):
        hier.hierarchical_model(frame, scaler, FEATURES, "shrunk")


def test_the_prior_predictive_of_every_variant_is_inside_the_unit_interval(frame, scaler):
    for pooling in hier.POOLING:
        model = hier.hierarchical_model(frame, scaler, FEATURES, pooling)
        with model:
            prior = pm.sample_prior_predictive(draws=40, random_seed=1)
        values = prior.prior_predictive["y"].to_numpy()
        # A Beta draw can reach exactly 0 or 1 in floating point for extreme prior values, so the bounds are closed.
        assert (values >= 0).all() and (values <= 1).all() and np.isfinite(values).all()


def test_a_new_cell_under_partial_pooling_is_centred_on_its_chemistry():
    idata = posterior(
        alpha_chem=(("chemistry",), [2.0, 1.0]),
        loglambda_chem=(("chemistry",), [0.5, -0.5]),
        amplitude_chem=(("chemistry",), [1.0, 3.0]),
        sigma_cell_alpha=(("chemistry",), [1e-9, 1e-9]),
        sigma_cell_lambda=(("chemistry",), [1e-9, 1e-9]),
        phi=((), 100.0),
        beta=(("feature",), [0.1, 0.2]),
    )
    generator = np.random.default_rng(0)
    pars = hier.cell_parameters(idata, "partial", 20, generator, "Y", None)
    assert (
        np.allclose(pars["alpha"], 1.0)
        and np.allclose(pars["amplitude"], 3.0)
        and np.allclose(pars["rate"], np.exp(-0.5))
    )


def test_a_fitted_cell_uses_its_own_draws():
    idata = posterior(
        alpha_cell=(("cell",), [2.0, 1.0]),
        lambda_cell=(("cell",), [1.5, 2.5]),
        amplitude_cell=(("cell",), [1.0, 3.0]),
        phi=((), 100.0),
        beta=(("feature",), [0.1, 0.2]),
    )
    pars = hier.cell_parameters(idata, "none", 10, np.random.default_rng(0), "X", "Y0")
    assert np.allclose(pars["alpha"], 1.0) and np.allclose(pars["rate"], 2.5)


def test_a_new_cell_without_pooling_comes_from_the_prior_and_not_from_the_data():
    idata = posterior(
        alpha_cell=(("cell",), [9.0, 9.0]),
        lambda_cell=(("cell",), [9.0, 9.0]),
        amplitude_cell=(("cell",), [9.0, 9.0]),
        phi=((), 100.0),
    )
    pars = hier.cell_parameters(idata, "none", 50, np.random.default_rng(0), "X", None)
    assert abs(pars["alpha"].mean() - 2.0) < 1.0 and pars["amplitude"].max() < 5.0


def test_complete_pooling_gives_every_cell_the_common_parameters():
    idata = posterior(alpha=((), 1.7), loglambda=((), 0.2), amplitude=((), 0.9), phi=((), 100.0))
    new = hier.cell_parameters(idata, "complete", 10, np.random.default_rng(0), "X", None)
    known = hier.cell_parameters(idata, "complete", 10, np.random.default_rng(0), "X", "X0")
    assert np.allclose(new["alpha"], 1.7) and np.allclose(known["alpha"], 1.7)


def test_the_chemistry_intervals_return_the_known_mean_and_bounds():
    idata = posterior(
        alpha_chem=(("chemistry",), [2.0, 1.0]),
        loglambda_chem=(("chemistry",), [0.5, -0.5]),
        amplitude_chem=(("chemistry",), [1.0, 3.0]),
    )
    table = hier.chemistry_intervals(idata)
    row = table[(table["quantity"] == "Initial level (logit)") & (table["chemistry"] == "Y")].iloc[0]
    assert (
        row["mean"] == pytest.approx(1.0) and row["lower"] == pytest.approx(1.0) and row["upper"] == pytest.approx(1.0)
    )
    assert len(table) == 6


def test_cell_estimates_can_return_the_logarithm():
    idata = posterior(lambda_cell=(("cell",), [np.e, np.e**2]))
    assert hier.cell_estimates(idata, "lambda_cell", log=True)["mean"].round(6).tolist() == [1.0, 2.0]


@pytest.fixture(scope="module")
def fitted(frame, scaler, tmp_path_factory):
    path = tmp_path_factory.mktemp("fit") / "partial.nc"
    model = hier.hierarchical_model(frame, scaler, FEATURES, "partial")
    return path, hier.sample(model, path, draws=30, tune=30, chains=2, seed=7)


def test_a_fit_is_stored_and_read_back_without_sampling_again(fitted, frame, scaler):
    path, idata = fitted
    assert path.exists()
    again = hier.sample(
        hier.hierarchical_model(frame, scaler, FEATURES, "partial"), path, draws=30, tune=30, chains=2, seed=99
    )
    assert np.allclose(again.posterior["phi"].to_numpy(), idata.posterior["phi"].to_numpy())


def test_convergence_reports_the_four_diagnostics(fitted):
    report = hier.convergence(fitted[1])
    assert set(report) == {"divergences", "r_hat", "ess_bulk", "ess_tail"} and report["r_hat"] > 0


def test_predictions_have_one_row_per_draw_and_stay_inside_the_state_of_health_range(fitted, frame, scaler):
    rows = frame[frame["cell"] == "X0"]
    for known in (True, False):
        draws = hier.predict(fitted[1], "partial", rows, scaler, FEATURES, "X", seed=1, known=known, draws=20)
        assert draws.shape == (20, len(rows))
        assert (draws > 0).all() and (draws < 100 * hier.CEILING).all()


def test_a_prediction_repeats_with_the_same_seed_and_differs_with_another(fitted, frame, scaler):
    rows = frame[frame["cell"] == "Y1"]
    first = hier.predict(fitted[1], "partial", rows, scaler, FEATURES, "Y", seed=3, draws=15)
    second = hier.predict(fitted[1], "partial", rows, scaler, FEATURES, "Y", seed=3, draws=15)
    other = hier.predict(fitted[1], "partial", rows, scaler, FEATURES, "Y", seed=4, draws=15)
    assert np.array_equal(first, second) and not np.array_equal(first, other)


def test_the_prior_of_the_precision_is_an_argument(frame, scaler):
    tight = hier.hierarchical_model(frame, scaler, FEATURES, "partial", phi_prior=(float(np.log(100.0)), 0.3))
    wide = hier.hierarchical_model(frame, scaler, FEATURES, "partial", phi_prior=(float(np.log(1000.0)), 1.5))
    with tight:
        low = pm.draw(tight["phi"], draws=2000, random_seed=1)
    with wide:
        high = pm.draw(wide["phi"], draws=2000, random_seed=1)
    assert np.median(low) == pytest.approx(100.0, rel=0.15) and np.median(high) == pytest.approx(1000.0, rel=0.2)
    assert high.std() > low.std()


def test_prior_posterior_gives_the_prior_once_and_the_posterior_for_each_chemistry():
    levels = {name: (("chemistry",), [1.0, 2.0]) for name in hier.PRIOR_LEVELS}
    fit = posterior(**levels, phi=((), 1500.0))
    prior = az.InferenceData(
        prior=xr.Dataset(
            {
                **{name: (("chain", "draw", "chemistry"), np.zeros((1, 100, 2))) for name in hier.PRIOR_LEVELS},
                "phi": (("chain", "draw"), np.full((1, 100), 100.0)),
            },
            coords={"chemistry": ["X", "Y"]},
        )
    )
    frame = hier.prior_posterior(prior, fit, draws=40)
    assert set(frame["quantity"]) == {*hier.PRIOR_LEVELS.values(), "Precision (log10)"}
    sources = frame[frame["quantity"] == "Initial level (logit)"]["source"].unique()
    assert set(sources) == {"Prior", "X", "Y"}
    precision = frame[frame["quantity"] == "Precision (log10)"].groupby("source")["value"].mean().round(2)
    assert precision.to_dict() == {"Prior": 2.0, "Posterior": 3.18}
    assert frame.groupby(["quantity", "source"]).size().max() == 40


def test_prior_sensitivity_has_a_row_set_for_each_setting_and_the_precision_of_the_fit():
    levels = {name: (("chemistry",), [1.0, 2.0]) for name in hier.PRIOR_LEVELS}
    fits = {"Narrow": posterior(**levels, phi=((), 100.0)), "Wide": posterior(**levels, phi=((), 900.0))}
    table = hier.prior_sensitivity(fits)
    assert list(table["setting"].unique()) == ["Narrow", "Wide"] and len(table) == 2 * 3 * 2
    assert table.groupby("setting")["precision"].first().tolist() == [100.0, 900.0]


def test_sensitivity_draws_keep_the_setting_and_the_chemistry_of_each_draw():
    levels = {name: (("chemistry",), [1.0, 2.0]) for name in hier.PRIOR_LEVELS}
    fits = {"Narrow": posterior(**levels, phi=((), 100.0)), "Wide": posterior(**levels, phi=((), 900.0))}
    draws = hier.sensitivity_draws(fits, "alpha_chem", draws=30)
    assert set(draws["setting"]) == {"Narrow", "Wide"} and set(draws["chemistry"]) == {"X", "Y"}
    assert draws.groupby(["setting", "chemistry"]).size().eq(30).all()
    assert draws.groupby("chemistry")["value"].mean().round(2).to_dict() == {"X": 1.0, "Y": 2.0}
