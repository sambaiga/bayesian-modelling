import numpy as np
import pymc as pm
import pytest

from bayes import lifetime


def test_exposure_counts_a_failed_machine_only_until_its_day_of_failure():
    years = lifetime.exposure_years(machines=12, days=91, failure_days=[40])
    assert years == pytest.approx((11 * 91 + 40) / 365.25)


def test_exposure_without_failures_is_machines_times_days():
    assert lifetime.exposure_years(10, 365, []) == pytest.approx(10 * 365 / 365.25)


def test_exposure_rejects_more_failures_than_machines_and_a_failure_outside_the_pilot():
    with pytest.raises(ValueError, match="more failures"):
        lifetime.exposure_years(1, 10, [1, 2])
    with pytest.raises(ValueError, match="outside"):
        lifetime.exposure_years(3, 10, [11])


def test_gamma_posterior_adds_failures_to_the_shape_and_exposure_to_the_rate():
    assert lifetime.gamma_posterior(4.0, 40.0, 1, 2.85) == (5.0, 42.85)


def test_gamma_posterior_rejects_a_bad_prior_and_negative_data():
    with pytest.raises(ValueError):
        lifetime.gamma_posterior(0.0, 1.0, 0, 1.0)
    with pytest.raises(ValueError):
        lifetime.gamma_posterior(1.0, 1.0, -1, 1.0)


def test_the_sampler_agrees_with_the_conjugate_posterior():
    shape, rate, failures, exposure = 4.0, 40.0, 1, 2.85
    model = lifetime.failure_model(shape, rate, failures, exposure)
    with model:
        idata = pm.sample(2000, tune=500, chains=2, cores=1, random_seed=1, progressbar=False)
    draws = idata.posterior["theta"].to_numpy().ravel()
    post_shape, post_rate = lifetime.gamma_posterior(shape, rate, failures, exposure)
    assert draws.mean() == pytest.approx(post_shape / post_rate, rel=0.03)
    assert draws.std() == pytest.approx(np.sqrt(post_shape) / post_rate, rel=0.06)


def test_failure_probability_matches_its_closed_form_for_a_gamma_rate():
    theta = np.random.default_rng(0).gamma(5.0, 1.0 / 42.85, 400000)
    assert lifetime.failure_probability(theta, 3.0) == pytest.approx(
        lifetime.exact_probability(5.0, 42.85, 3.0), abs=2e-3
    )


def test_survival_starts_at_one_and_falls_with_time():
    theta = np.random.default_rng(1).gamma(5.0, 1.0 / 42.85, 5000)
    curve = lifetime.survival(theta, np.array([0.0, 1.0, 5.0, 10.0]))
    assert curve[0] == 1.0 and np.all(np.diff(curve) < 0)


def test_lifetime_interval_is_ordered_and_matches_a_constant_rate():
    low, mid, high = lifetime.lifetime_interval(np.full(10, 0.1))
    assert low == mid == high == pytest.approx(10.0)
    low, mid, high = lifetime.lifetime_interval(np.random.default_rng(2).gamma(5.0, 1.0 / 42.85, 5000))
    assert low < mid < high


def test_contract_worth_is_the_expected_cost_and_the_share_of_draws_that_exceed_the_price():
    theta = np.array([0.05, 0.10, 0.20, 0.40])
    cost, probability = lifetime.contract_worth(theta, years=3.0, repair=60.0, price=20.0)
    assert cost == pytest.approx(60.0 * 3.0 * theta.mean())
    assert probability == 0.5  # 60 * 3 * theta > 20 only for the rates 0.2 and 0.4, those above 0.111


def test_prior_failures_puts_most_mass_on_few_failures_for_a_small_rate():
    shares = lifetime.prior_failures(4.0, 400.0, 2.85, 5, np.random.default_rng(3))
    assert shares.shape == (6,) and shares.sum() <= 1.0 and shares[0] > 0.9


def test_exact_probability_is_zero_at_time_zero_and_approaches_one():
    assert lifetime.exact_probability(5.0, 42.85, 0.0) == 0.0
    assert lifetime.exact_probability(5.0, 42.85, 1e6) > 0.999
