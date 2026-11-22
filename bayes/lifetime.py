"""A constant failure rate learned from a pilot: the Gamma prior, the Poisson likelihood and the decision they inform.

A pilot runs identical machines for a fixed time and records the failures. With a constant rate of failure ``theta``
per machine-year, the number of failures in ``exposure`` machine-years is Poisson with mean ``theta * exposure``, and
a Gamma prior on ``theta`` is conjugate: the posterior is Gamma again, which gives an exact answer to check the sampler
against. The lifetime of a machine is exponential with mean ``1 / theta``.
"""

from __future__ import annotations

import numpy as np
import pymc as pm

DAYS_PER_YEAR = 365.25


def exposure_years(machines: int, days: int, failure_days: list[int]) -> float:
    """Machine-years of operation of a pilot in which the failed machines stop at their day of failure.

    Args:
        machines: Number of machines at the start of the pilot.
        days: Length of the pilot in days.
        failure_days: Day of each failure, within the pilot.

    Returns:
        The total time that the machines ran, in years.

    Raises:
        ValueError: If a failure falls outside the pilot or there are more failures than machines.
    """
    if len(failure_days) > machines:
        raise ValueError("there are more failures than machines")
    if any(day < 0 or day > days for day in failure_days):
        raise ValueError("a failure falls outside the pilot")
    running = machines - len(failure_days)
    return (running * days + sum(failure_days)) / DAYS_PER_YEAR


def gamma_posterior(shape: float, rate: float, failures: int, exposure: float) -> tuple[float, float]:
    """Shape and rate of the Gamma posterior of the failure rate, from the conjugate update.

    Args:
        shape: Shape of the Gamma prior.
        rate: Rate of the Gamma prior, in machine-years.
        failures: Number of failures in the pilot.
        exposure: Machine-years of the pilot.

    Returns:
        The shape and rate of the posterior.

    Raises:
        ValueError: If a parameter is not positive, or the failures or the exposure are negative.
    """
    if shape <= 0 or rate <= 0:
        raise ValueError("the prior shape and rate must be positive")
    if failures < 0 or exposure < 0:
        raise ValueError("failures and exposure cannot be negative")
    return shape + failures, rate + exposure


def failure_model(shape: float, rate: float, failures: int, exposure: float) -> pm.Model:
    """The Gamma-Poisson model of the pilot, for the sampler.

    Args:
        shape: Shape of the Gamma prior.
        rate: Rate of the Gamma prior, in machine-years.
        failures: Number of failures in the pilot.
        exposure: Machine-years of the pilot.

    Returns:
        The PyMC model, with the rate ``theta`` and the observed failures.
    """
    with pm.Model() as model:
        theta = pm.Gamma("theta", alpha=shape, beta=rate)
        pm.Poisson("failures", mu=theta * exposure, observed=failures)
    return model


def failure_probability(theta: np.ndarray, years: float) -> float:
    """Probability that a machine fails within ``years``, averaged over the draws of the rate.

    Args:
        theta: Draws of the failure rate per machine-year.
        years: Time of interest in years.

    Returns:
        The posterior predictive probability ``mean(1 - exp(-theta * years))``.
    """
    return float(np.mean(-np.expm1(-np.asarray(theta) * years)))


def survival(theta: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Probability that a machine survives each time in ``years``, averaged over the draws of the rate.

    Args:
        theta: Draws of the failure rate per machine-year.
        years: Times in years.

    Returns:
        The posterior predictive survival curve, one value for each time.
    """
    return np.exp(-np.outer(np.asarray(years, dtype=float), np.asarray(theta))).mean(axis=1)


def lifetime_interval(theta: np.ndarray, level: float = 0.9) -> tuple[float, float, float]:
    """Median and central interval of the mean lifetime ``1 / theta``, in years.

    Args:
        theta: Draws of the failure rate per machine-year.
        level: Probability of the interval.

    Returns:
        The lower end, the median and the upper end.
    """
    lifetime = 1.0 / np.asarray(theta)
    lower, median, upper = np.quantile(lifetime, [(1 - level) / 2, 0.5, (1 + level) / 2])
    return float(lower), float(median), float(upper)


def contract_worth(theta: np.ndarray, years: float, repair: float, price: float) -> tuple[float, float]:
    """Expected cost of repairs over the contract period and the probability that they exceed the price.

    Failures of a repaired machine are taken to follow the same constant rate, so the expected number of repairs in
    ``years`` is ``theta * years``.

    Args:
        theta: Draws of the failure rate per machine-year.
        years: Length of the contract in years.
        repair: Cost of one repair.
        price: Price of the service contract for one machine.

    Returns:
        The expected cost of repairs without a contract, and the probability that it exceeds ``price``.
    """
    cost = repair * years * np.asarray(theta)
    return float(cost.mean()), float(np.mean(cost > price))


def prior_failures(
    shape: float, rate: float, exposure: float, upper: int, generator: np.random.Generator, draws: int = 20000
) -> np.ndarray:
    """Share of prior predictive draws for each number of failures in a pilot, from zero to ``upper``.

    Args:
        shape: Shape of the Gamma prior.
        rate: Rate of the Gamma prior.
        exposure: Machine-years of the pilot.
        upper: Largest number of failures to report.
        generator: Source of the draws.
        draws: Number of prior predictive draws.

    Returns:
        Array of length ``upper + 1`` with the share of draws for each count; larger counts are not reported.
    """
    theta = generator.gamma(shape, 1.0 / rate, draws)
    counts = generator.poisson(theta * exposure)
    return np.bincount(counts[counts <= upper], minlength=upper + 1) / draws


def exact_probability(shape: float, rate: float, years: float) -> float:
    """Exact probability of failure within ``years`` under a Gamma posterior, from its closed form.

    The mean of ``1 - exp(-theta * years)`` over a Gamma(shape, rate) is ``1 - (rate / (rate + years)) ** shape``.

    Args:
        shape: Posterior shape.
        rate: Posterior rate.
        years: Time of interest in years.

    Returns:
        The probability of failing within ``years``.
    """
    return float(1.0 - (rate / (rate + years)) ** shape)
