"""The priors of the studies as tables, written once so that a post and the model code describe the same thing.

The values mirror ``bayes.regression.hierarchical.hierarchical_model`` and ``bayes.lifetime``; a test compares them
with the model so that the description cannot drift from the code.
"""

from __future__ import annotations

import pandas as pd


def complete_pooling() -> pd.DataFrame:
    """The priors of the single-level Beta regression of Part 2.

    Returns:
        Rows with ``Parameter``, ``Prior`` and ``Role``.
    """
    return pd.DataFrame(
        [
            ("Initial level α", "Normal(2, 1)", "Logit of the state of health at cycle zero"),
            ("Log rate ln λ", "Normal(ln 5, 0.5²)", "Rate of the decay, per thousand cycles"),
            ("Amplitude A", "HalfNormal(1)", "Size of the saturating decay on the logit scale"),
            ("Effects β", "Normal(0, 0.2²)", "Effect of each standardised diagnostic"),
            ("Precision φ", "LogNormal(ln 100, 1)", "Noise of the Beta likelihood"),
        ],
        columns=["Parameter", "Prior", "Role"],
    )


def hierarchical() -> pd.DataFrame:
    """The priors of the hierarchical Beta regression of Part 3.

    Returns:
        Rows with ``Parameter``, ``Prior`` and ``Role``.
    """
    return pd.DataFrame(
        [
            ("Fleet level of α", "Normal(2, 1)", "Mean initial level across chemistries"),
            ("Spread of α between chemistries", "HalfNormal(1)", "How far chemistries differ in initial level"),
            ("Fleet level of ln λ", "Normal(ln 5, 0.5²)", "Mean log rate across chemistries"),
            ("Spread of ln λ between chemistries", "HalfNormal(0.5)", "How far chemistries differ in rate"),
            ("Amplitude scale", "HalfNormal(1)", "Scale of the chemistry amplitudes"),
            ("Amplitude A of a chemistry", "HalfNormal(scale)", "Size of the decay of a chemistry"),
            ("Spread of α, ln λ between cells", "HalfNormal(0.5)", "How far a cell departs from its chemistry"),
            ("Effects β", "Normal(0, 0.2²)", "Effect of each standardised diagnostic"),
            ("Precision φ", "LogNormal(ln 100, 1)", "Noise of the Beta likelihood"),
        ],
        columns=["Parameter", "Prior", "Role"],
    )


def failure_rate(priors: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """The Gamma priors of the failure rate of Part 1.

    Args:
        priors: For each name, the shape and the rate of the Gamma prior, the rate in machine-years.

    Returns:
        Rows with ``Prior``, ``Gamma(shape, rate)``, ``Mean life (years)`` and ``Worth (machine-years)``.
    """
    return pd.DataFrame(
        [
            (name, f"Gamma({shape:g}, {rate:g})", f"{rate / shape:.1f}", f"{rate:g}")
            for name, (shape, rate) in priors.items()
        ],
        columns=["Prior", "Gamma(shape, rate)", "Mean life (years)", "Worth (machine-years)"],
    )
