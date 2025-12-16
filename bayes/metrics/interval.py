import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length


def winkler_score(
    pred: np.array, true: np.array, lower: np.array, upper: np.array, alpha: float | None = None
) -> float:
    r"""Calculate the median Winkler score to evaluate prediction interval coverage and width.

    The Winkler score for each data point is:
    $$
    WinklerScore = Upper - Lower + \\frac{2}{\\alpha} \\times (Lower - True)
      \\times (True < Lower) + \\frac{2}{\\alpha} \\times (True - Upper) \\times (True > Upper)
    $$
    This function returns the median of these scores, as per Winkler's 1972 paper:
    R. L. Winkler, "A Decision-Theoretic Approach to Interval Estimation,"
    Journal of the American Statistical Association, vol. 67, no. 337, pp. 187–191, 1972.

    Args:
        pred (np.array): Predicted values.
        true (np.array): True values.
        lower (np.array): Lower bounds of the prediction intervals.
        upper (np.array): Upper bounds of the prediction intervals.
        alpha (float, optional): Significance level for the intervals. Defaults to None.

    Returns:
        float: Median Winkler score.

    Raises:
        ValueError: If alpha is None.

    Examples:
        >>> pred = np.array([10, 15, 20, 25, 30])
        >>> true = np.array([11, 14, 19, 26, 29])
        >>> lower = np.array([9, 13, 18, 24, 28])
        >>> upper = np.array([12, 17, 22, 27, 31])
        >>> score = winkler_score(pred, true, lower, upper, 0.05)
        >>> print(f"Median Winkler Score: {score:.2f}")
    """
    if alpha is None:
        raise ValueError("Alpha cannot be None for the Winkler score calculation.")

    scores = (
        upper
        - lower
        + (2.0 / alpha) * (lower - true) * (true < lower)
        + (2.0 / alpha) * (true - upper) * (true > upper)
    )
    return np.median(scores)


def interval_coverage(true: np.array, lower: np.array, upper: np.array) -> float:
    r"""Calculate the coverage proportion of prediction intervals.

    The coverage is:
    $$
    Coverage = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbb{1}(lower_i \\leq true_i \\leq upper_i)
    $$
    where \( \\mathbb{1} \) is the indicator function (1 if true, 0 otherwise), and
      \( n \) is the number of observations.

    Args:
        true (np.array): True values.
        lower (np.array): Lower bounds of the prediction intervals.
        upper (np.array): Upper bounds of the prediction intervals.

    Returns:
        float: Proportion of true values within the intervals (0 to 1).

    Examples:
        >>> true = np.array([10, 15, 20, 25, 30])
        >>> lower = np.array([8, 14, 18, 22, 28])
        >>> upper = np.array([12, 16, 22, 28, 32])
        >>> coverage = interval_coverage(true, lower, upper)
        >>> print(f"Coverage: {coverage:.2f}")
    """
    coverage = ((true >= lower) & (true <= upper)).mean()
    return coverage


def ace(coverage: float, alpha: float) -> float:
    r"""Calculate the Average Coverage Error (ACE).

    The ACE is:
    $$
    ACE = Coverage - (1 - \\alpha)
    $$
    where \( Coverage \) is the proportion of true values within the prediction intervals.

    Args:
        coverage (float): Coverage proportion (0 to 1).
        alpha (float): Significance level for the prediction intervals.

    Returns:
        float: Average Coverage Error.

    Examples:
        >>> coverage = 0.95
        >>> alpha = 0.05
        >>> error = ace(coverage, alpha)
        >>> print(f"ACE: {error:.2f}")
    """
    return coverage - (1 - alpha)


def nmpi(lower: np.array, upper: np.array, scale: float) -> float:
    r"""Calculate the Normalized Median Prediction Interval (NMPI).

    The NMPI is:
    $$
    NMPI = \\text{median}(\\left| Upper - Lower \\right|)
    $$

    Args:
        lower (np.array): Lower bounds of the prediction intervals.
        upper (np.array): Upper bounds of the prediction intervals.
        scale (float): Scaling factor for the NMPI.

    Returns:
        float: Median prediction interval width.

    Examples:
        >>> lower = np.array([8, 14, 18, 22, 28])
        >>> upper = np.array([12, 16, 22, 28, 32])
        >>> width = nmpi(lower, upper)
        >>> print(f"NMPI: {width:.2f}")
    """
    return np.median(np.abs(upper - lower)) / scale


def get_interval_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.1,
    digits: int = 2,
) -> pd.DataFrame:
    """Calculates interval prediction metrics and returns them as a DataFrame.

    Args:
        pred: Predicted point estimates.
        true: Ground truth values.
        lower: Lower bounds of prediction intervals.
        upper: Upper bounds of prediction intervals.
        alpha: Significance level (e.g., 0.1 for 90% interval). Must be in (0, 1).
        digits: Number of decimal places for rounding.

    Returns:
        DataFrame with interval metrics: Winkler Score, PICP, ACE, NMPI.

    Raises:
        ValueError: If alpha is invalid or array shapes don't match.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be between 0 and 1 (exclusive), got {alpha}")

    # Check all arrays have same length
    for arr in (pred, lower, upper):
        check_consistent_length(true, arr)

    coverage = interval_coverage(true, lower, upper)
    interval_width = nmpi(lower, upper, scale=true.max())

    metrics = [
        ("Winkler Score", winkler_score(pred, true, lower, upper, alpha), "Winkler score (lower = better)"),
        ("PICP", coverage, "Prediction Interval Coverage Probability\n(Fraction of true values covered)"),
        ("ACE", ace(coverage, alpha), f"Absolute Coverage Error = |PICP - {1 - alpha:.0%}| × 100%"),
        ("NMPI", interval_width, "Normalized Mean Prediction Interval width\n(Avg width / nominal capacity)"),
    ]

    df_metrics = pd.DataFrame(metrics, columns=["Metric", "Value", "Description"])
    df_metrics["Value"] = df_metrics["Value"].round(digits)

    return df_metrics
