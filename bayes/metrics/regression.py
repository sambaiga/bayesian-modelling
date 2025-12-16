import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.utils.validation import check_consistent_length


def regression_report(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    digits: int = 2,
) -> pd.DataFrame:
    """Generate a comprehensive regression performance report for battery SoH prediction.

    Includes standard regression metrics plus battery-specific error interpretations
    useful for State of Health (SoH) modeling.

    Args:
        y_true: Ground truth SoH values (typically in % or normalized [0,1]).
        y_pred: Predicted SoH values.
        digits: Number of decimal places to round metric values.

    Returns:
        DataFrame with Metric, Value, and Description columns.

    Raises:
        ValueError: If inputs have mismatched lengths or invalid types.
    """
    # Input validation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    check_consistent_length(y_true, y_pred)

    mae = metrics.mean_absolute_error(y_true, y_pred)
    rmse = metrics.root_mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    map = metrics.mean_absolute_percentage_error(y_true, y_pred) * 100

    max_error = metrics.max_error(y_true, y_pred)
    within_3pct = np.mean(np.abs(y_true - y_pred) <= 0.03) * 100
    within_5pct = np.mean(np.abs(y_true - y_pred) <= 0.05) * 100

    rows = [
        ("MAE", mae, "Mean Absolute Error (SoH units)"),
        ("RMSE", rmse, "Root Mean Squared Error (SoH units)"),
        ("MAP", map, "Mean Absolute Percentage Error"),
        ("R2", r2, "Coefficient of Determination"),
        ("Max Error", max_error, "Largest single prediction error"),
        ("±3% MAE", within_3pct, "Predictions within ±3% of true SoH"),
        ("±5% MAE", within_5pct, "Predictions within ±5% of true SoH"),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Value", "Description"])
    df["Value"] = df["Value"].round(digits)
    return df
