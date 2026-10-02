"""
FORESIGHT forecast evaluation metrics.

All metrics are computed from actual vs predicted arrays — nothing is hardcoded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def wape(actual, predicted) -> float | None:
    """
    Weighted Absolute Percentage Error (%).

    WAPE = 100 * sum(|actual − predicted|) / sum(|actual|)
    Returns None when the denominator is zero.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.abs(actual).sum()
    if denom == 0:
        return None
    return float(np.abs(actual - predicted).sum() / denom * 100.0)


def mae(actual, predicted) -> float:
    return float(mean_absolute_error(actual, predicted))


def rmse(actual, predicted) -> float:
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def summarize_backtest(detail: pd.DataFrame) -> dict:
    """Aggregate rolling-origin fold results into mean metrics."""
    if detail is None or detail.empty:
        return {}
    return {
        "folds_evaluated": int(len(detail)),
        "sku_count": int(detail["SKU_Standard"].nunique()),
        "mean_model_wape_percent": round(float(detail["model_wape_percent"].mean()), 2),
        "mean_seasonal_naive_wape_percent": round(
            float(detail["seasonal_naive_wape_percent"].mean()), 2
        ),
        "mean_model_mae": round(float(detail["model_mae"].mean()), 2),
        "mean_model_rmse": round(float(detail["model_rmse"].mean()), 2),
        "mean_seasonal_naive_mae": round(float(detail["seasonal_naive_mae"].mean()), 2)
        if "seasonal_naive_mae" in detail.columns
        else None,
        "mean_seasonal_naive_rmse": round(float(detail["seasonal_naive_rmse"].mean()), 2)
        if "seasonal_naive_rmse" in detail.columns
        else None,
    }
