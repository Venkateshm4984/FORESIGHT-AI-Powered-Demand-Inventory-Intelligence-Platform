"""Tests for FORESIGHT forecasting and inventory logic."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.pipeline import load_all
from ml.evaluation import mae, rmse, wape
from ml.forecasting import (
    build_inventory_recommendations,
    forecast_weekly,
    rolling_origin_backtest,
    seasonal_naive,
)


@pytest.fixture(scope="module")
def project_data():
    return load_all()


def test_wape_mae_rmse():
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])
    assert wape(actual, pred) == pytest.approx(100 * 7 / 60)
    assert mae(actual, pred) == pytest.approx(7 / 3)
    assert rmse(actual, pred) > 0
    assert wape(np.array([0.0, 0.0]), pred) is None


def test_seasonal_naive_with_history():
    idx = pd.date_range("2024-01-07", periods=60, freq="W-SUN")
    s = pd.Series(np.arange(60, dtype=float), index=idx)
    out = seasonal_naive(s, 4)
    assert len(out) == 4
    assert out[0] == float(s.iloc[-52])


def test_forecast_weekly_shape(project_data):
    sales, _, calendar, *_ = project_data
    # Use a small SKU subset for speed
    skus = sorted(sales["SKU_Standard"].astype(str).unique())[:3]
    fc = forecast_weekly(sales, horizon=4, calendar=calendar, skus=skus)
    assert len(fc) == 12
    assert set(fc.columns) >= {"SKU_Standard", "Date", "Forecast_Units", "Seasonal_Naive_Units"}
    assert (fc["Forecast_Units"] >= 0).all()


def test_no_future_leakage_in_features():
    """Lag features must be shift(1)+ so target week is excluded."""
    from ml.forecasting import _features, _weekly_frame

    sales, _, calendar, *_ = load_all()
    sku = sorted(sales["SKU_Standard"].astype(str).unique())[0]
    weekly = _weekly_frame(sales, sku, calendar)
    feat = _features(weekly)
    # lag_1 at time t equals Units_Sold at t-1 on the full weekly series
    expected = weekly["Units_Sold"].shift(1).loc[feat.index]
    assert np.allclose(feat["lag_1"].to_numpy(), expected.to_numpy())


def test_rolling_origin_backtest(project_data):
    sales, _, calendar, *_ = project_data
    skus = sorted(sales["SKU_Standard"].astype(str).unique())[:5]
    result = rolling_origin_backtest(sales, horizon=4, folds=2, calendar=calendar, skus=skus)
    assert "summary" in result and "detail" in result
    if result["summary"]:
        assert "mean_model_wape_percent" in result["summary"]
        assert "mean_seasonal_naive_wape_percent" in result["summary"]
        assert result["summary"]["folds_evaluated"] == len(result["detail"])


def test_inventory_recommendations(project_data):
    sales, master, calendar, inventory, *_ = project_data
    skus = sorted(sales["SKU_Standard"].astype(str).unique())[:5]
    fc = forecast_weekly(sales, horizon=4, calendar=calendar, skus=skus)
    rec = build_inventory_recommendations(fc, inventory, horizon_weeks=4, master=master)
    assert len(rec) >= 1
    assert {"Stockout_Risk", "Overstock_Risk", "Recommended_Order", "Recommended_Action"} <= set(rec.columns)
    assert (rec["Recommended_Order"] >= 0).all()
