"""Forecast service — weekly forecasts, baselines, and evaluation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.pipeline import PROCESSED_DIR, load_all, persist_pipeline_outputs
from ml.forecasting import forecast_weekly, rolling_origin_backtest


class ForecastService:
    def forecast(self, horizon_weeks: int = 4, sku: str | None = None) -> pd.DataFrame:
        sales, _, calendar, *_ = load_all()
        skus = [str(sku)] if sku else None
        return forecast_weekly(sales, horizon=horizon_weeks, calendar=calendar, skus=skus)

    def evaluate(self, horizon_weeks: int = 4, folds: int = 4, sku: str | None = None) -> dict:
        sales, _, calendar, *_ = load_all()
        skus = [str(sku)] if sku else None
        return rolling_origin_backtest(
            sales, horizon=horizon_weeks, folds=folds, calendar=calendar, skus=skus
        )

    def load_cached_forecast(self) -> pd.DataFrame | None:
        path = PROCESSED_DIR / "future_weekly_forecast_generated.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"])
            df["SKU_Standard"] = df["SKU_Standard"].astype(str)
            return df
        return None

    def load_cached_evaluation(self) -> dict | None:
        detail_path = PROCESSED_DIR / "rolling_origin_backtest.csv"
        summary_path = PROCESSED_DIR / "model_evaluation_summary.csv"
        if not detail_path.exists():
            return None
        detail = pd.read_csv(detail_path)
        summary = {}
        if summary_path.exists():
            summary = pd.read_csv(summary_path).iloc[0].to_dict()
        return {"detail": detail, "summary": summary}

    def run_and_persist(self, horizon_weeks: int = 4, folds: int = 4) -> dict:
        forecast = self.forecast(horizon_weeks=horizon_weeks)
        backtest = self.evaluate(horizon_weeks=horizon_weeks, folds=folds)
        paths = persist_pipeline_outputs(forecast=forecast, backtest=backtest)
        return {"forecast": forecast, "backtest": backtest, "paths": {k: str(v) for k, v in paths.items()}}
