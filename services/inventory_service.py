"""Inventory intelligence service."""

from __future__ import annotations

import pandas as pd

from data.pipeline import PROCESSED_DIR, latest_inventory, load_all, persist_pipeline_outputs
from ml.forecasting import build_inventory_recommendations, forecast_weekly
from services.forecast_service import ForecastService


class InventoryService:
    def recommendations(self, horizon_weeks: int = 4, forecast: pd.DataFrame | None = None) -> pd.DataFrame:
        sales, master, calendar, inventory, *_ = load_all()
        if forecast is None:
            forecast = forecast_weekly(sales, horizon=horizon_weeks, calendar=calendar)
        return build_inventory_recommendations(
            forecast, inventory, horizon_weeks=horizon_weeks, master=master
        )

    def risk_view(self) -> pd.DataFrame:
        return latest_inventory()

    def load_cached_recommendations(self) -> pd.DataFrame | None:
        path = PROCESSED_DIR / "inventory_recommendations_generated.csv"
        if path.exists():
            return pd.read_csv(path)
        return None

    def run_and_persist(self, horizon_weeks: int = 4) -> dict:
        fs = ForecastService()
        forecast = fs.forecast(horizon_weeks=horizon_weeks)
        rec = self.recommendations(horizon_weeks=horizon_weeks, forecast=forecast)
        paths = persist_pipeline_outputs(forecast=forecast, recommendations=rec)
        return {"forecast": forecast, "recommendations": rec, "paths": {k: str(v) for k, v in paths.items()}}
