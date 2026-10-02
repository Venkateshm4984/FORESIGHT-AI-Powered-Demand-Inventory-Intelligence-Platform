"""
Regenerate processed forecast / evaluation / inventory artefacts.

Usage (from foresight_demand_inventory/):
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.pipeline import persist_pipeline_outputs
from services.forecast_service import ForecastService
from services.inventory_service import InventoryService


def main():
    horizon, folds = 4, 4
    print("Running forecast…")
    fs = ForecastService()
    forecast = fs.forecast(horizon_weeks=horizon)
    print(f"  forecast rows: {len(forecast)}")

    print("Running rolling-origin backtest…")
    backtest = fs.evaluate(horizon_weeks=horizon, folds=folds)
    print(f"  summary: {backtest['summary']}")

    print("Building inventory recommendations…")
    inv = InventoryService()
    rec = inv.recommendations(horizon_weeks=horizon, forecast=forecast)
    print(f"  recommendations: {len(rec)}")

    paths = persist_pipeline_outputs(forecast=forecast, recommendations=rec, backtest=backtest)
    print("Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
