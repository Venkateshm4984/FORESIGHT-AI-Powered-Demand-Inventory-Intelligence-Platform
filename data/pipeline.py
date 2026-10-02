"""
Data access helpers for FORESIGHT.

Delegates cleaning to ml.preprocessing and exposes convenience loaders used by
the Streamlit app, Flask API, and services layer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from ml.preprocessing import (
    PROCESSED_DIR,
    RAW_DIR,
    run_preprocessing,
    save_processed_frame,
)

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def _bundle():
    return run_preprocessing(RAW_DIR)


def clear_cache() -> None:
    _bundle.cache_clear()


def load_all():
    """Return (sales, master, calendar, inventory, risk, provided_forecast)."""
    b = _bundle()
    return (
        b["sales"],
        b["master"],
        b["calendar"],
        b["inventory"],
        b["risk"],
        b["provided_forecast"],
    )


def load_sales() -> pd.DataFrame:
    return _bundle()["sales"]


def load_master() -> pd.DataFrame:
    return _bundle()["master"]


def load_calendar() -> pd.DataFrame:
    return _bundle()["calendar"]


def load_inventory() -> pd.DataFrame:
    return _bundle()["inventory"]


def load_risk() -> pd.DataFrame:
    return _bundle()["risk"]


def load_provided_forecast() -> pd.DataFrame:
    return _bundle()["provided_forecast"]


def load_joined_daily() -> pd.DataFrame:
    return _bundle()["joined_daily"]


def load_weekly() -> pd.DataFrame:
    return _bundle()["weekly"]


def preprocess_report() -> dict:
    return _bundle()["report"].to_dict()


def latest_inventory() -> pd.DataFrame:
    """Latest inventory snapshot per SKU joined with master + supplied risk fields."""
    b = _bundle()
    inventory = b["inventory"]
    master = b["master"]
    risk = b["risk"]
    latest = inventory.sort_values("Snapshot_Date").groupby("SKU_Standard", as_index=False).tail(1)
    cols = [
        "SKU_Standard",
        "Current_Stock",
        "On_Order",
        "Lead_Time_Days",
        "Safety_Stock",
        "Reorder_Point",
        "Inventory_Value",
        "Snapshot_Date",
    ]
    out = latest[cols].merge(
        master[["SKU_Standard", "sku_name", "category", "subcategory", "unit_price", "cost_price", "brand"]],
        on="SKU_Standard",
        how="left",
    )
    risk_cols = [
        c
        for c in [
            "SKU_Standard",
            "Forecast_4W_Units",
            "Forecast_Weekly_Units",
            "Stock_Coverage_Weeks",
            "Stockout_Risk",
            "Overstock_Risk",
            "Recommended_Action",
        ]
        if c in risk.columns
    ]
    if risk_cols:
        out = out.merge(risk[risk_cols], on="SKU_Standard", how="left", suffixes=("", "_supplied"))
    return out.sort_values("SKU_Standard").reset_index(drop=True)


def build_daily_model_frame() -> pd.DataFrame:
    """Compatibility helper used by older report scripts."""
    df = load_joined_daily().copy()
    df["sales"] = pd.to_numeric(df["Units_Sold"], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["Date"])
    df["sku"] = df["SKU_Standard"]
    df["unit_cost"] = pd.to_numeric(df["cost_price"], errors="coerce")
    return df.sort_values(["sku", "date"]).reset_index(drop=True)


def persist_pipeline_outputs(
    forecast: pd.DataFrame | None = None,
    recommendations: pd.DataFrame | None = None,
    backtest: dict | None = None,
) -> dict[str, Path]:
    """Write generated artefacts to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    weekly = load_weekly()
    paths["weekly_demand"] = save_processed_frame(weekly, "weekly_sku_demand.csv")
    paths["latest_inventory"] = save_processed_frame(latest_inventory(), "latest_inventory_scope.csv")
    report = preprocess_report()
    pd.DataFrame([{"key": k, "value": str(v)} for k, v in report["stats"].items()]).to_csv(
        PROCESSED_DIR / "preprocess_stats.csv", index=False
    )
    paths["preprocess_stats"] = PROCESSED_DIR / "preprocess_stats.csv"
    if forecast is not None:
        paths["forecast"] = save_processed_frame(forecast, "future_weekly_forecast_generated.csv")
    if recommendations is not None:
        paths["recommendations"] = save_processed_frame(
            recommendations, "inventory_recommendations_generated.csv"
        )
    if backtest is not None:
        if not backtest["detail"].empty:
            paths["backtest"] = save_processed_frame(backtest["detail"], "rolling_origin_backtest.csv")
        if backtest.get("summary"):
            pd.DataFrame([backtest["summary"]]).to_csv(
                PROCESSED_DIR / "model_evaluation_summary.csv", index=False
            )
            paths["evaluation_summary"] = PROCESSED_DIR / "model_evaluation_summary.csv"
    return paths
