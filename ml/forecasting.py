"""
FORESIGHT weekly SKU-level demand forecasting.

Models:
- Random Forest regressor with lag / rolling / calendar / promo features
- Seasonal-naive baseline (same week prior year when available)

Evaluation uses chronological rolling-origin backtesting with WAPE / MAE / RMSE.
Features for a forecast date use only information available before that date
(no future leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml.evaluation import mae, rmse, summarize_backtest, wape

FEATURE_COLS = [
    "lag_1",
    "lag_4",
    "lag_8",
    "lag_13",
    "lag_52",
    "rolling_4",
    "rolling_8",
    "rolling_13",
    "weekofyear",
    "month",
    "quarter",
    "promo_share",
    "is_holiday",
    "has_promotion_event",
]


def _weekly_frame(sales: pd.DataFrame, sku: str, calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a continuous weekly demand series for one SKU, optionally with calendar flags."""
    g = sales[sales["SKU_Standard"].astype(str) == str(sku)].copy()
    if g.empty:
        raise ValueError(f"Unknown SKU: {sku}")
    g["Date"] = pd.to_datetime(g["Date"])
    daily = g.groupby("Date").agg(
        Units_Sold=("Units_Sold", "sum"),
        Promotion=("Promotion", "mean") if "Promotion" in g.columns else ("Units_Sold", "sum"),
    ).sort_index()
    if "Promotion" not in daily.columns:
        daily["Promotion"] = 0.0

    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0)
    daily.index.name = "date"

    weekly = daily.resample("W-SUN").agg(
        Units_Sold=("Units_Sold", "sum"),
        promo_share=("Promotion", "mean"),
    ).astype(float)

    weekly["is_holiday"] = 0.0
    weekly["has_promotion_event"] = 0.0
    if calendar is not None and not calendar.empty:
        cal = calendar.copy()
        cal["date"] = pd.to_datetime(cal["date"] if "date" in cal.columns else cal["Date"])
        cal = cal.set_index("date").sort_index()
        cal_daily = cal.reindex(full_idx)
        for col in ["is_holiday", "has_promotion_event"]:
            if col in cal_daily.columns:
                weekly[col] = (
                    cal_daily[col].fillna(0).astype(float).resample("W-SUN").max().reindex(weekly.index).fillna(0)
                )
    weekly.index.name = "date"
    return weekly


def _features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Build supervised features. All lag/rolling stats are shifted so that
    row t uses only observations strictly before t.
    """
    s = weekly["Units_Sold"]
    x = pd.DataFrame(index=weekly.index)
    x["lag_1"] = s.shift(1)
    x["lag_4"] = s.shift(4)
    x["lag_8"] = s.shift(8)
    x["lag_13"] = s.shift(13)
    x["lag_52"] = s.shift(52)
    x["rolling_4"] = s.shift(1).rolling(4).mean()
    x["rolling_8"] = s.shift(1).rolling(8).mean()
    x["rolling_13"] = s.shift(1).rolling(13).mean()
    x["weekofyear"] = weekly.index.isocalendar().week.astype(int)
    x["month"] = weekly.index.month
    x["quarter"] = weekly.index.quarter
    # Promo / holiday known for historical weeks; for recursive future steps we
    # carry forward 0 unless a calendar value is supplied on the weekly frame.
    x["promo_share"] = weekly["promo_share"].shift(1).fillna(0)
    x["is_holiday"] = weekly.get("is_holiday", 0)
    x["has_promotion_event"] = weekly.get("has_promotion_event", 0)
    return x.join(s.rename("target")).dropna()


def seasonal_naive(train: pd.Series, horizon: int) -> np.ndarray:
    """Same ISO week from the prior year when ≥52 weeks exist; else recent mean."""
    if len(train) >= 52:
        return np.array([float(train.iloc[-52 + i]) for i in range(horizon)], dtype=float)
    return np.repeat(float(train.tail(min(4, len(train))).mean()), horizon)


def _fit_predict(weekly_train: pd.DataFrame, future_dates: pd.DatetimeIndex) -> np.ndarray:
    feat = _features(weekly_train)
    train_units = weekly_train["Units_Sold"]
    if len(feat) < 20:
        avg = float(train_units.tail(min(8, len(train_units))).mean()) if len(train_units) else 0.0
        return np.repeat(max(0.0, avg), len(future_dates))

    model = RandomForestRegressor(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(feat[FEATURE_COLS], feat["target"])

    hist = weekly_train.copy()
    preds: list[float] = []
    for d in future_dates:
        units = hist["Units_Sold"]
        row = {
            "lag_1": float(units.iloc[-1]),
            "lag_4": float(units.iloc[-4]) if len(units) >= 4 else float(units.iloc[-1]),
            "lag_8": float(units.iloc[-8]) if len(units) >= 8 else float(units.iloc[-1]),
            "lag_13": float(units.iloc[-13]) if len(units) >= 13 else float(units.iloc[-1]),
            "lag_52": float(units.iloc[-52]) if len(units) >= 52 else float(units.iloc[-1]),
            "rolling_4": float(units.iloc[-4:].mean()),
            "rolling_8": float(units.iloc[-8:].mean()),
            "rolling_13": float(units.iloc[-13:].mean()),
            "weekofyear": int(d.isocalendar().week),
            "month": int(d.month),
            "quarter": int((d.month - 1) // 3 + 1),
            # Future promo/holiday unknown → neutral 0 (no leakage of unrealized events).
            "promo_share": 0.0,
            "is_holiday": 0.0,
            "has_promotion_event": 0.0,
        }
        pred = max(0.0, float(model.predict(pd.DataFrame([row])[FEATURE_COLS])[0]))
        preds.append(pred)
        hist.loc[d, "Units_Sold"] = pred
        hist.loc[d, "promo_share"] = 0.0
        hist.loc[d, "is_holiday"] = 0.0
        hist.loc[d, "has_promotion_event"] = 0.0
    return np.array(preds, dtype=float)


def forecast_weekly(
    sales: pd.DataFrame,
    horizon: int = 4,
    calendar: pd.DataFrame | None = None,
    skus: list[str] | None = None,
) -> pd.DataFrame:
    """Generate weekly SKU-level forecasts for `horizon` weeks ahead."""
    if not 1 <= int(horizon) <= 12:
        raise ValueError("horizon must be between 1 and 12")
    sku_list = skus or sorted(sales["SKU_Standard"].astype(str).unique())
    rows = []
    for sku in sku_list:
        weekly = _weekly_frame(sales, sku, calendar)
        future_dates = pd.date_range(
            weekly.index.max() + pd.Timedelta(weeks=1), periods=horizon, freq="W-SUN"
        )
        preds = _fit_predict(weekly, future_dates)
        naive = seasonal_naive(weekly["Units_Sold"], horizon)
        for d, pred, base in zip(future_dates, preds, naive):
            rows.append(
                {
                    "SKU_Standard": sku,
                    "Date": d,
                    "Forecast_Units": round(float(pred), 2),
                    "Seasonal_Naive_Units": round(float(base), 2),
                }
            )
    return pd.DataFrame(rows)


def rolling_origin_backtest(
    sales: pd.DataFrame,
    horizon: int = 4,
    folds: int = 4,
    calendar: pd.DataFrame | None = None,
    skus: list[str] | None = None,
) -> dict:
    """
    Chronological rolling-origin backtest.

    For each SKU with enough history, evaluate successive cutoff origins.
    Training data for each fold ends at the cutoff; test weeks are never
    used as features.
    """
    if not 1 <= int(horizon) <= 12 or not 1 <= int(folds) <= 12:
        raise ValueError("horizon and folds must be between 1 and 12")
    sku_list = skus or sorted(sales["SKU_Standard"].astype(str).unique())
    results = []
    for sku in sku_list:
        weekly = _weekly_frame(sales, sku, calendar)
        if len(weekly) < 52 + horizon:
            continue
        cutoffs = list(range(len(weekly) - folds * horizon, len(weekly), horizon))
        cutoffs = [c for c in cutoffs if c >= 52 and c + horizon <= len(weekly)]
        for cutoff in cutoffs:
            train = weekly.iloc[:cutoff]
            test = weekly.iloc[cutoff : cutoff + horizon]
            dates = test.index
            pred = _fit_predict(train, dates)
            base = seasonal_naive(train["Units_Sold"], horizon)
            actual = test["Units_Sold"].values
            results.append(
                {
                    "SKU_Standard": sku,
                    "cutoff": str(train.index.max().date()),
                    "horizon_weeks": horizon,
                    "model_wape_percent": None if wape(actual, pred) is None else round(wape(actual, pred), 2),
                    "seasonal_naive_wape_percent": None
                    if wape(actual, base) is None
                    else round(wape(actual, base), 2),
                    "model_mae": round(mae(actual, pred), 2),
                    "model_rmse": round(rmse(actual, pred), 2),
                    "seasonal_naive_mae": round(mae(actual, base), 2),
                    "seasonal_naive_rmse": round(rmse(actual, base), 2),
                }
            )
    detail = pd.DataFrame(results)
    summary = summarize_backtest(detail)
    # Drop None placeholders from summary for cleaner JSON
    summary = {k: v for k, v in summary.items() if v is not None}
    return {"detail": detail, "summary": summary}


def build_inventory_recommendations(
    forecast: pd.DataFrame,
    inventory: pd.DataFrame,
    horizon_weeks: int = 4,
    master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Explainable inventory intelligence.

    Business rules (documented):
    - Available_Stock = Current_Stock + On_Order
    - Forecast_Weekly_Units = sum(forecast over horizon) / horizon_weeks
    - Lead_Time_Demand = Forecast_Weekly_Units × (Lead_Time_Days / 7)
    - Stock_Coverage_Weeks = Available_Stock / Forecast_Weekly_Units
      (inf when weekly forecast is 0)
    - Stockout_Risk = Available_Stock < Lead_Time_Demand
    - Overstock_Risk = Stock_Coverage_Weeks > 8
    - Recommended_Order = max(0, ceil(Reorder_Point − Available_Stock))
    - Action priority: Stockout → Overstock → Plan Reorder → Watch / Healthy
    """
    horizon_weeks = max(1, int(horizon_weeks))
    total = (
        forecast.groupby("SKU_Standard", as_index=False)["Forecast_Units"]
        .sum()
        .rename(columns={"Forecast_Units": "Forecast_Horizon_Units"})
    )
    x = inventory.sort_values("Snapshot_Date").groupby("SKU_Standard", as_index=False).tail(1).copy()
    x = x.merge(total, on="SKU_Standard", how="left")
    x["Forecast_Horizon_Units"] = x["Forecast_Horizon_Units"].fillna(0)
    x["Forecast_4W_Units"] = x["Forecast_Horizon_Units"]  # alias for compatibility
    x["Forecast_Weekly_Units"] = x["Forecast_Horizon_Units"] / horizon_weeks
    x["Lead_Time_Demand"] = x["Forecast_Weekly_Units"] * (
        pd.to_numeric(x["Lead_Time_Days"], errors="coerce") / 7.0
    )
    x["Available_Stock"] = pd.to_numeric(x["Current_Stock"], errors="coerce") + pd.to_numeric(
        x["On_Order"], errors="coerce"
    )
    x["Stock_Coverage_Weeks"] = np.where(
        x["Forecast_Weekly_Units"] > 0,
        x["Available_Stock"] / x["Forecast_Weekly_Units"],
        np.inf,
    )
    x["Stockout_Risk"] = x["Available_Stock"] < x["Lead_Time_Demand"]
    x["Overstock_Risk"] = x["Stock_Coverage_Weeks"] > 8
    x["Recommended_Order"] = np.maximum(
        0,
        np.ceil(
            pd.to_numeric(x["Reorder_Point"], errors="coerce")
            - pd.to_numeric(x["Available_Stock"], errors="coerce")
        ),
    ).astype(int)
    x["Recommended_Action"] = np.select(
        [x["Stockout_Risk"], x["Overstock_Risk"], x["Recommended_Order"] > 0],
        ["Expedite / Reorder", "Markdown / Clear", "Plan Reorder"],
        default="Watch / Healthy",
    )
    if master is not None and not master.empty:
        cols = [c for c in ["SKU_Standard", "sku_name", "category", "subcategory", "brand", "unit_price", "cost_price"] if c in master.columns]
        x = x.merge(master[cols], on="SKU_Standard", how="left", suffixes=("", "_master"))
    return x.reset_index(drop=True)
