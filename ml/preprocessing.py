"""
FORESIGHT data preprocessing pipeline.

Loads the supplied cleaned project CSVs, validates schemas and relationships,
and produces analysis-ready frames. Transformations are documented and
deterministic. No synthetic rows are invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

REQUIRED_FILES = {
    "sales": "sales_daily_clean.csv",
    "sku_master": "sku_master_clean.csv",
    "calendar": "calendar_clean.csv",
    "inventory": "inventory_snapshots_clean.csv",
    "risk": "inventory_risk_scores.csv",
    "provided_forecast": "future_weekly_forecast.csv",
}


@dataclass
class PreprocessReport:
    """Documents every important transformation applied during preprocessing."""

    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add(self, message: str) -> None:
        self.steps.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {"steps": self.steps, "warnings": self.warnings, "stats": self.stats}


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_sku(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def load_raw_tables(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all required raw CSVs with date parsing. Raises if a file is missing."""
    base = Path(raw_dir) if raw_dir else RAW_DIR
    paths = {key: base / filename for key, filename in REQUIRED_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset file(s). Place the supplied FORESIGHT CSVs "
            f"in {base}. Missing: {missing}"
        )

    tables = {
        "sales": pd.read_csv(paths["sales"], parse_dates=["Date"]),
        "sku_master": pd.read_csv(paths["sku_master"]),
        "calendar": pd.read_csv(paths["calendar"], parse_dates=["date"]),
        "inventory": pd.read_csv(paths["inventory"], parse_dates=["Snapshot_Date"]),
        "risk": pd.read_csv(paths["risk"], parse_dates=["Snapshot_Date"]),
        "provided_forecast": pd.read_csv(paths["provided_forecast"], parse_dates=["Date"]),
    }
    return tables


def preprocess_sales(df: pd.DataFrame, report: PreprocessReport) -> pd.DataFrame:
    _require_columns(
        df,
        ["Date", "SKU", "Units_Sold", "Revenue", "Price", "Promotion", "SKU_Standard"],
        "sales",
    )
    out = df.copy()
    before = len(out)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    bad_dates = int(out["Date"].isna().sum())
    if bad_dates:
        report.warn(f"sales: dropped {bad_dates} rows with invalid Date")
        out = out.dropna(subset=["Date"])
    out["SKU_Standard"] = _normalize_sku(out["SKU_Standard"])
    out["SKU"] = out["SKU"].astype(str).str.strip()
    for col in ["Units_Sold", "Revenue", "Price", "Promotion"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    # Units_Sold missing → 0 (no sale recorded). Do not invent revenue/price.
    missing_units = int(out["Units_Sold"].isna().sum())
    if missing_units:
        report.add(f"sales: filled {missing_units} missing Units_Sold with 0")
        out["Units_Sold"] = out["Units_Sold"].fillna(0)
    neg = int((out["Units_Sold"] < 0).sum())
    if neg:
        report.warn(f"sales: clipped {neg} negative Units_Sold to 0")
        out.loc[out["Units_Sold"] < 0, "Units_Sold"] = 0
    out["Promotion"] = out["Promotion"].fillna(0).astype(int).clip(0, 1)

    dup_mask = out.duplicated(subset=["Date", "SKU_Standard"], keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        report.warn(f"sales: removed {n_dup} duplicate Date+SKU_Standard rows (kept first)")
        out = out.loc[~dup_mask].copy()

    dropped_pct = 100.0 * (before - len(out)) / before if before else 0.0
    if dropped_pct > 5:
        report.warn(
            f"sales: dropped {dropped_pct:.1f}% of rows during cleaning — review upstream data"
        )
    report.add(
        f"sales: {len(out):,} rows, {out['SKU_Standard'].nunique()} SKUs, "
        f"{out['Date'].min().date()} → {out['Date'].max().date()}"
    )
    report.stats["sales_rows"] = len(out)
    report.stats["sales_skus"] = int(out["SKU_Standard"].nunique())
    report.stats["sales_date_min"] = str(out["Date"].min().date())
    report.stats["sales_date_max"] = str(out["Date"].max().date())
    return out.sort_values(["SKU_Standard", "Date"]).reset_index(drop=True)


def preprocess_master(
    df: pd.DataFrame, sales_skus: set[str], report: PreprocessReport
) -> pd.DataFrame:
    _require_columns(
        df,
        ["sku_id", "sku_name", "category", "subcategory", "unit_price", "cost_price", "brand", "SKU_Standard"],
        "sku_master",
    )
    out = df.copy()
    out["SKU_Standard"] = _normalize_sku(out["SKU_Standard"])
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")
    out["cost_price"] = pd.to_numeric(out["cost_price"], errors="coerce")
    full_count = len(out)
    # Product master has 5,000 SKUs; project sales scope is smaller.
    scoped = out[out["SKU_Standard"].isin(sales_skus)].copy()
    report.add(
        f"sku_master: filtered {full_count:,} products → {len(scoped):,} in sales scope "
        f"({len(sales_skus)} sales SKUs)"
    )
    missing = sales_skus - set(scoped["SKU_Standard"])
    if missing:
        report.warn(f"sku_master: {len(missing)} sales SKUs missing from master: {sorted(missing)[:10]}")
    report.stats["master_in_scope"] = len(scoped)
    return scoped.sort_values("SKU_Standard").reset_index(drop=True)


def preprocess_calendar(df: pd.DataFrame, report: PreprocessReport) -> pd.DataFrame:
    _require_columns(
        df,
        ["date", "year", "month", "week", "is_weekend", "season", "is_holiday"],
        "calendar",
    )
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    bad = int(out["date"].isna().sum())
    if bad:
        report.warn(f"calendar: dropped {bad} rows with invalid date")
        out = out.dropna(subset=["date"])
    for col in ["year", "month", "week", "is_weekend", "is_holiday"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    # holiday / promotion_event are sparse by design — keep as nullable strings.
    out["holiday"] = out.get("holiday", pd.Series(index=out.index, dtype="object")).fillna("")
    out["promotion_event"] = out.get("promotion_event", pd.Series(index=out.index, dtype="object")).fillna("")
    out["has_promotion_event"] = (out["promotion_event"].astype(str).str.len() > 0).astype(int)
    dup = out.duplicated(subset=["date"], keep="first")
    if dup.any():
        report.warn(f"calendar: removed {int(dup.sum())} duplicate dates")
        out = out.loc[~dup].copy()
    report.add(
        f"calendar: {len(out):,} days, holidays={int(out['is_holiday'].sum())}, "
        f"promo_events={int(out['has_promotion_event'].sum())}"
    )
    report.stats["calendar_rows"] = len(out)
    return out.sort_values("date").reset_index(drop=True)


def preprocess_inventory(
    df: pd.DataFrame, sales_skus: set[str], report: PreprocessReport
) -> pd.DataFrame:
    _require_columns(
        df,
        [
            "Snapshot_Date",
            "SKU",
            "Current_Stock",
            "On_Order",
            "Lead_Time_Days",
            "Safety_Stock",
            "Reorder_Point",
            "Inventory_Value",
            "SKU_Standard",
        ],
        "inventory",
    )
    out = df.copy()
    out["Snapshot_Date"] = pd.to_datetime(out["Snapshot_Date"], errors="coerce")
    out = out.dropna(subset=["Snapshot_Date"])
    out["SKU_Standard"] = _normalize_sku(out["SKU_Standard"])
    for col in ["Current_Stock", "On_Order", "Lead_Time_Days", "Safety_Stock", "Reorder_Point", "Inventory_Value"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    full = len(out)
    scoped = out[out["SKU_Standard"].isin(sales_skus)].copy()
    report.add(
        f"inventory: filtered {full:,} snapshots → {len(scoped):,} for sales-scope SKUs"
    )
    report.stats["inventory_rows"] = len(scoped)
    report.stats["inventory_skus"] = int(scoped["SKU_Standard"].nunique())
    return scoped.sort_values(["SKU_Standard", "Snapshot_Date"]).reset_index(drop=True)


def preprocess_risk(df: pd.DataFrame, sales_skus: set[str], report: PreprocessReport) -> pd.DataFrame:
    out = df.copy()
    if "Snapshot_Date" in out.columns:
        out["Snapshot_Date"] = pd.to_datetime(out["Snapshot_Date"], errors="coerce")
    out["SKU_Standard"] = _normalize_sku(out["SKU_Standard"])
    scoped = out[out["SKU_Standard"].isin(sales_skus)].copy()
    report.add(f"risk: {len(scoped):,} rows in sales scope")
    report.stats["risk_rows"] = len(scoped)
    return scoped.reset_index(drop=True)


def preprocess_provided_forecast(
    df: pd.DataFrame, sales_skus: set[str], report: PreprocessReport
) -> pd.DataFrame:
    _require_columns(df, ["SKU_Standard", "Date", "Forecast_Units"], "provided_forecast")
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])
    out["SKU_Standard"] = _normalize_sku(out["SKU_Standard"])
    out["Forecast_Units"] = pd.to_numeric(out["Forecast_Units"], errors="coerce").fillna(0)
    scoped = out[out["SKU_Standard"].isin(sales_skus)].copy()
    report.add(
        f"provided_forecast: {len(scoped):,} rows, "
        f"{scoped['Date'].min().date()} → {scoped['Date'].max().date()}"
    )
    report.stats["provided_forecast_rows"] = len(scoped)
    return scoped.sort_values(["SKU_Standard", "Date"]).reset_index(drop=True)


def build_joined_daily(
    sales: pd.DataFrame,
    master: pd.DataFrame,
    calendar: pd.DataFrame,
    report: PreprocessReport | None = None,
) -> pd.DataFrame:
    """Join sales + master + calendar on SKU_Standard and Date/date."""
    rep = report or PreprocessReport()
    df = sales.merge(
        master[
            ["SKU_Standard", "sku_name", "category", "subcategory", "unit_price", "cost_price", "brand"]
        ],
        on="SKU_Standard",
        how="left",
    )
    unmatched = int(df["sku_name"].isna().sum())
    if unmatched:
        rep.warn(f"joined_daily: {unmatched} sales rows without master match")
    cal_cols = [
        "date",
        "week",
        "season",
        "is_holiday",
        "holiday",
        "promotion_event",
        "has_promotion_event",
        "is_weekend",
        "day_of_week",
    ]
    available = [c for c in cal_cols if c in calendar.columns]
    df = df.merge(calendar[available], left_on="Date", right_on="date", how="left")
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    for col in ["is_holiday", "has_promotion_event", "is_weekend"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    rep.add(f"joined_daily: {len(df):,} rows after sales⋈master⋈calendar")
    return df.sort_values(["SKU_Standard", "Date"]).reset_index(drop=True)


def aggregate_weekly_demand(
    sales: pd.DataFrame,
    calendar: pd.DataFrame | None = None,
    report: PreprocessReport | None = None,
) -> pd.DataFrame:
    """
    Aggregate daily sales to Monday–Sunday weekly SKU demand (W-SUN).

    Calendar promotion/holiday flags are aggregated as weekly maxima / sums
    where calendar is provided.
    """
    rep = report or PreprocessReport()
    s = sales.copy()
    s["Date"] = pd.to_datetime(s["Date"])
    # Align to Sunday week-end used by forecasting (W-SUN).
    s["week_end"] = s["Date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()

    agg = (
        s.groupby(["SKU_Standard", "week_end"], as_index=False)
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Revenue=("Revenue", "sum"),
            Promo_Days=("Promotion", "sum"),
            Avg_Price=("Price", "mean"),
        )
        .rename(columns={"week_end": "Date"})
    )

    if calendar is not None and not calendar.empty:
        cal = calendar.copy()
        cal["Date"] = pd.to_datetime(cal["date"]).dt.to_period("W-SUN").dt.end_time.dt.normalize()
        cal_week = (
            cal.groupby("Date", as_index=False)
            .agg(
                is_holiday=("is_holiday", "max"),
                has_promotion_event=("has_promotion_event", "max")
                if "has_promotion_event" in cal.columns
                else ("is_holiday", "max"),
                season=("season", "first"),
            )
        )
        agg = agg.merge(cal_week, on="Date", how="left")
        agg["is_holiday"] = agg.get("is_holiday", 0)
        agg["is_holiday"] = agg["is_holiday"].fillna(0).astype(int)
        if "has_promotion_event" in agg.columns:
            agg["has_promotion_event"] = agg["has_promotion_event"].fillna(0).astype(int)

    rep.add(
        f"weekly_demand: {len(agg):,} SKU-weeks, "
        f"{agg['SKU_Standard'].nunique()} SKUs, "
        f"{agg['Date'].min().date()} → {agg['Date'].max().date()}"
    )
    rep.stats["weekly_rows"] = len(agg)
    return agg.sort_values(["SKU_Standard", "Date"]).reset_index(drop=True)


def run_preprocessing(raw_dir: Path | None = None) -> dict[str, Any]:
    """
    Full preprocessing entrypoint.

    Returns dict with cleaned tables, joined daily frame, weekly demand,
    and a PreprocessReport documenting transformations.
    """
    report = PreprocessReport()
    report.add(f"Loading raw tables from {raw_dir or RAW_DIR}")
    tables = load_raw_tables(raw_dir)

    sales = preprocess_sales(tables["sales"], report)
    sales_skus = set(sales["SKU_Standard"].astype(str))
    master = preprocess_master(tables["sku_master"], sales_skus, report)
    calendar = preprocess_calendar(tables["calendar"], report)
    inventory = preprocess_inventory(tables["inventory"], sales_skus, report)
    risk = preprocess_risk(tables["risk"], sales_skus, report)
    provided_forecast = preprocess_provided_forecast(tables["provided_forecast"], sales_skus, report)
    joined_daily = build_joined_daily(sales, master, calendar, report)
    weekly = aggregate_weekly_demand(sales, calendar, report)

    report.stats["promo_rate"] = float(sales["Promotion"].mean())
    report.stats["total_units"] = float(sales["Units_Sold"].sum())
    report.stats["total_revenue"] = float(sales["Revenue"].sum())

    return {
        "sales": sales,
        "master": master,
        "calendar": calendar,
        "inventory": inventory,
        "risk": risk,
        "provided_forecast": provided_forecast,
        "joined_daily": joined_daily,
        "weekly": weekly,
        "report": report,
    }


def save_processed_frame(df: pd.DataFrame, name: str, processed_dir: Path | None = None) -> Path:
    """Persist a processed DataFrame under data/processed/."""
    out_dir = Path(processed_dir) if processed_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    df.to_csv(path, index=False)
    return path
