"""Tests for FORESIGHT preprocessing and data relationships."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.pipeline import latest_inventory, load_all, load_weekly, preprocess_report
from ml.preprocessing import aggregate_weekly_demand, run_preprocessing


@pytest.fixture(scope="module")
def bundle():
    return run_preprocessing()


def test_raw_files_exist():
    raw = ROOT / "data" / "raw"
    for name in [
        "sales_daily_clean.csv",
        "sku_master_clean.csv",
        "calendar_clean.csv",
        "inventory_snapshots_clean.csv",
        "future_weekly_forecast.csv",
        "inventory_risk_scores.csv",
    ]:
        assert (raw / name).exists(), f"Missing {name}"


def test_sales_schema_and_scope(bundle):
    sales = bundle["sales"]
    assert len(sales) == 36550
    assert sales["SKU_Standard"].nunique() == 50
    assert sales["Date"].min().date().isoformat() == "2024-01-01"
    assert sales["Date"].max().date().isoformat() == "2025-12-31"
    assert sales.duplicated(["Date", "SKU_Standard"]).sum() == 0
    assert (sales["Units_Sold"] < 0).sum() == 0


def test_master_filtered_to_sales_scope(bundle):
    sales = bundle["sales"]
    master = bundle["master"]
    assert set(master["SKU_Standard"]) == set(sales["SKU_Standard"])
    assert len(master) == 50


def test_inventory_filtered_to_sales_scope(bundle):
    sales_skus = set(bundle["sales"]["SKU_Standard"])
    inv_skus = set(bundle["inventory"]["SKU_Standard"])
    assert inv_skus == sales_skus
    assert len(bundle["inventory"]) == 50 * 24  # 24 monthly snapshots × 50 SKUs


def test_calendar_coverage(bundle):
    cal = bundle["calendar"]
    assert len(cal) == 731
    assert "has_promotion_event" in cal.columns


def test_weekly_aggregation(bundle):
    weekly = bundle["weekly"]
    assert weekly["SKU_Standard"].nunique() == 50
    assert weekly["Units_Sold"].sum() == pytest.approx(bundle["sales"]["Units_Sold"].sum(), rel=1e-9)


def test_joined_daily_has_category(bundle):
    joined = bundle["joined_daily"]
    assert joined["category"].notna().all()
    assert "is_holiday" in joined.columns


def test_preprocess_report_populated(bundle):
    report = bundle["report"].to_dict()
    assert report["steps"]
    assert report["stats"]["sales_rows"] == 36550


def test_load_all_pipeline():
    sales, master, calendar, inventory, risk, provided = load_all()
    assert len(sales) > 0
    assert len(master) == 50
    assert len(provided) == 200
    assert len(risk) == 50


def test_latest_inventory():
    latest = latest_inventory()
    assert len(latest) == 50
    assert "Current_Stock" in latest.columns
