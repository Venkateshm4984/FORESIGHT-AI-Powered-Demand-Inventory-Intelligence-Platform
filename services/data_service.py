"""Data service — load processed tables and overview KPIs."""

from __future__ import annotations

import pandas as pd

from data.pipeline import (
    latest_inventory,
    load_all,
    load_joined_daily,
    load_weekly,
    preprocess_report,
)


class DataService:
    def overview(self) -> dict:
        sales, master, calendar, inventory, risk, provided = load_all()
        inv_latest = latest_inventory()
        return {
            "sales_rows": int(len(sales)),
            "sku_count": int(sales["SKU_Standard"].nunique()),
            "date_min": str(sales["Date"].min().date()),
            "date_max": str(sales["Date"].max().date()),
            "total_units": float(sales["Units_Sold"].sum()),
            "total_revenue": float(sales["Revenue"].sum()),
            "categories": int(master["category"].nunique()),
            "inventory_snapshots": int(len(inventory)),
            "current_stock_units": float(inv_latest["Current_Stock"].sum()),
            "inventory_value": float(inv_latest["Inventory_Value"].sum()),
            "stockout_risk_skus": int(inv_latest["Stockout_Risk"].fillna(False).astype(bool).sum())
            if "Stockout_Risk" in inv_latest.columns
            else 0,
            "overstock_risk_skus": int(inv_latest["Overstock_Risk"].fillna(False).astype(bool).sum())
            if "Overstock_Risk" in inv_latest.columns
            else 0,
            "calendar_days": int(len(calendar)),
            "holidays": int(calendar["is_holiday"].sum()),
            "provided_forecast_rows": int(len(provided)),
            "risk_rows": int(len(risk)),
            "preprocess": preprocess_report(),
        }

    def sales(self) -> pd.DataFrame:
        return load_all()[0]

    def master(self) -> pd.DataFrame:
        return load_all()[1]

    def calendar(self) -> pd.DataFrame:
        return load_all()[2]

    def inventory(self) -> pd.DataFrame:
        return load_all()[3]

    def risk(self) -> pd.DataFrame:
        return load_all()[4]

    def provided_forecast(self) -> pd.DataFrame:
        return load_all()[5]

    def joined_daily(self) -> pd.DataFrame:
        return load_joined_daily()

    def weekly(self) -> pd.DataFrame:
        return load_weekly()

    def sku_detail(self, sku: str) -> dict:
        sku = str(sku).strip()
        sales, master, _, inventory, risk, provided = load_all()
        sales_sku = sales[sales["SKU_Standard"].astype(str) == sku]
        if sales_sku.empty:
            raise KeyError(f"SKU not found in sales scope: {sku}")
        master_row = master[master["SKU_Standard"].astype(str) == sku]
        inv_latest = (
            inventory[inventory["SKU_Standard"].astype(str) == sku]
            .sort_values("Snapshot_Date")
            .tail(1)
        )
        risk_row = risk[risk["SKU_Standard"].astype(str) == sku]
        fc = provided[provided["SKU_Standard"].astype(str) == sku]
        return {
            "sku": sku,
            "master": master_row.to_dict(orient="records")[0] if not master_row.empty else {},
            "sales_summary": {
                "rows": int(len(sales_sku)),
                "total_units": float(sales_sku["Units_Sold"].sum()),
                "total_revenue": float(sales_sku["Revenue"].sum()),
                "date_min": str(sales_sku["Date"].min().date()),
                "date_max": str(sales_sku["Date"].max().date()),
            },
            "latest_inventory": inv_latest.to_dict(orient="records")[0] if not inv_latest.empty else {},
            "supplied_risk": risk_row.to_dict(orient="records")[0] if not risk_row.empty else {},
            "provided_forecast": fc.to_dict(orient="records"),
        }
