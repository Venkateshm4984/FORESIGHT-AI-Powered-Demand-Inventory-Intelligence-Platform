"""
FORESIGHT Streamlit dashboard — Demand & Inventory Intelligence.

Combines Member 1–4 deliverables: preprocessing overview, demand analysis,
forecasting, inventory intelligence, SKU detail, and model evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.pipeline import persist_pipeline_outputs
from services.data_service import DataService
from services.forecast_service import ForecastService
from services.inventory_service import InventoryService

st.set_page_config(
    page_title="FORESIGHT | Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
)

# Visual direction: deep slate + teal (high contrast theme-aware styling)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px; }

/* Hero Banner */
.hero {
  padding: 1.5rem 1.75rem;
  border-radius: 18px;
  background:
    radial-gradient(1200px 400px at 10% -20%, rgba(20,184,166,0.35), transparent 55%),
    linear-gradient(125deg, #0f172a 0%, #134e4a 55%, #0f766e 100%);
  color: #ffffff !important;
  margin-bottom: 1.25rem;
  border: 1px solid rgba(255, 255, 255, 0.15);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
}
.hero h1 {
  margin: 0;
  font-family: 'Fraunces', Georgia, serif;
  font-size: 2.35rem;
  letter-spacing: -0.02em;
  color: #ffffff !important;
}
.hero p {
  margin: 0.4rem 0 0;
  color: #2dd4bf !important;
  font-size: 1.05rem;
  font-weight: 500;
}
.metric-note { color: var(--text-color); opacity: 0.75; font-size: 0.85rem; }

/* Metric Cards - Theme Aware & High Contrast */
div[data-testid="stMetric"] {
  background-color: var(--secondary-background-color, #1e293b) !important;
  border: 1px solid rgba(148, 163, 184, 0.25) !important;
  border-radius: 14px !important;
  padding: 0.85rem 1rem !important;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08) !important;
}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
  color: var(--text-color, #f8fafc) !important;
  font-size: 0.88rem !important;
  font-weight: 600 !important;
  opacity: 0.9 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--text-color, #ffffff) !important;
  font-size: 1.55rem !important;
  font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
  font-weight: 600 !important;
}

/* Tabs Styling */
button[data-baseweb="tab"] {
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  padding: 0.6rem 1.1rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: #2dd4bf !important;
  border-bottom-color: #2dd4bf !important;
  font-weight: 700 !important;
}

/* Expander & Cards */
.stExpander {
  border: 1px solid rgba(148, 163, 184, 0.2) !important;
  border-radius: 10px !important;
}
</style>
<div class="hero">
  <h1>FORESIGHT</h1>
  <p>AI-Powered Demand &amp; Inventory Intelligence Platform</p>
</div>
""",
    unsafe_allow_html=True,
)

data_svc = DataService()
forecast_svc = ForecastService()
inventory_svc = InventoryService()

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif"),
    xaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.15)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
    ),
    yaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.15)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
    ),
    margin=dict(l=40, r=20, t=50, b=40),
)
TEAL = "#14b8a6"
AMBER = "#f59e0b"
SLATE = "#94a3b8"



@st.cache_data(show_spinner="Loading project datasets…")
def cached_overview():
    return data_svc.overview()


@st.cache_data(show_spinner="Loading sales…")
def cached_sales():
    return data_svc.sales()


@st.cache_data(show_spinner="Loading master…")
def cached_master():
    return data_svc.master()


@st.cache_data(show_spinner="Loading calendar…")
def cached_calendar():
    return data_svc.calendar()


@st.cache_data(show_spinner="Loading inventory…")
def cached_inventory():
    return data_svc.inventory()


@st.cache_data(show_spinner="Loading weekly demand…")
def cached_weekly():
    return data_svc.weekly()


@st.cache_data(show_spinner="Loading joined daily…")
def cached_joined():
    return data_svc.joined_daily()


@st.cache_data(show_spinner="Generating weekly forecast…")
def cached_forecast(horizon: int):
    return forecast_svc.forecast(horizon_weeks=horizon)


@st.cache_data(show_spinner="Running rolling-origin backtest…")
def cached_backtest(horizon: int, folds: int):
    return forecast_svc.evaluate(horizon_weeks=horizon, folds=folds)


@st.cache_data(show_spinner="Building inventory recommendations…")
def cached_recommendations(horizon: int):
    fc = cached_forecast(horizon)
    return inventory_svc.recommendations(horizon_weeks=horizon, forecast=fc)


overview = cached_overview()
sales = cached_sales()
master = cached_master()
calendar = cached_calendar()
inventory = cached_inventory()
weekly = cached_weekly()
joined = cached_joined()

with st.sidebar:
    st.header("Controls")
    horizon = st.slider("Forecast horizon (weeks)", 1, 8, 4)
    folds = st.slider("Backtest folds", 1, 8, 4)
    st.caption("Real project extracts under `data/raw/`")
    st.divider()
    st.write(f"Sales rows: **{overview['sales_rows']:,}**")
    st.write(f"Active SKUs: **{overview['sku_count']}**")
    st.write(f"Date range: **{overview['date_min']} → {overview['date_max']}**")
    st.write(f"Inventory snapshots: **{overview['inventory_snapshots']:,}**")
    if st.button("Persist outputs to data/processed"):
        fc = cached_forecast(horizon)
        bt = cached_backtest(horizon, folds)
        rec = cached_recommendations(horizon)
        paths = persist_pipeline_outputs(forecast=fc, recommendations=rec, backtest=bt)
        st.success(f"Wrote {len(paths)} files to data/processed/")

tabs = st.tabs(
    [
        "Executive Overview",
        "Demand Analysis",
        "Forecasting",
        "Inventory Intelligence",
        "SKU Analysis",
        "Model Evaluation",
    ]
)

# ---------------------------------------------------------------------------
# 1. Executive Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total units sold", f"{overview['total_units']:,.0f}")
    c2.metric("Total revenue", f"₹{overview['total_revenue']:,.0f}")
    c3.metric("Active SKUs", f"{overview['sku_count']}")
    c4.metric("Current stock", f"{overview['current_stock_units']:,.0f}")
    c5.metric("Inventory value", f"₹{overview['inventory_value']:,.0f}")

    r1, r2, r3 = st.columns(3)
    r1.metric("Stockout-risk SKUs", overview["stockout_risk_skus"])
    r2.metric("Overstock-risk SKUs", overview["overstock_risk_skus"])
    r3.metric("Categories", overview["categories"])

    daily_trend = sales.groupby("Date", as_index=False)["Units_Sold"].sum()
    fig = px.area(
        daily_trend,
        x="Date",
        y="Units_Sold",
        title="Daily demand (all SKUs)",
        color_discrete_sequence=[TEAL],
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    fc = cached_forecast(horizon)
    st.subheader("Forecast summary")
    f1, f2, f3 = st.columns(3)
    f1.metric("SKUs forecasted", fc["SKU_Standard"].nunique())
    f2.metric("Horizon weeks", horizon)
    f3.metric("Total forecast units", f"{fc['Forecast_Units'].sum():,.0f}")

    with st.expander("Preprocessing report"):
        report = overview["preprocess"]
        for step in report.get("steps", []):
            st.write(f"• {step}")
        for warn in report.get("warnings", []):
            st.warning(warn)
        st.json(report.get("stats", {}))

# ---------------------------------------------------------------------------
# 2. Demand Analysis
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Sales trends & category demand")
    grain = st.radio("Aggregation", ["Daily", "Weekly"], horizontal=True)
    if grain == "Daily":
        trend = sales.groupby("Date", as_index=False)["Units_Sold"].sum()
        fig = px.line(trend, x="Date", y="Units_Sold", title="Daily units sold", color_discrete_sequence=[TEAL])
    else:
        trend = weekly.groupby("Date", as_index=False)["Units_Sold"].sum()
        fig = px.line(trend, x="Date", y="Units_Sold", title="Weekly units sold", color_discrete_sequence=[TEAL])
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    cat = (
        joined.groupby("category", as_index=False)["Units_Sold"]
        .sum()
        .sort_values("Units_Sold", ascending=False)
    )
    fig_cat = px.bar(
        cat,
        x="category",
        y="Units_Sold",
        title="Demand by category",
        color_discrete_sequence=[AMBER],
    )
    fig_cat.update_layout(**PLOTLY_LAYOUT, xaxis_tickangle=-30)
    st.plotly_chart(fig_cat, use_container_width=True)

    st.subheader("Promotion & holiday effects")
    promo = joined.groupby("Promotion", as_index=False)["Units_Sold"].mean()
    promo["Promotion"] = promo["Promotion"].map({0: "No promo", 1: "Promo day"})
    hol = joined.groupby("is_holiday", as_index=False)["Units_Sold"].mean()
    hol["is_holiday"] = hol["is_holiday"].map({0: "Regular day", 1: "Holiday"})
    p1, p2 = st.columns(2)
    fig_p = px.bar(promo, x="Promotion", y="Units_Sold", title="Avg daily units — promotion", color_discrete_sequence=[TEAL])
    fig_p.update_layout(**PLOTLY_LAYOUT)
    p1.plotly_chart(fig_p, use_container_width=True)
    fig_h = px.bar(hol, x="is_holiday", y="Units_Sold", title="Avg daily units — holiday", color_discrete_sequence=[SLATE])
    fig_h.update_layout(**PLOTLY_LAYOUT)
    p2.plotly_chart(fig_h, use_container_width=True)

    top_skus = (
        sales.groupby("SKU_Standard", as_index=False)["Units_Sold"]
        .sum()
        .sort_values("Units_Sold", ascending=False)
        .head(15)
        .merge(master[["SKU_Standard", "sku_name", "category"]], on="SKU_Standard", how="left")
    )
    st.dataframe(top_skus, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Forecasting
# ---------------------------------------------------------------------------
with tabs[2]:
    forecast = cached_forecast(horizon)
    st.subheader("Weekly demand forecast")
    m1, m2, m3 = st.columns(3)
    m1.metric("SKUs", forecast["SKU_Standard"].nunique())
    m2.metric("Horizon", f"{horizon} weeks")
    m3.metric("Forecast units", f"{forecast['Forecast_Units'].sum():,.0f}")

    selected = st.selectbox("Select SKU", sorted(forecast["SKU_Standard"].unique()), key="fc_sku")
    hist = weekly[weekly["SKU_Standard"] == selected][["Date", "Units_Sold"]].copy()
    hist["Type"] = "Actual"
    pred = forecast[forecast["SKU_Standard"] == selected][["Date", "Forecast_Units"]].rename(
        columns={"Forecast_Units": "Units_Sold"}
    )
    pred["Type"] = "ML Forecast"
    naive = forecast[forecast["SKU_Standard"] == selected][["Date", "Seasonal_Naive_Units"]].rename(
        columns={"Seasonal_Naive_Units": "Units_Sold"}
    )
    naive["Type"] = "Seasonal-naive"
    chart = pd.concat([hist, pred, naive], ignore_index=True)
    fig = px.line(
        chart,
        x="Date",
        y="Units_Sold",
        color="Type",
        title=f"Historical weekly demand vs {horizon}-week forecast — {selected}",
        color_discrete_map={"Actual": TEAL, "ML Forecast": AMBER, "Seasonal-naive": SLATE},
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast[forecast["SKU_Standard"] == selected], use_container_width=True)
    st.download_button(
        "Download forecast CSV",
        forecast.to_csv(index=False),
        "future_weekly_forecast_generated.csv",
        "text/csv",
    )

# ---------------------------------------------------------------------------
# 4. Inventory Intelligence
# ---------------------------------------------------------------------------
with tabs[3]:
    rec = cached_recommendations(horizon)
    st.subheader("Inventory risk & reorder recommendations")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Stockout risk", int(rec["Stockout_Risk"].sum()))
    i2.metric("Overstock risk", int(rec["Overstock_Risk"].sum()))
    i3.metric("Plan reorder", int((rec["Recommended_Action"] == "Plan Reorder").sum()))
    i4.metric("Inventory value", f"₹{rec['Inventory_Value'].sum():,.0f}")

    with st.expander("Business rules used"):
        st.markdown(
            """
- **Available stock** = Current stock + On order  
- **Weekly forecast** = Horizon forecast units ÷ horizon weeks  
- **Lead-time demand** = Weekly forecast × (Lead time days ÷ 7)  
- **Stockout risk** when available stock &lt; lead-time demand  
- **Overstock risk** when coverage &gt; 8 weeks  
- **Recommended order** = max(0, ceil(Reorder point − available stock))  
- Action priority: Stockout → Overstock → Plan Reorder → Watch / Healthy  
            """
        )

    view_cols = [
        c
        for c in [
            "SKU_Standard",
            "sku_name",
            "category",
            "Current_Stock",
            "On_Order",
            "Lead_Time_Days",
            "Safety_Stock",
            "Reorder_Point",
            "Forecast_Horizon_Units",
            "Forecast_Weekly_Units",
            "Lead_Time_Demand",
            "Available_Stock",
            "Stock_Coverage_Weeks",
            "Stockout_Risk",
            "Overstock_Risk",
            "Recommended_Action",
            "Recommended_Order",
        ]
        if c in rec.columns
    ]
    display = rec[view_cols].copy()
    display["Stock_Coverage_Weeks"] = display["Stock_Coverage_Weeks"].replace([float("inf")], None).round(2)
    st.dataframe(display, use_container_width=True)

    action_counts = rec["Recommended_Action"].value_counts().reset_index()
    action_counts.columns = ["Action", "SKUs"]
    fig_a = px.bar(action_counts, x="Action", y="SKUs", title="Recommended actions", color_discrete_sequence=[TEAL])
    fig_a.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_a, use_container_width=True)
    st.download_button(
        "Download recommendations CSV",
        rec.to_csv(index=False),
        "inventory_recommendations.csv",
        "text/csv",
    )

# ---------------------------------------------------------------------------
# 5. SKU-level Analysis
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("SKU deep dive")
    sku = st.selectbox("Select SKU", sorted(sales["SKU_Standard"].unique()), key="sku_page")
    detail = data_svc.sku_detail(sku)
    master_row = detail["master"]
    inv_row = detail["latest_inventory"]
    risk_row = detail["supplied_risk"]

    st.markdown(
        f"**{master_row.get('sku_name', sku)}** · {master_row.get('category', '—')} / "
        f"{master_row.get('subcategory', '—')} · Brand: {master_row.get('brand', '—')}"
    )
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Historical units", f"{detail['sales_summary']['total_units']:,.0f}")
    s2.metric("Current stock", f"{inv_row.get('Current_Stock', 0):,.0f}")
    s3.metric("On order", f"{inv_row.get('On_Order', 0):,.0f}")
    s4.metric("Lead time (days)", inv_row.get("Lead_Time_Days", "—"))
    s5.metric("Reorder point", inv_row.get("Reorder_Point", "—"))

    t1, t2, t3 = st.columns(3)
    t1.metric("Safety stock", inv_row.get("Safety_Stock", "—"))
    t2.metric(
        "Stockout risk (supplied)",
        str(risk_row.get("Stockout_Risk", "—")),
    )
    t3.metric("Action (supplied)", risk_row.get("Recommended_Action", "—"))

    hist = weekly[weekly["SKU_Standard"] == sku][["Date", "Units_Sold"]].copy()
    hist["Series"] = "Actual"
    fc_sku = cached_forecast(horizon)
    fc_sku = fc_sku[fc_sku["SKU_Standard"] == sku][["Date", "Forecast_Units"]].rename(
        columns={"Forecast_Units": "Units_Sold"}
    )
    fc_sku["Series"] = "Forecast"
    chart = pd.concat([hist, fc_sku], ignore_index=True)
    fig = go.Figure()
    for series, color in [("Actual", TEAL), ("Forecast", AMBER)]:
        part = chart[chart["Series"] == series]
        fig.add_trace(
            go.Scatter(
                x=part["Date"],
                y=part["Units_Sold"],
                mode="lines+markers",
                name=series,
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(title=f"Demand & forecast — {sku}", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    rec = cached_recommendations(horizon)
    rec_sku = rec[rec["SKU_Standard"] == sku]
    if not rec_sku.empty:
        st.write("Model-based inventory recommendation")
        st.dataframe(rec_sku, use_container_width=True)

# ---------------------------------------------------------------------------
# 6. Model Evaluation
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Rolling-origin backtesting")
    st.caption(
        "Chronological folds only — training ends at each cutoff; future weeks are never used as features. "
        "WAPE = Σ|actual − forecast| / Σ|actual|."
    )
    result = cached_backtest(horizon, folds)
    if result["summary"]:
        s = result["summary"]
        a, b, c, d, e = st.columns(5)
        a.metric("Folds evaluated", s.get("folds_evaluated", 0))
        b.metric("SKUs", s.get("sku_count", 0))
        c.metric("Model WAPE", f'{s.get("mean_model_wape_percent", 0):.2f}%')
        d.metric("Seasonal-naive WAPE", f'{s.get("mean_seasonal_naive_wape_percent", 0):.2f}%')
        e.metric("Model RMSE", f'{s.get("mean_model_rmse", 0):.2f}')

        m1, m2 = st.columns(2)
        m1.metric("Model MAE", f'{s.get("mean_model_mae", 0):.2f}')
        if "mean_seasonal_naive_mae" in s:
            m2.metric("Seasonal-naive MAE", f'{s["mean_seasonal_naive_mae"]:.2f}')

        detail = result["detail"]
        fig = px.box(
            detail.melt(
                id_vars=["SKU_Standard", "cutoff"],
                value_vars=["model_wape_percent", "seasonal_naive_wape_percent"],
                var_name="Model",
                value_name="WAPE %",
            ),
            x="Model",
            y="WAPE %",
            title="WAPE distribution across folds",
            color="Model",
            color_discrete_sequence=[TEAL, AMBER],
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download backtest CSV",
            detail.to_csv(index=False),
            "rolling_origin_backtest.csv",
            "text/csv",
        )
    else:
        st.warning("Not enough weekly history for the selected backtest settings.")
