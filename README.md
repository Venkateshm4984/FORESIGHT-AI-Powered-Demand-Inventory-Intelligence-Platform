# FORESIGHT — AI-Powered Demand & Inventory Intelligence Platform

**Domain:** Data Science & Analytics (Zidio project)  
**Scope:** This repository implements the combined responsibilities of all four team members — data collection & preprocessing, ML demand forecasting, Streamlit dashboard, and Flask backend integration — as one complete student project.

## Problem statement

Retailers need reliable SKU-level demand forecasts and clear inventory signals to avoid stockouts and overstock. FORESIGHT turns the supplied project datasets into weekly demand forecasts, rolling-origin evaluation, inventory risk scores, and an interactive decision-support dashboard backed by a REST API.

## Features

- Load and validate real project CSVs (no fabricated sales rows)
- Documented preprocessing pipeline with joins and weekly aggregation
- Weekly SKU-level Random Forest forecasting with lag / rolling / calendar / promo features
- Seasonal-naive baseline and rolling-origin backtesting (WAPE, MAE, RMSE)
- Explainable inventory intelligence (lead-time demand, coverage, reorder quantity)
- Professional Streamlit dashboard (six modules)
- Flask REST API with validation and JSON responses

## Architecture

```text
CSV (data/raw)
    → ml/preprocessing.py
    → services/ (data / forecast / inventory)
         ├─→ Streamlit app.py  (dashboard)
         └─→ Flask backend/app.py  (REST API)
    → data/processed/  (generated forecasts & metrics)
```

## Dataset description

Files live under `data/raw/`:

| File | Role | Approx. size |
|------|------|--------------|
| `sales_daily_clean.csv` | Daily sales, revenue, price, promotion | 36,550 rows × 7 cols · 50 SKUs · 2024-01-01 → 2025-12-31 |
| `sku_master_clean.csv` | Product master | 5,000 products (filtered to 50 sales-scope SKUs) |
| `calendar_clean.csv` | Season / holiday / promo events | 731 days |
| `inventory_snapshots_clean.csv` | Stock, lead time, safety stock, ROP | 4,800 snapshots (200 SKUs; 1,200 after sales-scope filter) |
| `future_weekly_forecast.csv` | Supplied reference forecast | 200 rows (50 SKUs × 4 weeks) |
| `inventory_risk_scores.csv` | Supplied risk reference | 50 rows |

`data/sample_sales.csv` is a tiny synthetic demo file and is **not** used by the main pipeline.

## Data preprocessing (Member 1)

Implemented in `ml/preprocessing.py` (not inside the Streamlit UI):

1. Parse dates; normalize `SKU_Standard`
2. Coerce numeric fields; clip invalid negative units
3. Deduplicate `Date + SKU_Standard` if present
4. Filter product master and inventory to the 50 sales-scope SKUs
5. Join sales ⋈ master ⋈ calendar
6. Aggregate daily sales → weekly SKU demand (`W-SUN`)
7. Emit a `PreprocessReport` documenting every important step

## ML & forecasting methodology (Member 2)

- Aggregate daily units to weekly SKU demand
- Features (all lagged / shifted to avoid leakage): lag 1/4/8/13/52, rolling means 4/8/13, week-of-year, month, quarter, promo share, holiday & promotion-event flags
- Model: `RandomForestRegressor` per SKU (recursive multi-step forecast)
- Baseline: seasonal-naive (same week prior year when ≥52 weeks exist)
- Evaluation: chronological rolling-origin backtesting with WAPE, MAE, RMSE
- Outputs written to `data/processed/`

## Inventory intelligence

Documented rules in `ml/forecasting.py` → `build_inventory_recommendations`:

- Available stock = current + on-order  
- Weekly forecast = horizon forecast ÷ horizon weeks  
- Lead-time demand = weekly forecast × (lead time days / 7)  
- Stockout if available &lt; lead-time demand  
- Overstock if coverage &gt; 8 weeks  
- Recommended order = max(0, ceil(ROP − available))

## Backend API (Member 4)

Base URL: `http://127.0.0.1:5000`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service + dataset health |
| GET | `/overview` | Executive KPIs |
| GET | `/forecast?horizon_weeks=4&sku=SKU00001` | Weekly forecasts |
| GET | `/inventory?horizon_weeks=4` | Inventory recommendations |
| GET | `/risk` | Latest inventory / risk view |
| GET | `/evaluate?horizon_weeks=4&folds=4` | Rolling-origin metrics |
| GET | `/sku/<sku>` | SKU detail + short forecast |

## Dashboard (Member 3)

`streamlit run app.py` opens six tabs:

1. **Executive Overview** — sales, SKUs, stock, risk, forecast summary  
2. **Demand Analysis** — daily/weekly trends, categories, promo/holiday effects  
3. **Forecasting** — SKU selector, horizon, ML vs seasonal-naive  
4. **Inventory Intelligence** — stockout/overstock, reorder actions  
5. **SKU Analysis** — history, forecast, stock, ROP, risk  
6. **Model Evaluation** — WAPE / MAE / RMSE vs baseline  

Metrics are computed from data/models — not hardcoded.

## Project structure

```text
foresight_demand_inventory/
├── app.py                 # Streamlit dashboard
├── requirements.txt
├── README.md
├── .gitignore
├── backend/
│   └── app.py             # Flask API
├── data/
│   ├── pipeline.py
│   ├── raw/               # supplied CSVs
│   └── processed/         # generated outputs
├── ml/
│   ├── preprocessing.py
│   ├── forecasting.py
│   └── evaluation.py
├── services/
│   ├── data_service.py
│   ├── forecast_service.py
│   └── inventory_service.py
├── scripts/
│   └── run_pipeline.py
├── reports/
└── tests/
    ├── test_data.py
    ├── test_forecasting.py
    └── test_api.py
```

## Installation (Windows)

```powershell
cd foresight_demand_inventory
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Running

### Dashboard

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

### Flask API

```powershell
python backend/app.py
```

### Regenerate processed artefacts

```powershell
python scripts/run_pipeline.py
```

### Tests

```powershell
pytest -q
```

## API examples

```powershell
curl http://127.0.0.1:5000/health
curl "http://127.0.0.1:5000/forecast?horizon_weeks=4&sku=SKU00001"
curl "http://127.0.0.1:5000/inventory?horizon_weeks=4"
curl http://127.0.0.1:5000/risk
curl "http://127.0.0.1:5000/evaluate?horizon_weeks=4&folds=4"
curl http://127.0.0.1:5000/sku/SKU00001
```

## Team responsibilities (combined in this repo)

| Member | Responsibility | Primary modules |
|--------|----------------|-----------------|
| 1 | Data collection & preprocessing | `ml/preprocessing.py`, `data/pipeline.py` |
| 2 | ML & demand forecasting | `ml/forecasting.py`, `ml/evaluation.py`, `services/forecast_service.py` |
| 3 | Dashboard & visualization | `app.py` |
| 4 | Backend & integration | `backend/app.py`, `services/` |

## Important notes

- Forecasts and inventory recommendations are analytical estimates for the student project demo.
- Do not publish confidential raw CSVs to a public repository unless explicitly permitted (see `.gitignore`).
- Never treat fabricated metrics as scored accuracy — evaluation numbers come from `rolling_origin_backtest`.
