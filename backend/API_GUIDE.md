# FORESIGHT Flask API Guide

Base URL: `http://127.0.0.1:5000`

The API loads the **real** project extracts from `data/raw/` via `ml/preprocessing.py` and the services layer. It does **not** use `data/sample_sales.csv`.

## Endpoints

### `GET /health`
Returns service status and dataset row counts.

### `GET /overview`
Executive KPIs (units, revenue, SKUs, inventory value, risk counts, preprocess stats).

### `GET /forecast`
Query params:
- `horizon_weeks` (int, 1–12, default 4)
- `sku` (optional SKU_Standard filter)
- `cache=1` (optional — serve `data/processed/future_weekly_forecast_generated.csv` when present)

### `GET /inventory`
Query params:
- `horizon_weeks` (int, 1–12, default 4)

Returns explainable stockout/overstock/reorder recommendations.

### `GET /risk`
Latest inventory snapshot joined with supplied risk fields.

### `GET /evaluate`
Query params:
- `horizon_weeks` (int, 1–12, default 4)
- `folds` (int, 1–12, default 4)
- `sku` (optional)

Returns rolling-origin WAPE / MAE / RMSE for the ML model and seasonal-naive baseline.

### `GET /sku/<sku>`
SKU master row, sales summary, latest inventory, supplied risk, and a 4-week model forecast.

## Error handling

- `400` — invalid query parameters  
- `404` — unknown SKU or endpoint  
- `500` — unexpected server/data error  

All error bodies are JSON: `{"error": "..."}`.
