"""
FORESIGHT Flask REST API.

Exposes processed forecasts, inventory intelligence, evaluation metrics,
and SKU-level detail. Business logic lives in services/ and ml/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.data_service import DataService
from services.forecast_service import ForecastService
from services.inventory_service import InventoryService

app = Flask(__name__)
data_service = DataService()
forecast_service = ForecastService()
inventory_service = InventoryService()


def _parse_int(name: str, default: int, lo: int, hi: int):
    raw = request.args.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None, (jsonify({"error": f"{name} must be an integer"}), 400)
    if not lo <= value <= hi:
        return None, (jsonify({"error": f"{name} must be between {lo} and {hi}"}), 400)
    return value, None


def _json_records(df):
    if df is None or df.empty:
        return []
    out = df.copy()
    for col in out.select_dtypes(include=["datetimetz", "datetime"]).columns:
        out[col] = out[col].astype(str)
    # Replace inf for JSON safety
    out = out.replace([float("inf"), float("-inf")], None)
    return out.to_dict(orient="records")


@app.get("/health")
def health():
    try:
        overview = data_service.overview()
        return jsonify(
            {
                "status": "ok",
                "service": "FORESIGHT Flask API",
                "sales_rows": overview["sales_rows"],
                "forecast_skus": overview["sku_count"],
                "inventory_rows": overview["inventory_snapshots"],
                "date_range": [overview["date_min"], overview["date_max"]],
            }
        )
    except Exception as exc:  # noqa: BLE001 — surface as API error
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.get("/forecast")
def forecast():
    horizon, err = _parse_int("horizon_weeks", 4, 1, 12)
    if err:
        return err
    sku = request.args.get("sku")
    try:
        # Prefer fresh generation; fall back to cached if requested
        use_cache = request.args.get("cache", "0") == "1"
        if use_cache and sku is None:
            cached = forecast_service.load_cached_forecast()
            if cached is not None:
                return jsonify(
                    {
                        "horizon_weeks": horizon,
                        "source": "cache",
                        "forecasts": _json_records(cached),
                    }
                )
        f = forecast_service.forecast(horizon_weeks=horizon, sku=sku)
        return jsonify(
            {
                "horizon_weeks": horizon,
                "sku": sku,
                "source": "model",
                "count": int(len(f)),
                "forecasts": _json_records(f),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/inventory")
def inventory():
    horizon, err = _parse_int("horizon_weeks", 4, 1, 12)
    if err:
        return err
    try:
        rec = inventory_service.recommendations(horizon_weeks=horizon)
        num_cols = rec.select_dtypes(include=["number"]).columns
        rec_out = rec.copy()
        rec_out[num_cols] = rec_out[num_cols].round(2)
        return jsonify(
            {
                "horizon_weeks": horizon,
                "count": int(len(rec_out)),
                "recommendations": _json_records(rec_out),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/risk")
def risk():
    try:
        view = inventory_service.risk_view()
        return jsonify({"count": int(len(view)), "risk_scores": _json_records(view)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/evaluate")
def evaluate():
    horizon, err = _parse_int("horizon_weeks", 4, 1, 12)
    if err:
        return err
    folds, err = _parse_int("folds", 4, 1, 12)
    if err:
        return err
    sku = request.args.get("sku")
    try:
        result = forecast_service.evaluate(horizon_weeks=horizon, folds=folds, sku=sku)
        return jsonify(
            {
                "horizon_weeks": horizon,
                "folds": folds,
                "summary": result["summary"],
                "detail": _json_records(result["detail"]),
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/sku/<sku>")
def sku_detail(sku: str):
    try:
        detail = data_service.sku_detail(sku)
        # Attach a short live forecast for convenience
        fc = forecast_service.forecast(horizon_weeks=4, sku=sku)
        detail["model_forecast"] = _json_records(fc)
        return jsonify(detail)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/overview")
def overview():
    try:
        return jsonify(data_service.overview())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(404)
def not_found(_err):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_err):
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
