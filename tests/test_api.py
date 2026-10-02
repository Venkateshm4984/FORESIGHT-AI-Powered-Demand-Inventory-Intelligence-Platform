"""Tests for FORESIGHT Flask API."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    assert body["sales_rows"] == 36550
    assert body["forecast_skus"] == 50


def test_forecast_validation(client):
    res = client.get("/forecast?horizon_weeks=99")
    assert res.status_code == 400
    res = client.get("/forecast?horizon_weeks=abc")
    assert res.status_code == 400


def test_forecast_single_sku(client):
    res = client.get("/forecast?horizon_weeks=2&sku=SKU00001")
    assert res.status_code == 200
    body = res.get_json()
    assert body["count"] == 2
    assert all(r["SKU_Standard"] == "SKU00001" for r in body["forecasts"])


def test_risk(client):
    res = client.get("/risk")
    assert res.status_code == 200
    body = res.get_json()
    assert body["count"] == 50


def test_sku_not_found(client):
    res = client.get("/sku/DOES_NOT_EXIST")
    assert res.status_code == 404


def test_sku_detail(client):
    res = client.get("/sku/SKU00001")
    assert res.status_code == 200
    body = res.get_json()
    assert body["sku"] == "SKU00001"
    assert "sales_summary" in body
    assert "model_forecast" in body
    assert len(body["model_forecast"]) == 4


def test_evaluate_validation(client):
    res = client.get("/evaluate?folds=0")
    assert res.status_code == 400


def test_overview(client):
    res = client.get("/overview")
    assert res.status_code == 200
    body = res.get_json()
    assert body["sku_count"] == 50


def test_inventory_endpoint(client):
    # Uses a short horizon; still trains models — allow generous timeout via pytest default
    res = client.get("/inventory?horizon_weeks=2")
    assert res.status_code == 200
    body = res.get_json()
    assert body["count"] >= 1
    assert "recommendations" in body
