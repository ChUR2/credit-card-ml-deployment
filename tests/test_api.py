"Тесты API. Запуск из корня проекта: pytest -v"
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api import app  # noqa: E402

EXAMPLE = json.loads((Path(__file__).parents[1] / "examples" / "request_example.json").read_text())


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "healthy"
    assert "v1" in r.get_json()["models_loaded"]


def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    assert set(r.get_json()["models"]) >= {"v1", "v2"}


def test_predict_ok(client):
    r = client.post("/predict", json=dict(EXAMPLE))
    assert r.status_code == 200
    body = r.get_json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] in ("v1", "v2")
    assert body["risk_level"] in ("low", "medium", "high")


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_explicit_version(client, version):
    payload = dict(EXAMPLE, model_version=version)
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert r.get_json()["model_version"] == version


def test_ab_split_is_deterministic(client):
    """Один и тот же client_id всегда попадает в одну группу."""
    payload = dict(EXAMPLE, client_id="client-42")
    versions = {client.post("/predict", json=dict(payload)).get_json()["model_version"]
                for _ in range(5)}
    assert len(versions) == 1


def test_ab_split_distributes_traffic(client):
    """На большом числе разных client_id используются обе версии."""
    versions = set()
    for i in range(200):
        payload = dict(EXAMPLE, client_id=f"client-{i}")
        versions.add(client.post("/predict", json=payload).get_json()["model_version"])
    assert versions == {"v1", "v2"}


def test_missing_features(client):
    payload = dict(EXAMPLE)
    payload.pop("AGE")
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert "AGE" in r.get_json()["detail"]


def test_wrong_type(client):
    payload = dict(EXAMPLE, LIMIT_BAL="много")
    r = client.post("/predict", json=payload)
    assert r.status_code == 400
    assert r.get_json()["error"] == "validation_error"


def test_unknown_version(client):
    r = client.post("/predict", json=dict(EXAMPLE, model_version="v99"))
    assert r.status_code == 400


def test_empty_body(client):
    r = client.post("/predict", data="", content_type="application/json")
    assert r.status_code == 400


def test_batch(client):
    r = client.post("/predict/batch", json={"clients": [dict(EXAMPLE), dict(EXAMPLE)],
                                            "model_version": "v2"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["count"] == 2
    assert all(item["model_version"] == "v2" for item in body["results"])


def test_unknown_endpoint(client):
    assert client.get("/nope").status_code == 404
