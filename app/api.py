"""Flask-сервис прогнозирования дефолта по кредитной карте

Эндпоинты:
    GET  /health   — проверка живости сервиса и готовности моделей
    GET  /models   — список загруженных версий модели и их метрики
    POST /predict  — прогноз по одному клиенту
    POST /predict/batch — прогноз по списку клиентов (до 1000 записей)
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.logging_config import setup_logging  # noqa: E402
from app.model_handler import ValidationError, registry  # noqa: E402

MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))

app = Flask(__name__)
app.json.ensure_ascii = False  # сообщения об ошибках на русском читаемы как есть
logger = setup_logging()

# Модели грузятся один раз при импорте модуля — не на каждый запрос.
try:
    registry.load_all()
    logger.info("models loaded", extra={"event": "startup", "versions": registry.versions})
except FileNotFoundError as exc:
    logger.error(str(exc), extra={"event": "startup_failed"})


@app.before_request
def _start_timer():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    g.started = time.perf_counter()


@app.after_request
def _log_request(response):
    if request.path != "/health":
        logger.info(
            "request",
            extra={
                "event": "request",
                "request_id": getattr(g, "request_id", None),
                "method": request.method,
                "path": request.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - g.started) * 1000, 2),
                "remote_addr": request.remote_addr,
            },
        )
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return response


@app.get("/health")
def health():
    """Liveness + readiness. 503, если модели не загрузились."""
    ready = registry.is_ready()
    body = {
        "status": "healthy" if ready else "unhealthy",
        "models_loaded": registry.versions,
        "service": "credit-default-prediction",
    }
    return jsonify(body), (200 if ready else 503)


@app.get("/models")
def models():
    """Метаданные и офлайн-метрики загруженных версий."""
    return jsonify({"models": registry.info(), "default_version": "v1"}), 200


@app.post("/predict")
def predict():
    """Прогноз дефолта.

    Тело запроса: JSON с 23 признаками. Опционально:
        model_version — 'v1' или 'v2' (явный выбор, минуя A/B-роутинг)
        client_id     — для детерминированного распределения по группам A/B
    """
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValidationError("Ожидается JSON (заголовок Content-Type: application/json)")

    requested = payload.pop("model_version", None) or request.args.get("version")
    client_id = payload.pop("client_id", None)

    version, routing = registry.resolve_version(requested, client_id)
    result = registry.predict(payload, version)
    result["ab_group"] = "control" if version == "v1" else "treatment"
    result["request_id"] = g.request_id

    logger.info(
        "prediction",
        extra={
            "event": "prediction",
            "request_id": g.request_id,
            "client_id": client_id,
            "model_version": version,
            "ab_group": result["ab_group"],
            "routing": routing,
            "prediction": result["prediction"],
            "probability": result["probability"],
        },
    )
    return jsonify(result), 200


@app.post("/predict/batch")
def predict_batch():
    """Пакетный прогноз: {'clients': [{...}, {...}]}."""
    payload = request.get_json(silent=True) or {}
    clients = payload.get("clients")
    if not isinstance(clients, list) or not clients:
        raise ValidationError("Ожидается непустой список в поле 'clients'")
    if len(clients) > MAX_BATCH_SIZE:
        raise ValidationError(f"Максимум {MAX_BATCH_SIZE} записей за запрос")

    requested = payload.get("model_version") or request.args.get("version")
    results = []
    for i, client in enumerate(clients):
        client = dict(client)
        client_id = client.pop("client_id", None)
        client.pop("model_version", None)
        version, _ = registry.resolve_version(requested, client_id)
        item = registry.predict(client, version)
        item["index"] = i
        item["client_id"] = client_id
        results.append(item)

    logger.info(
        "batch_prediction",
        extra={
            "event": "batch_prediction",
            "request_id": g.request_id,
            "count": len(results),
        },
    )
    return jsonify({"count": len(results), "results": results}), 200


# ---------- обработчики ошибок: клиент всегда получает JSON ----------


@app.errorhandler(ValidationError)
def handle_validation_error(exc: ValidationError):
    logger.warning(
        "validation_error",
        extra={"event": "validation_error", "request_id": getattr(g, "request_id", None),
               "detail": str(exc)},
    )
    return jsonify({"error": "validation_error", "detail": str(exc)}), 400


@app.errorhandler(404)
def handle_404(_):
    return jsonify({"error": "not_found", "detail": "Неизвестный эндпоинт"}), 404


@app.errorhandler(405)
def handle_405(_):
    return jsonify({"error": "method_not_allowed"}), 405


@app.errorhandler(Exception)
def handle_unexpected(exc: Exception):
    logger.exception(
        "internal_error",
        extra={"event": "internal_error", "request_id": getattr(g, "request_id", None)},
    )
    return jsonify({"error": "internal_error", "detail": str(exc)}), 500


if __name__ == "__main__":
    # Dev-режим. В production используется gunicorn (см. Dockerfile).
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
