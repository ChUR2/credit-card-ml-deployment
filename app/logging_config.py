"""Структурированное JSON-логирование.

Логи пишутся в stdout одной строкой на событие — это стандарт для контейнеров
(12-factor app): приложение не управляет ротацией файлов, а поток stdout
забирает Docker и передаёт дальше в сборщик (Filebeat/Fluentd -> Elasticsearch -> Kibana).

Пример строки лога:
{"ts": "2026-08-11T09:12:33Z", "level": "INFO", "event": "prediction",
 "request_id": "...", "model_version": "v2", "prediction": 1, "latency_ms": 3.4}
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Всё, что передано через logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(name: str = "credit_api") -> logging.Logger:
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    # Логи werkzeug тоже в JSON, чтобы формат был единым
    werkzeug = logging.getLogger("werkzeug")
    werkzeug.handlers.clear()
    werkzeug.addHandler(handler)
    werkzeug.setLevel(logging.WARNING)

    return logger
