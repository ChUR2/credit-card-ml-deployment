from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))

# Доля трафика, уходящая на тестовую модель v2 (0.5 = сплит 50/50)
AB_TRAFFIC_SPLIT = float(os.getenv("AB_TRAFFIC_SPLIT", "0.5"))
DEFAULT_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "v1")

logger = logging.getLogger("credit_api")


class ValidationError(ValueError):
    "Некорректные входные данные (отдаём клиенту 400)"


class ModelRegistry:
    "Реестр версий моделей. Загружает бандлы один раз при старте сервиса"

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._bundles: dict[str, dict] = {}
        self._lock = threading.Lock()

    def load_all(self) -> dict[str, dict]:
        """Загружает все model_*.pkl из каталога моделей."""
        with self._lock:
            self._bundles.clear()
            for path in sorted(self.models_dir.glob("model_*.pkl")):
                version = path.stem.replace("model_", "")
                self._bundles[version] = self._load_bundle(path)
                logger.info(
                    "model loaded",
                    extra={"event": "model_loaded", "model_version": version, "path": str(path)},
                )
        if not self._bundles:
            raise FileNotFoundError(
                f"В {self.models_dir} нет файлов model_*.pkl. "
                f"Сначала обучите модель: python models/train_model.py"
            )
        return self._bundles

    @staticmethod
    def _load_bundle(path: Path) -> dict:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle:
            # Поддержка «голого» пайплайна без метаданных
            bundle = {"model": bundle, "feature_columns": None}
        bundle.setdefault("threshold", 0.5)
        bundle.setdefault("model_version", path.stem.replace("model_", ""))
        return bundle

    @property
    def versions(self) -> list[str]:
        return sorted(self._bundles)

    def is_ready(self) -> bool:
        return bool(self._bundles)

    def get(self, version: str) -> dict:
        if version not in self._bundles:
            raise ValidationError(
                f"Неизвестная версия модели '{version}'. Доступны: {self.versions}"
            )
        return self._bundles[version]

    def info(self) -> dict:
        return {
            v: {
                "model_version": b.get("model_version"),
                "estimator": type(b["model"].steps[-1][1]).__name__
                if hasattr(b["model"], "steps")
                else type(b["model"]).__name__,
                "trained_at": b.get("trained_at"),
                "threshold": b.get("threshold"),
                "metrics": b.get("metrics"),
            }
            for v, b in self._bundles.items()
        }

    #  A/B-роутинг 

    def resolve_version(self, requested: str | None, client_id: str | None) -> tuple[str, str]:
        if requested:
            requested = str(requested).lower()
            self.get(requested)  # проверка существования
            return requested, "explicit"

        if len(self.versions) < 2:
            return self.versions[0], "single_model"

        control, treatment = "v1", "v2"
        if client_id is not None:
            digest = hashlib.md5(str(client_id).encode()).hexdigest()
            bucket = int(digest[:8], 16) / 0xFFFFFFFF
            return (treatment if bucket < AB_TRAFFIC_SPLIT else control), "hash(client_id)"

        return (treatment if random.random() < AB_TRAFFIC_SPLIT else control), "random"

    # инференс 

    def predict(self, payload: dict, version: str) -> dict:
        bundle = self.get(version)
        features = validate_and_order(payload, bundle.get("feature_columns"))
        model = bundle["model"]

        proba = float(model.predict_proba(features)[0][1])
        threshold = float(bundle["threshold"])
        prediction = int(proba >= threshold)

        return {
            "prediction": prediction,
            "probability": round(proba, 6),
            "threshold": threshold,
            "risk_level": risk_level(proba),
            "model_version": version,
        }


def risk_level(proba: float) -> str:
    "Интерпретация вероятности для бизнес-пользователя"
    if proba < 0.2:
        return "low"
    if proba < 0.5:
        return "medium"
    return "high"


def validate_and_order(payload: dict, feature_columns: list[str] | None) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise ValidationError("Тело запроса должно быть JSON-объектом")

    if feature_columns is None:
        raise ValidationError("В бандле модели нет feature_columns — переобучите модель")

    missing = [c for c in feature_columns if c not in payload]
    if missing:
        raise ValidationError(f"Отсутствуют обязательные признаки: {missing}")

    values, bad_types = [], []
    for col in feature_columns:
        value = payload[col]
        if isinstance(value, bool) or value is None:
            bad_types.append(col)
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            bad_types.append(col)

    if bad_types:
        raise ValidationError(f"Признаки должны быть числами: {bad_types}")

    if not np.isfinite(values).all():
        raise ValidationError("Признаки содержат NaN или inf")

    return pd.DataFrame([values], columns=feature_columns)


# Единый экземпляр реестра на процесс
registry = ModelRegistry()
