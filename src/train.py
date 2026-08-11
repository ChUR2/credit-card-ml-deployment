"""Обучение и сохранение моделей прогнозирования дефолта.

v1 (контрольная)  — LogisticRegression в пайплайне со StandardScaler.
v2 (тестовая)     — RandomForestClassifier, используется как challenger в A/B-тесте.

Обе модели сохраняются через joblib в виде «бандла»: пайплайн + метаданные
(список признаков, версия, метрики). Это защищает от рассинхронизации
порядка признаков между обучением и инференсом.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_V1_PATH,
    MODEL_V2_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from src.data import load_dataset, split_xy

CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ],
        remainder="drop",
    )


def build_pipeline(kind: str) -> Pipeline:
    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def evaluate(pipe: Pipeline, X_test, y_test) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "f1_default": round(float(f1_score(y_test, pred)), 4),
        "precision_default": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall_default": round(float(recall_score(y_test, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "positive_rate": round(float(pred.mean()), 4),
    }


def save_bundle(pipe: Pipeline, version: str, metrics: dict, path) -> None:
    bundle = {
        "model": pipe,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": version,
        "metrics": metrics,
        "threshold": 0.5,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_pipeline": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"[train] сохранено: {path} ({path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[train] train={len(X_train)}, test={len(X_test)}")

    all_metrics = {}
    for version, kind, path in [
        ("v1", "logreg", MODEL_V1_PATH),
        ("v2", "rf", MODEL_V2_PATH),
    ]:
        print(f"\n=== Модель {version} ({kind}) ===")
        pipe = build_pipeline(kind)
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_test, y_test)
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, pipe.predict(X_test), digits=3))
        save_bundle(pipe, version, metrics, path)
        all_metrics[version] = metrics

    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"\n[train] метрики записаны в {METRICS_PATH}")


if __name__ == "__main__":
    main()
