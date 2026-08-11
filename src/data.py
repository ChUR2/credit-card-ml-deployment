"""Загрузка датасета Default of Credit Card Clients.

Приоритет источников:
1. Локальный файл data/raw/UCI_Credit_Card.csv (если уже скачан).
2. Kaggle через kagglehub (нужен доступ в интернет / kaggle.json).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLUMNS, KAGGLE_DATASET, RAW_CSV_NAME, RAW_DATA_DIR, TARGET


def download_from_kaggle() -> Path:
    """Скачивает датасет через kagglehub и копирует CSV в data/raw."""
    import kagglehub

    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"[data] kagglehub cache: {path}")

    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"CSV не найден в {path}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DATA_DIR / RAW_CSV_NAME
    shutil.copy(csv_files[0], target)
    print(f"[data] скопирован в {target}")
    return target


def get_raw_path() -> Path:
    local = RAW_DATA_DIR / RAW_CSV_NAME
    if local.exists():
        print(f"[data] использую локальный файл {local}")
        return local
    return download_from_kaggle()


def load_dataset() -> pd.DataFrame:
    """Возвращает DataFrame с проверенной схемой."""
    df = pd.read_csv(get_raw_path())

    # В части выгрузок таргет называется 'default payment next month'
    if TARGET not in df.columns:
        alt = [c for c in df.columns if "default" in c.lower()]
        if not alt:
            raise KeyError(f"Не найдена колонка таргета среди {list(df.columns)}")
        df = df.rename(columns={alt[0]: TARGET})

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"В датасете отсутствуют признаки: {missing}")

    print(f"[data] загружено {len(df)} строк, доля дефолта: {df[TARGET].mean():.3f}")
    return df


def split_xy(df: pd.DataFrame):
    return df[FEATURE_COLUMNS].copy(), df[TARGET].astype(int).copy()
