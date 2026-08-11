#!/usr/bin/env python
"""ЗАПАСНОЙ вариант: генерация синтетического датасета той же схемы.

Нужен только для отладки пайплайна в окружении без доступа к Kaggle.
Для сдачи проекта используйте реальные данные:
    python models/train_model.py   # сам скачает через kagglehub
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURE_COLUMNS, ID_COL, RAW_DATA_DIR, RAW_CSV_NAME, TARGET  # noqa: E402

N = 30000
rng = np.random.default_rng(42)

df = pd.DataFrame({ID_COL: np.arange(1, N + 1)})
df["LIMIT_BAL"] = rng.lognormal(11.6, 0.7, N).round(-3).clip(10000, 1_000_000)
df["SEX"] = rng.choice([1, 2], N, p=[0.4, 0.6])
df["EDUCATION"] = rng.choice([0, 1, 2, 3, 4, 5, 6], N, p=[0.005, 0.35, 0.47, 0.16, 0.005, 0.005, 0.005])
df["MARRIAGE"] = rng.choice([0, 1, 2, 3], N, p=[0.002, 0.455, 0.53, 0.013])
df["AGE"] = rng.integers(21, 70, N)

pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
base = rng.choice([-2, -1, 0, 1, 2, 3], N, p=[0.13, 0.19, 0.5, 0.13, 0.04, 0.01])
for i, c in enumerate(pay_cols):
    df[c] = np.clip(base + rng.integers(-1, 2, N), -2, 8)

for i in range(1, 7):
    df[f"BILL_AMT{i}"] = (df["LIMIT_BAL"] * rng.beta(2, 4, N)).round(0)
    df[f"PAY_AMT{i}"] = (df[f"BILL_AMT{i}"] * rng.beta(1.5, 6, N)).round(0)

# Таргет зависит от просрочек, утилизации лимита и возраста + шум
logit = (
    -2.4
    + 0.75 * df["PAY_0"]
    + 0.25 * df["PAY_2"]
    + 0.10 * df["PAY_3"]
    + 1.2 * (df["BILL_AMT1"] / df["LIMIT_BAL"])
    - 0.012 * (df["AGE"] - 35)
    + rng.normal(0, 0.6, N)
)
p = 1 / (1 + np.exp(-logit))
df[TARGET] = (rng.random(N) < p).astype(int)

df = df[[ID_COL] + FEATURE_COLUMNS + [TARGET]]
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
out = RAW_DATA_DIR / RAW_CSV_NAME
df.to_csv(out, index=False)
print(f"synthetic -> {out}, rows={len(df)}, default_rate={df[TARGET].mean():.3f}")
