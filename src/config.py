"Единая конфигурация проекта: пути, имена признаков, параметры обучения."
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

KAGGLE_DATASET = "uciml/default-of-credit-card-clients-dataset"
RAW_CSV_NAME = "UCI_Credit_Card.csv"
TARGET = "default.payment.next.month"
ID_COL = "ID"

# 23 признака в порядке, ожидаемом моделью и API
FEATURE_COLUMNS = [
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Модели: v1 — контрольная (baseline), v2 — тестовая (для A/B)
MODEL_V1_PATH = MODELS_DIR / "model_v1.pkl"
MODEL_V2_PATH = MODELS_DIR / "model_v2.pkl"
METRICS_PATH = REPORTS_DIR / "metrics.json"
