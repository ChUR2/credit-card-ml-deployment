#!/usr/bin/env python
"""Точка входа для обучения моделей.

Запуск из корня проекта:
    python models/train_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import main  # noqa: E402

if __name__ == "__main__":
    main()
