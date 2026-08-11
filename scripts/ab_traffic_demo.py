#!/usr/bin/env python
"""Живая демонстрация A/B-роутинга на работающем сервисе.

Отправляет N запросов с разными client_id и показывает, как трафик
распределился между v1 и v2, а также проверяет детерминированность:
повторный запрос с тем же client_id обязан попасть в ту же группу.

Запуск (сервис должен быть поднят):
    python scripts/ab_traffic_demo.py --url http://localhost:5001 --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests

EXAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "examples" / "request_example.json").read_text()
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:5001")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    try:
        health = requests.get(f"{args.url}/health", timeout=5).json()
    except requests.RequestException as exc:
        sys.exit(f"Сервис недоступен по адресу {args.url}: {exc}")
    print(f"[demo] health: {health}")

    counter = Counter()
    probs = defaultdict(list)
    assignments = {}

    for i in range(args.n):
        client_id = f"client-{i:04d}"
        payload = dict(EXAMPLE, client_id=client_id)
        r = requests.post(f"{args.url}/predict", json=payload, timeout=10).json()
        counter[r["model_version"]] += 1
        probs[r["model_version"]].append(r["probability"])
        assignments[client_id] = r["model_version"]

    print(f"\n[demo] распределение трафика по {args.n} запросам:")
    for version, count in sorted(counter.items()):
        share = 100 * count / args.n
        mean_p = sum(probs[version]) / len(probs[version])
        group = "control" if version == "v1" else "treatment"
        print(f"  {version} ({group:<9}): {count:>4} запросов ({share:5.1f}%), "
              f"средняя вероятность дефолта {mean_p:.4f}")

    # Проверка детерминированности разбиения
    stable = True
    for client_id, version in list(assignments.items())[:50]:
        payload = dict(EXAMPLE, client_id=client_id)
        again = requests.post(f"{args.url}/predict", json=payload, timeout=10).json()
        if again["model_version"] != version:
            stable = False
            print(f"  [!] {client_id}: {version} -> {again['model_version']}")

    print(f"\n[demo] детерминированность разбиения (50 повторов): "
          f"{'OK — группа не меняется' if stable else 'НАРУШЕНА'}")


if __name__ == "__main__":
    main()
