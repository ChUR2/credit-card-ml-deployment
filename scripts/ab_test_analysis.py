#!/usr/bin/env python
"""Офлайн-анализ A/B-теста: v1 (control) против v2 (treatment).

Симуляция повторяет логику сервиса: клиент попадает в группу по md5-хешу
своего ID, группы не пересекаются. Каждая группа скорится только своей моделью —
как это происходило бы в реальном тесте.

Считает:
   технические метрики по группам (F1, Precision, Recall);
   статистическую значимость: z-тест для долей (Recall/Precision), бутстреп-доверительный интервал для разницы F1;
   бизнес-метрику - ожидаемые потери банка в деньгах, t-тест Уэлча.

Запуск из корня проекта:
    python scripts/ab_test_analysis.py
Результат: reports/ab_test_results.json + таблица в консоли.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    MODEL_V1_PATH,
    MODEL_V2_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TEST_SIZE,
)
from src.data import load_dataset, split_xy  # noqa: E402

# параметры

AB_TRAFFIC_SPLIT = 0.5      # доля трафика на v2
ALPHA = 0.05                # уровень значимости
N_BOOTSTRAP = 2000          # итераций бутстрепа для ДИ разницы F1

# Экономика решения (значения согласуются с риск-подразделением банка).
# LGD — доля невозврата при дефолте; MARGIN — доходность по обслуживаемому долгу.
LGD = 0.60
MARGIN = 0.08
REVIEW_COST = 300.0         # стоимость ручной проверки заявки, NT$


# утилиты


def assign_group(client_id, split: float = AB_TRAFFIC_SPLIT) -> str:
    "Та же функция разбиения, что и в app/model_handler.py."
    digest = hashlib.md5(str(client_id).encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "treatment" if bucket < split else "control"


def two_proportion_ztest(succ_a: int, n_a: int, succ_b: int, n_b: int) -> dict:
    "Двусторонний z-тест для разницы двух долей + ДИ (Вальд)."
    p_a, p_b = succ_a / n_a, succ_b / n_b
    p_pool = (succ_a + succ_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = stats.norm.ppf(1 - ALPHA / 2)
    return {
        "control": round(p_a, 4),
        "treatment": round(p_b, 4),
        "diff": round(p_b - p_a, 4),
        "ci_95": [round(p_b - p_a - crit * se_diff, 4), round(p_b - p_a + crit * se_diff, 4)],
        "z": round(float(z), 3),
        "p_value": round(float(p_value), 6),
        "significant": bool(p_value < ALPHA),
    }


def bootstrap_f1_diff(y_c, pred_c, y_t, pred_t, rng) -> dict:
    "Бутстреп-ДИ для разницы F1, ресемплируем группы независимо"
    diffs = np.empty(N_BOOTSTRAP)
    n_c, n_t = len(y_c), len(y_t)
    for i in range(N_BOOTSTRAP):
        idx_c = rng.integers(0, n_c, n_c)
        idx_t = rng.integers(0, n_t, n_t)
        diffs[i] = f1_score(y_t[idx_t], pred_t[idx_t], zero_division=0) - f1_score(
            y_c[idx_c], pred_c[idx_c], zero_division=0
        )
    lo, hi = np.percentile(diffs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    observed = f1_score(y_t, pred_t, zero_division=0) - f1_score(y_c, pred_c, zero_division=0)
    return {
        "diff": round(float(observed), 4),
        "ci_95": [round(float(lo), 4), round(float(hi), 4)],
        # ДИ не накрывает ноль => разница значима на уровне ALPHA
        "significant": bool(lo > 0 or hi < 0),
        "p_value_bootstrap": round(float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())), 6),
    }


def expected_cost(y_true, y_pred, exposure) -> np.ndarray:
    """Стоимость решения по каждому клиенту, NT$.

    FN — выдали кредит будущему дефолтёру: теряем exposure * LGD
    FP — отказали платёжеспособному: упускаем маржу exposure * MARGIN
    TP — дефолт предотвращён, но заявка ушла на ручную проверку: REVIEW_COST
    TN — штатная выдача, стоимость решения нулевая
    """
    cost = np.zeros(len(y_true), dtype=float)
    fn = (y_true == 1) & (y_pred == 0)
    fp = (y_true == 0) & (y_pred == 1)
    tp = (y_true == 1) & (y_pred == 1)

    cost[fn] = exposure[fn] * LGD
    cost[fp] = exposure[fp] * MARGIN
    cost[tp] = REVIEW_COST
    return cost


def required_sample_size(p_base: float, mde: float, alpha=ALPHA, power=0.8) -> int:
    "Размер выборки на группу для z-теста долей (двусторонний)."
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_new = p_base + mde
    p_bar = (p_base + p_new) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(
        p_base * (1 - p_base) + p_new * (1 - p_new)
    )) ** 2
    return int(np.ceil(num / mde**2))


# основной сценарий


def main() -> None:
    df = load_dataset()
    X, y = split_xy(df)

    # Тот же сплит, что при обучении: тестовая часть моделям незнакома
    ids = df["ID"].to_numpy()
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    y_test = y_test.to_numpy()
    exposure = X_test["BILL_AMT1"].to_numpy().clip(min=0)

    groups = np.array([assign_group(i) for i in id_test])
    is_ctrl = groups == "control"
    print(f"[ab] тестовая выборка: {len(y_test)} клиентов")
    print(f"[ab] control (v1): {is_ctrl.sum()}, treatment (v2): {(~is_ctrl).sum()}")

    model_v1 = joblib.load(MODEL_V1_PATH)["model"]
    model_v2 = joblib.load(MODEL_V2_PATH)["model"]

    # Каждая группа скорится своей моделью — как в реальном A/B
    pred = np.empty(len(y_test), dtype=int)
    pred[is_ctrl] = model_v1.predict(X_test[is_ctrl])
    pred[~is_ctrl] = model_v2.predict(X_test[~is_ctrl])

    y_c, pred_c, exp_c = y_test[is_ctrl], pred[is_ctrl], exposure[is_ctrl]
    y_t, pred_t, exp_t = y_test[~is_ctrl], pred[~is_ctrl], exposure[~is_ctrl]

    def block(y_, p_):
        return {
            "n": int(len(y_)),
            "defaults": int(y_.sum()),
            "f1": round(float(f1_score(y_, p_, zero_division=0)), 4),
            "precision": round(float(precision_score(y_, p_, zero_division=0)), 4),
            "recall": round(float(recall_score(y_, p_, zero_division=0)), 4),
            "flagged_rate": round(float(p_.mean()), 4),
        }

    results = {
        "config": {
            "traffic_split": AB_TRAFFIC_SPLIT,
            "alpha": ALPHA,
            "bootstrap_iterations": N_BOOTSTRAP,
            "economics": {"LGD": LGD, "MARGIN": MARGIN, "review_cost": REVIEW_COST},
        },
        "groups": {"control_v1": block(y_c, pred_c), "treatment_v2": block(y_t, pred_t)},
    }

    # статистика
    rng = np.random.default_rng(RANDOM_STATE)
    results["tests"] = {
        "f1_bootstrap": bootstrap_f1_diff(y_c, pred_c, y_t, pred_t, rng),
        "recall_ztest": two_proportion_ztest(
            int(((y_c == 1) & (pred_c == 1)).sum()), int((y_c == 1).sum()),
            int(((y_t == 1) & (pred_t == 1)).sum()), int((y_t == 1).sum()),
        ),
        "precision_ztest": two_proportion_ztest(
            int(((y_c == 1) & (pred_c == 1)).sum()), int((pred_c == 1).sum()),
            int(((y_t == 1) & (pred_t == 1)).sum()), int((pred_t == 1).sum()),
        ),
    }

    # бизнес-метрика
    cost_c = expected_cost(y_c, pred_c, exp_c)
    cost_t = expected_cost(y_t, pred_t, exp_t)
    t_stat, p_cost = stats.ttest_ind(cost_c, cost_t, equal_var=False)
    diff = cost_t.mean() - cost_c.mean()
    se = np.sqrt(cost_c.var(ddof=1) / len(cost_c) + cost_t.var(ddof=1) / len(cost_t))
    crit = stats.norm.ppf(1 - ALPHA / 2)

    results["business"] = {
        "avg_cost_per_client_control": round(float(cost_c.mean()), 2),
        "avg_cost_per_client_treatment": round(float(cost_t.mean()), 2),
        "diff_per_client": round(float(diff), 2),
        "diff_ci_95": [round(float(diff - crit * se), 2), round(float(diff + crit * se), 2)],
        "relative_change_pct": round(float(100 * diff / cost_c.mean()), 2),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_cost), 6),
        "significant": bool(p_cost < ALPHA),
        # Положительное значение = экономия банка при переходе на v2,
        # отрицательное = рост потерь
        "savings_per_100k_clients": round(float(-diff * 100_000), 0),
    }

    # планирование выборки
    base_recall = results["groups"]["control_v1"]["recall"]
    results["sample_size_planning"] = {
        "base_recall_v1": base_recall,
        "per_group_for_mde": {
            f"{mde:+.2f}": required_sample_size(base_recall, mde) for mde in (0.03, 0.05, 0.10)
        },
        "comment": "Размер выборки считается по доле дефолтёров, а не по всему трафику",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "ab_test_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    _print_report(results)
    print(f"\n[ab] полный отчёт: {out}")


def _print_report(r: dict) -> None:
    c, t = r["groups"]["control_v1"], r["groups"]["treatment_v2"]
    print("\n" + "=" * 68)
    print("РЕЗУЛЬТАТЫ A/B-ТЕСТА")
    print("=" * 68)
    print(f"{'Метрика':<22}{'control (v1)':>15}{'treatment (v2)':>17}{'Δ':>12}")
    print("-" * 68)
    for key, label in [("n", "Клиентов"), ("defaults", "Дефолтов"),
                       ("f1", "F1"), ("precision", "Precision"),
                       ("recall", "Recall"), ("flagged_rate", "Доля отказов")]:
        delta = t[key] - c[key]
        fmt = "d" if isinstance(c[key], int) else ".4f"
        print(f"{label:<22}{c[key]:>15{fmt}}{t[key]:>17{fmt}}{delta:>12{fmt}}")

    print("\nСТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ")
    print("-" * 68)
    f1 = r["tests"]["f1_bootstrap"]
    print(f"ΔF1 = {f1['diff']:+.4f}, 95% ДИ [{f1['ci_95'][0]:+.4f}; {f1['ci_95'][1]:+.4f}]"
          f" -> {'значимо' if f1['significant'] else 'не значимо'}")
    for name, label in [("recall_ztest", "Recall"), ("precision_ztest", "Precision")]:
        z = r["tests"][name]
        print(f"Δ{label} = {z['diff']:+.4f}, ДИ [{z['ci_95'][0]:+.4f}; {z['ci_95'][1]:+.4f}], "
              f"z = {z['z']:+.2f}, p = {z['p_value']:.2e} -> "
              f"{'значимо' if z['significant'] else 'не значимо'}")

    b = r["business"]
    print("\nБИЗНЕС-МЕТРИКА: ожидаемые потери на клиента, NT$")
    print("-" * 68)
    print(f"control (v1):   {b['avg_cost_per_client_control']:>12,.2f}")
    print(f"treatment (v2): {b['avg_cost_per_client_treatment']:>12,.2f}")
    print(f"разница:        {b['diff_per_client']:>12,.2f}  "
          f"({b['relative_change_pct']:+.2f}%), p = {b['p_value']:.2e}")
    print(f"95% ДИ разницы: [{b['diff_ci_95'][0]:,.2f}; {b['diff_ci_95'][1]:,.2f}]")
    print(f"экономия на 100 000 клиентов: {b['savings_per_100k_clients']:+,.0f} NT$ "
          f"(минус = потери растут)")

    print("\nПЛАНИРОВАНИЕ ВЫБОРКИ (клиентов-дефолтёров на группу)")
    print("-" * 68)
    for mde, n in r["sample_size_planning"]["per_group_for_mde"].items():
        print(f"  MDE {mde} по Recall: {n:>8,} на группу")


if __name__ == "__main__":
    main()
