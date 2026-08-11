# Сервис прогнозирования дефолта по кредитным картам

Production-like ML-сервис: обученная модель упакована в REST API, контейнеризована и подготовлена к A/B-тестированию двух версий модели.

**Образ:** [`churm2k/credit-default-api`](https://hub.docker.com/r/churm2k/credit-default-api) · **Репозиторий:** [ChUR2/credit-card-ml-deployment](https://github.com/ChUR2/credit-card-ml-deployment)

| | |
|---|---|
| Домен | Финансы, кредитный скоринг |
| Датасет | [Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) (UCI), 30 000 клиентов, 23 признака |
| Задача | Бинарная классификация: наступит ли дефолт в следующем месяце |
| Модели | v1 — LogisticRegression, v2 — RandomForest |
| Стек | Python 3.11, scikit-learn, Flask, gunicorn, Docker, NGINX |

---

## Что делает сервис

Принимает данные клиента (лимит, демография, история платежей за 6 месяцев), возвращает вероятность дефолта и бинарное решение. Одновременно держит в памяти две версии модели и распределяет трафик между ними для A/B-теста — детерминированно, по хешу `client_id`.

### Качество моделей

Отложенная выборка 6 000 клиентов, порог 0.5:

| Модель | F1 (дефолт) | Precision | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| v1 LogisticRegression | 0.465 | 0.368 | 0.632 | 0.711 | 0.491 |
| v2 RandomForest | **0.546** | **0.501** | 0.599 | **0.773** | **0.545** |

Обе модели обучены с `class_weight='balanced'` — классы несбалансированы (22.1% дефолтов).

Подробный разбор с бизнес-метрикой и статистикой — в [ab_test_plan.md](ab_test_plan.md). Кратко: v2 лучше технически, но по ожидаемым потерям банка выигрыша нет, поэтому на полный трафик она не раскатывается.

---

## Быстрый старт

### Вариант 1: готовый образ из Docker Hub

```bash
docker run -d --name credit-api -p 5001:5000 churm2k/credit-default-api:1.0
curl http://localhost:5001/health
```

### Вариант 2: сборка из исходников

```bash
git clone https://github.com/ChUR2/credit-card-ml-deployment.git
cd credit-card-ml-deployment
docker build -f docker/Dockerfile -t credit-default-api:1.0 .
docker run -d --name credit-api -p 5001:5000 credit-default-api:1.0
```

### Вариант 3: сервис + NGINX через Docker Compose

```bash
docker compose up --build -d
docker compose ps          # ждём статус healthy у credit-api
curl http://localhost/health
```

### Вариант 4: локальный запуск без Docker

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Обучение моделей (датасет скачается через kagglehub автоматически;
# либо положите UCI_Credit_Card.csv в data/raw/ вручную)
python models/train_model.py

python app/api.py          # dev-сервер на порту 5000
pytest -v                  # 13 тестов
```

> **Порт 5001, а не 5000.** На macOS порт 5000 занят системным процессом AirPlay Receiver. Внутри контейнера сервис по-прежнему слушает 5000, наружу пробрасывается 5001. На Linux/Windows можно использовать `-p 5000:5000`.

---

## API

Базовый адрес: `http://localhost:5001` (напрямую) или `http://localhost` (через NGINX в compose).

### `GET /health`

Проверка живости и готовности. Возвращает 503, если модели не загрузились — по этому эндпоинту работает `HEALTHCHECK` в Dockerfile и оркестратор.

```bash
curl http://localhost:5001/health
```

```json
{
  "status": "healthy",
  "models_loaded": ["v1", "v2"],
  "service": "credit-default-prediction"
}
```

### `POST /predict`

Прогноз по одному клиенту.

**Обязательные поля — все 23 признака:**

| Поле | Тип | Описание |
|---|---|---|
| `LIMIT_BAL` | number | Кредитный лимит, NT$ |
| `SEX` | int | 1 — мужчина, 2 — женщина |
| `EDUCATION` | int | 1 — аспирантура, 2 — университет, 3 — школа, 4 — другое |
| `MARRIAGE` | int | 1 — женат/замужем, 2 — холост, 3 — другое |
| `AGE` | int | Возраст, лет |
| `PAY_0`, `PAY_2` … `PAY_6` | int | Статус платежа за последние 6 месяцев: −1 — оплачено вовремя, 1…9 — просрочка в месяцах |
| `BILL_AMT1` … `BILL_AMT6` | number | Сумма счёта за 6 месяцев, NT$ |
| `PAY_AMT1` … `PAY_AMT6` | number | Сумма платежа за 6 месяцев, NT$ |

**Необязательные поля:**

| Поле | Описание |
|---|---|
| `client_id` | Идентификатор клиента. Определяет группу A/B детерминированно: один клиент всегда получает одну версию модели |
| `model_version` | `"v1"` или `"v2"` — явный выбор версии в обход A/B-роутинга. Можно передать и как query-параметр `?version=v2` |

**Запрос:**

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d @examples/request_example.json
```

```json
{
  "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
  "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
  "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
  "BILL_AMT4": 0, "BILL_AMT5": 0, "BILL_AMT6": 0,
  "PAY_AMT1": 0, "PAY_AMT2": 689, "PAY_AMT3": 0,
  "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0
}
```

**Ответ 200:**

```json
{
  "prediction": 1,
  "probability": 0.782112,
  "threshold": 0.5,
  "risk_level": "high",
  "model_version": "v1",
  "ab_group": "control",
  "request_id": "6a200ace5ea855b73d14f2128d516899"
}
```

| Поле ответа | Описание |
|---|---|
| `prediction` | 1 — прогнозируется дефолт, 0 — нет |
| `probability` | Вероятность дефолта, 0…1 |
| `threshold` | Порог отсечения, применённый к вероятности |
| `risk_level` | `low` (<0.2), `medium` (0.2–0.5), `high` (≥0.5) |
| `model_version` | Какая версия модели обработала запрос |
| `ab_group` | `control` (v1) или `treatment` (v2) |
| `request_id` | Идентификатор запроса, дублируется в заголовке `X-Request-ID` и в логах |

**Явный выбор версии:**

```bash
curl -X POST "http://localhost:5001/predict?version=v2" \
  -H "Content-Type: application/json" \
  -d @examples/request_example.json
```

### `POST /predict/batch`

Пакетный прогноз, до 1000 клиентов за запрос.

```bash
curl -X POST http://localhost:5001/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"clients": [{"LIMIT_BAL": 20000, "SEX": 2, ...}], "model_version": "v2"}'
```

```json
{
  "count": 1,
  "results": [{"index": 0, "client_id": null, "prediction": 1, "probability": 0.77158, "model_version": "v2", "...": "..."}]
}
```

### `GET /models`

Метаданные загруженных версий: тип модели, дата обучения, офлайн-метрики, порог.

### Ошибки

Все ошибки возвращаются в JSON.

| Код | Когда | Пример тела |
|---|---|---|
| 400 | Нет обязательных признаков, нечисловые значения, неизвестная версия модели | `{"error": "validation_error", "detail": "Отсутствуют обязательные признаки: ['AGE']"}` |
| 404 | Неизвестный эндпоинт | `{"error": "not_found", "detail": "Неизвестный эндпоинт"}` |
| 500 | Внутренняя ошибка | `{"error": "internal_error", "detail": "..."}` |
| 503 | Модели не загрузились (только `/health`) | `{"status": "unhealthy", "models_loaded": []}` |

Проверка обработки ошибки:

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" -d '{"AGE": 30}'
```

---

## Демонстрация A/B-теста

Сервис поднят — проверяем распределение трафика:

```bash
python scripts/ab_traffic_demo.py --url http://localhost:5001 --n 200
```

```
[demo] распределение трафика по 200 запросам:
  v1 (control  ):   89 запросов ( 44.5%), средняя вероятность дефолта 0.7821
  v2 (treatment):  111 запросов ( 55.5%), средняя вероятность дефолта 0.7716

[demo] детерминированность разбиения (50 повторов): OK — группа не меняется
```

Статистический анализ на отложенной выборке:

```bash
python scripts/ab_test_analysis.py
```

Считает F1/Precision/Recall по группам, z-тесты для долей, бутстреп-ДИ для разницы F1, t-тест Уэлча для денежной метрики. Результат сохраняется в `reports/ab_test_results.json`.

---

## Логи

JSON-строка на событие, вывод в stdout — стандарт для контейнеров:

```bash
docker logs credit-api | tail -5
```

```json
{"ts": "2026-08-11T10:18:38.783+00:00", "level": "INFO", "event": "prediction",
 "request_id": "6d6927d9fd87b8008a5a5095dd3a47bc", "client_id": null,
 "model_version": "v2", "ab_group": "treatment", "routing": "explicit",
 "prediction": 1, "probability": 0.77158}
```

Типы событий: `startup`, `model_loaded`, `request`, `prediction`, `batch_prediction`, `validation_error`, `internal_error`. Как это собирается в production — см. [ARCHITECTURE.md](ARCHITECTURE.md#логирование-и-мониторинг).

---

## Структура репозитория

```
credit-card-ml-deployment/
├── app/                        # Веб-сервис
│   ├── api.py                  # Flask: эндпоинты, обработка ошибок
│   ├── model_handler.py        # Реестр моделей, валидация, A/B-роутинг
│   └── logging_config.py       # JSON-логирование
├── src/                        # Пайплайн обучения
│   ├── config.py               # Пути, список признаков, гиперпараметры
│   ├── data.py                 # Загрузка датасета (kagglehub / локальный CSV)
│   └── train.py                # Обучение и сохранение моделей
├── models/
│   ├── train_model.py          # Точка входа: python models/train_model.py
│   ├── model_v1.pkl            # LogisticRegression + метаданные (joblib)
│   └── model_v2.pkl            # RandomForest + метаданные (joblib)
├── scripts/
│   ├── ab_test_analysis.py     # Статистический анализ A/B
│   ├── ab_traffic_demo.py      # Демонстрация роутинга на живом сервисе
│   └── make_synthetic_data.py  # Запасной генератор данных для отладки
├── tests/test_api.py           # 13 тестов API
├── docker/
│   ├── Dockerfile              # Multi-stage, non-root, healthcheck
│   └── nginx/nginx.conf        # Reverse proxy, rate limiting
├── examples/request_example.json
├── reports/                    # metrics.json, ab_test_results.json
├── data/raw/                   # Датасет (не коммитится, качается скриптом)
├── docker-compose.yml          # api + nginx
├── requirements.txt            # Полное окружение разработки
├── requirements-prod.txt       # Только runtime — ставится в образ
├── ab_test_plan.md             # План и результаты A/B-теста
├── ARCHITECTURE.md             # Архитектурные решения и MLOps
└── Makefile                    # make train / test / docker-build / compose-up
```

---

## Модель: сохранение и загрузка

Модели сохраняются через **joblib** не как «голый» объект, а как бандл с метаданными:

```python
{
    "model": Pipeline(StandardScaler + OneHotEncoder -> Classifier),
    "feature_columns": ["LIMIT_BAL", "SEX", ..., "PAY_AMT6"],   # порядок признаков
    "model_version": "v1",
    "threshold": 0.5,
    "metrics": {...},
    "trained_at": "2026-08-11T08:20:11+00:00",
}
```

Зачем `feature_columns` внутри файла: при инференсе JSON приходит как словарь, а порядок ключей в словаре не гарантирует порядок признаков. Если восстанавливать его сортировкой ключей или полагаться на порядок в запросе, модель получит перепутанные колонки и будет молча выдавать неверные предсказания — без единой ошибки в логах. Порядок берётся из бандла, а `app/model_handler.py` проверяет наличие всех полей и приводит их к нужной последовательности.

joblib, а не чистый pickle: он эффективнее сериализует numpy-массивы, из которых на 99% состоит RandomForest.

Загрузка происходит один раз при старте процесса (`ModelRegistry.load_all()`), а не на каждый запрос — иначе каждое обращение к `/predict` стоило бы чтения 5.7 МБ с диска.

---

## Makefile

```bash
make help           # список команд
make install        # зависимости
make train          # обучить модели
make test           # pytest
make run            # dev-сервер
make docker-build   # собрать образ
make docker-run     # запустить контейнер
make docker-test    # curl-проверки
make compose-up     # api + nginx
make docker-push DOCKERHUB_USER=churm2k
```

---

## Документация

- [ARCHITECTURE.md](ARCHITECTURE.md) — монолит vs микросервисы, брокеры сообщений, логирование и мониторинг, DVC/MLflow, ONNX, uWSGI + NGINX
- [ab_test_plan.md](ab_test_plan.md) — план A/B-теста, статистические критерии, результаты, бизнес-метрики
