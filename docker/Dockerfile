

# сборка зависимостей
FROM python:3.11-slim AS builder

WORKDIR /build

# Ставим зависимости в отдельный префикс, чтобы скопировать их одним слоем
# Только runtime-зависимости
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-prod.txt


# runtime 
FROM python:3.11-slim

# PYTHONDONTWRITEBYTECODE — не плодим .pyc в контейнере
# PYTHONUNBUFFERED — логи сразу уходят в stdout, а не копятся в буфере
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=5000 \
    LOG_LEVEL=INFO \
    AB_TRAFFIC_SPLIT=0.5

WORKDIR /app

COPY --from=builder /install /usr/local

# Код сервиса, обученные модели и пример запроса
COPY app/ ./app/
COPY models/*.pkl ./models/
COPY examples/ ./examples/

# Запуск от непривилегированного пользователя: если сервис скомпрометируют,
# у процесса не будет root-прав внутри контейнера
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health', timeout=4).status == 200 else 1)"

# gunicorn вместо flask run: dev-сервер Flask однопоточный и не рассчитан
# на production. 2 воркера x 4 потока
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app.api:app"]
