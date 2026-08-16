# Короткие команды для типовых операций
# Использование: make <цель> например make docker-build

DOCKERHUB_USER ?= YOUR_DOCKERHUB_LOGIN
IMAGE          ?= credit-default-api
TAG            ?= 1.0

.PHONY: help install train test run docker-build docker-run docker-test docker-push compose-up compose-down clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## Установить зависимости разработки
	pip install -r requirements.txt

train:  ## Обучить модели v1 и v2
	python models/train_model.py

test:  ## Прогнать тесты
	pytest -v

run:  ## Запустить сервис локально (dev)
	python app/api.py

docker-build:  ## Собрать образ
	docker build -f docker/Dockerfile -t $(IMAGE):$(TAG) .

docker-run:  ## Запустить контейнер на порту 5000
	docker run -d --name credit-api -p 5000:5000 $(IMAGE):$(TAG)

docker-test:  ## Проверить запущенный контейнер
	curl -s http://localhost:5000/health | python -m json.tool
	curl -s -X POST http://localhost:5000/predict \
	  -H "Content-Type: application/json" \
	  -d @examples/request_example.json | python -m json.tool

docker-push:  ## Опубликовать образ в Docker Hub
	docker tag $(IMAGE):$(TAG) $(DOCKERHUB_USER)/$(IMAGE):$(TAG)
	docker tag $(IMAGE):$(TAG) $(DOCKERHUB_USER)/$(IMAGE):latest
	docker push $(DOCKERHUB_USER)/$(IMAGE):$(TAG)
	docker push $(DOCKERHUB_USER)/$(IMAGE):latest

compose-up:  ## Поднять сервис + NGINX
	docker compose up --build -d

compose-down:  ## Остановить всё
	docker compose down

clean:  ## Убрать контейнер и кеш Python
	-docker rm -f credit-api
	find . -type d -name __pycache__ -exec rm -rf {} +
