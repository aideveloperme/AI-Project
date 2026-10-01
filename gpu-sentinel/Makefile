VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: install backend simulator frontend dev test lint build up down

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r backend/requirements-dev.txt
	cd frontend && npm install

backend:           ## API + in-process simulator on :8000 (SQLite)
	cd backend && ../$(VENV)/bin/uvicorn sentinel.app:app --reload --port 8000

simulator:         ## standalone simulator exporter on :9400
	cd backend && ../$(PY) -m sentinel.simulator.server

frontend:          ## dashboard dev server on :3000
	cd frontend && npm run dev

test:
	cd backend && ../$(PY) -m pytest -q

lint:
	cd backend && ../$(VENV)/bin/ruff check sentinel tests
	cd frontend && npx tsc --noEmit

build:
	cd frontend && NODE_ENV=production npx next build

up:
	docker compose up -d --build

down:
	docker compose down
