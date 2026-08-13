.DEFAULT_GOAL := help
.PHONY: help setup install lint format test test-fast ingest validate features train evaluate register all serve app clean docker-build docker-up docker-down

PYTHON ?= python
VENV   ?= .venv
CONFIG ?= configs/config.yaml
FRAUD  := $(PYTHON) -m fraud_pipeline --config $(CONFIG)

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtual environment and install everything
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements-dev.txt
	$(VENV)/bin/python -m pip install -e .

install:  ## Install dependencies into the current environment
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

lint:  ## Run ruff checks
	ruff check src tests app
	ruff format --check src tests app

format:  ## Reformat the code with ruff
	ruff format src tests app
	ruff check --fix src tests app

test:  ## Run the full test suite
	pytest

test-fast:  ## Run only the quick tests, skipping training and dataset tests
	pytest -m "not slow and not needs_data"

# ----- pipeline stages, one target each -----

ingest:  ## Stage 1. Read the raw file into a typed interim table
	$(FRAUD) ingest

validate:  ## Stage 2. Enforce the schema and data quality rules
	$(FRAUD) validate

features:  ## Stage 3. Build the model ready feature sets
	$(FRAUD) features

train:  ## Stage 4. Train every model under every imbalance strategy
	$(FRAUD) train

evaluate:  ## Stage 5. Score the models and write the results table
	$(FRAUD) evaluate

register:  ## Stage 6. Promote the best model in the MLflow registry
	$(FRAUD) register

all:  ## Run stages 1 to 6 end to end
	$(FRAUD) run-all

serve:  ## Stage 7. Start the FastAPI scoring service
	$(FRAUD) serve

app:  ## Stage 7. Start the Streamlit dashboard
	streamlit run app/streamlit_app.py

# ----- packaging -----

docker-build:  ## Stage 8. Build the container images
	docker compose build

docker-up:  ## Start the API, the dashboard and MLflow in Docker
	docker compose up

docker-down:  ## Stop the containers
	docker compose down

clean:  ## Remove caches and generated artifacts, but never the raw data
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov coverage.xml
	rm -rf data/interim/* data/processed/* models/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
