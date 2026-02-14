.PHONY: install test lint run clean help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install ipykernel ruff
	@echo "\n✓ Run 'source $(VENV)/bin/activate' to activate."

test: ## Run pytest with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

lint: ## Lint with ruff
	$(PYTHON) -m ruff check src/ tests/

run: ## Run the full segmentation pipeline
	$(PYTHON) main.py

dashboard: ## Launch Streamlit dashboard
	$(VENV)/bin/streamlit run streamlit_app.py

clean: ## Remove generated artifacts
	rm -rf $(VENV) __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
