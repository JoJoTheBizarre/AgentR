.PHONY: help install install-dev install-cli test lint format clean run

help:
	@echo "AgentR Commands:"
	@echo "  make install      - Install package"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make install-cli  - Install with CLI dependencies"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Lint code"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build files"
	@echo "  make run          - Run CLI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-cli:
	pip install -e ".[cli]"

install-all:
	pip install -e ".[dev,cli]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=agentr --cov-report=html --cov-report=term

lint:
	ruff check agentr tests

format:
	ruff format agentr tests

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

run:
	python cli_example.py

.DEFAULT_GOAL := help
