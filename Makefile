# AgentR Makefile
# Testing and development commands

.PHONY: help test test-verbose test-coverage test-html test-module test-specific \
	test-last-failed test-failed-first test-no-cov test-debug install-coverage \
	clean-coverage

# Default target
help:
	@echo "AgentR Testing Commands"
	@echo "======================="
	@echo ""
	@echo "General Testing:"
	@echo "  make test              Run all tests"
	@echo "  make test-verbose      Run all tests with verbose output"
	@echo "  make test-no-cov       Run tests without coverage (faster)"
	@echo "  make test-debug        Run tests with output and debug info"
	@echo ""
	@echo "Coverage Reports:"
	@echo "  make test-coverage     Run tests with terminal coverage report"
	@echo "  make test-html         Run tests and generate HTML coverage report"
	@echo "  make install-coverage  Install coverage dependencies"
	@echo "  make clean-coverage    Clean coverage reports and cache"
	@echo ""
	@echo "Targeted Testing:"
	@echo "  make test-module MODULE=tests/unit/graph   Run tests in specific module"
	@echo "  make test-specific TEST=tests/unit/graph/test_utils.py::TestFormatResearchSynthesis"
	@echo "                         Run specific test or test class"
	@echo "  make test-last-failed  Run only tests that failed last run"
	@echo "  make test-failed-first Run failed tests first, then others"
	@echo ""
	@echo "Examples:"
	@echo "  make test-module MODULE=tests/unit/config"
	@echo "  make test-specific TEST=tests/unit/graph/test_utils.py::TestFormatResearchSynthesis::test_format_research_synthesis_empty"

# Install coverage dependencies
install-coverage:
	poetry add --group dev pytest-cov

# Run all tests with coverage
test:
	poetry run pytest tests/ --cov=src --cov-report=term

# Run all tests with verbose output
test-verbose:
	poetry run pytest tests/ --cov=src --cov-report=term -v

# Run tests without coverage (faster for development)
test-no-cov:
	poetry run pytest tests/

# Run tests with debug output
test-debug:
	poetry run pytest tests/ --cov=src --cov-report=term -v -s

# Run tests and generate HTML coverage report
test-html:
	poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo ""
	@echo "HTML coverage report generated in htmlcov/"
	@echo "Open htmlcov/index.html in your browser"

# Run tests with terminal coverage report showing missing lines
test-coverage:
	poetry run pytest tests/ --cov=src --cov-report=term-missing

# Run tests in a specific module
# Usage: make test-module MODULE=tests/unit/graph
test-module:
ifndef MODULE
	$(error MODULE is not set. Usage: make test-module MODULE=path/to/tests)
endif
	poetry run pytest $(MODULE) --cov=src --cov-report=term

# Run a specific test or test class
# Usage: make test-specific TEST=tests/unit/graph/test_utils.py::TestFormatResearchSynthesis
test-specific:
ifndef TEST
	$(error TEST is not set. Usage: make test-specific TEST=test_path)
endif
	poetry run pytest $(TEST) -v

# Run only tests that failed last run
test-last-failed:
	poetry run pytest tests/ --lf

# Run failed tests first, then others
test-failed-first:
	poetry run pytest tests/ --ff

# Clean coverage reports and cache
clean-coverage:
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage.*
	@echo "Coverage reports cleaned"