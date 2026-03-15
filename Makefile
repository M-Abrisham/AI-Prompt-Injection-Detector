.PHONY: install test lint bench bench-fast build clean publish help test-fast format evaluate-buffs

PYTHON ?= python3
PYTEST_ARGS ?= -v

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## Run tests with coverage
	$(PYTHON) -m coverage run -m pytest tests/ $(PYTEST_ARGS)
	$(PYTHON) -m coverage report --fail-under=50

lint:  ## Run flake8 linter
	$(PYTHON) -m flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	$(PYTHON) -m flake8 src/ tests/ --count --exit-zero --max-complexity=15 --max-line-length=120 --statistics

bench:  ## Run full benchmark suite
	$(PYTHON) scripts/benchmark.py --dataset all

bench-fast:  ## Run single-dataset quick benchmark
	$(PYTHON) scripts/benchmark.py --dataset data/benchmark/deepset_pi.jsonl --max-samples 500

build:  ## Build wheel and sdist
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .coverage htmlcov/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

publish:  ## Upload to PyPI (requires TWINE_USERNAME and TWINE_PASSWORD)
	$(PYTHON) -m twine upload dist/*

test-fast:  ## Run tests fast (fail-fast, no coverage)
	$(PYTHON) -m pytest tests/ -x --no-header -q

format:  ## Auto-format code with black
	$(PYTHON) -m black src/ tests/ scripts/

evaluate-buffs:  ## Evaluate probe buffs across all probes
	$(PYTHON) scripts/evaluate_probes.py --buffs

.DEFAULT_GOAL := help
