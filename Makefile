.PHONY: install test lint format backtest report clean

install:
	uv pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=statarb --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

backtest:
	python -m statarb.cli backtest --config configs/default.yaml

report:
	python -m statarb.cli report --latest

clean:
	rm -rf build dist *.egg-info .pytest_cache htmlcov .coverage
