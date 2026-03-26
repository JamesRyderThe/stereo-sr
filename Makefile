.PHONY: build format lint test full train

build:
	uv sync --extra dev

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

lint:
	uv run ruff check src tests scripts
	uv run mypy --strict src/sissr

test:
	uv run pytest

full: format lint test

train:
	uv run accelerate launch --config_file $${ACCELERATE_CONFIG_FILE:-configs/accelerate/multi_gpu.yaml} -m sissr.train --config configs/diffssr_baseline.yaml
