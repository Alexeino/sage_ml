PYTHON_BIN := .venv/bin

.PHONY: format lint typecheck test check

## Auto-fix lint issues and reformat code, then type-check
format:
	$(PYTHON_BIN)/ruff format .
	$(PYTHON_BIN)/ruff check --fix .
	$(PYTHON_BIN)/mypy .

## Report lint and formatting issues without changing files
lint:
	$(PYTHON_BIN)/ruff check .
	$(PYTHON_BIN)/ruff format --check .

typecheck:
	$(PYTHON_BIN)/mypy .

test:
	$(PYTHON_BIN)/pytest

## Everything CI should run: lint, type-check and tests, no file changes
check: lint typecheck test
