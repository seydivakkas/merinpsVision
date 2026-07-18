.PHONY: doctor test lint typecheck serve
doctor:
	uv run weavevision doctor
test:
	uv run pytest -q
lint:
	uv run ruff check .
	uv run ruff format --check .
typecheck:
	uv run mypy src
serve:
	uv run weavevision serve
