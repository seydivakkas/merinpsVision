#!/usr/bin/env bash
set -euo pipefail
uv venv --python 3.11
uv sync --extra dev
uv run pre-commit install
uv run weavevision doctor
