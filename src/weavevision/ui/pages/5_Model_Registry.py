"""Model registry provenance page."""

import streamlit as st

from weavevision.models.registry import ModelRegistry
from weavevision.settings import load_settings

st.title("Model Registry")
settings = load_settings()
registry = ModelRegistry(settings.resolved_artifacts_root() / "models")
st.json([item.model_dump(mode="json") for item in registry.list()])
st.caption(
    "Promotion requires hash integrity, locked threshold, metrics, latency, and report gates."
)
