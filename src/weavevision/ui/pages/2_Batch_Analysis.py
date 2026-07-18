"""Batch analysis page with degraded-ready upload state."""

import streamlit as st

from weavevision.models.registry import ModelRegistry
from weavevision.settings import load_settings
from weavevision.ui.components import model_not_ready_notice

st.title("Batch Analysis")
settings = load_settings()
if ModelRegistry(settings.resolved_artifacts_root() / "models").active() is None:
    model_not_ready_notice()
st.file_uploader("ZIP archive", type=["zip"])
st.caption("Her dosya bağımsız işlenir; bozuk öğeler tüm batch'i düşürmez.")
