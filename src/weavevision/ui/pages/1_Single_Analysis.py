"""Single-image analysis page."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from weavevision.domain.errors import WeaveVisionError
from weavevision.models.registry import ModelRegistry
from weavevision.settings import load_settings
from weavevision.ui.components import analysis_summary, model_not_ready_notice
from weavevision.ui.state import analysis_service

st.title("Single Analysis")
settings = load_settings()
active = ModelRegistry(settings.resolved_artifacts_root() / "models").active()
if active is None:
    model_not_ready_notice()
uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
if uploaded and st.button("Analyze", type="primary"):
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(uploaded.getbuffer())
        temporary = Path(handle.name)
    try:
        service = analysis_service(None, None, None, None)
        result = service.analyze(temporary)
        analysis_summary(result)
    except WeaveVisionError as exc:
        st.error(f"{exc.code}: {exc.message}")
    finally:
        temporary.unlink(missing_ok=True)
