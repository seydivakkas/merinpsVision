"""WeaveVision Streamlit home page and navigation entry point."""

from __future__ import annotations

import streamlit as st

from weavevision.models.registry import ModelRegistry
from weavevision.services.health_service import HealthService
from weavevision.settings import load_settings
from weavevision.ui.components import model_not_ready_notice

st.set_page_config(page_title="WeaveVision", page_icon="🔎", layout="wide")
settings = load_settings()
registry = ModelRegistry(settings.resolved_artifacts_root() / "models")
active = registry.active()
st.title("WeaveVision")
st.subheader("Carpet Anomaly Detection and Quality Analytics")
st.info(
    "Normal referanslardan öğrenen yerel görsel anomaly analiz hattı. Sonuçlar kalite "
    "uzmanının incelemesini destekler; fiziksel kalite sınıfı veya garanti değildir."
)
if active is None:
    model_not_ready_notice()
else:
    st.success(f"Aktif benchmark modeli: {active.model_id} · threshold: {active.threshold_id}")
health = HealthService(settings).collect()
first, second, third = st.columns(3)
first.metric("System doctor", health["status"])
second.metric("Runtime device", health["runtime"]["active_device"])
third.metric("Registry manifests", str(registry.health()["manifests"]))
st.markdown(
    "**Sınırlar:** Açık kaynak benchmark şirket performansı değildir. Şirket modeli için tek "
    "desen/kamera/aydınlatma pilotu ve uzman threshold onayı gerekir."
)
