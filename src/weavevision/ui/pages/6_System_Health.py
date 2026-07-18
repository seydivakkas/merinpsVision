"""System doctor details page."""

import streamlit as st

from weavevision.services.health_service import HealthService
from weavevision.settings import load_settings

st.title("System Health")
st.json(HealthService(load_settings()).collect())
