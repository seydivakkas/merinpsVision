"""Analysis history and expert feedback page."""

import streamlit as st

from weavevision.persistence.database import Database
from weavevision.persistence.repositories import AnalysisRepository
from weavevision.settings import load_settings

st.title("Review and Feedback")
settings = load_settings()
database = Database(settings.resolved_database())
database.migrate()
rows = AnalysisRepository(database).list_recent()
st.dataframe(rows, use_container_width=True)
st.caption("Feedback yeni bir audit kaydıdır; orijinal analiz kanıtını değiştirmez.")
