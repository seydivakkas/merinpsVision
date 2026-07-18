"""Generated benchmark evidence page."""

import json

import streamlit as st

from weavevision.settings import load_settings

st.title("Benchmark")
settings = load_settings()
metric_files = sorted(settings.resolved_artifacts_root().glob("experiments/*/metrics.json"))
if not metric_files:
    st.info("NOT_RUN — generated metrics.json bulunamadı.")
else:
    selected = st.selectbox(
        "Metrics artifact", metric_files, format_func=lambda path: str(path.parent.name)
    )
    st.json(json.loads(selected.read_text(encoding="utf-8")))
