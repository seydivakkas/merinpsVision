"""Reusable, presentation-only Streamlit components."""

from __future__ import annotations

import streamlit as st

from weavevision.domain.schemas import AnalysisResult


def model_not_ready_notice() -> None:
    """Render the degraded-ready model setup state."""
    st.warning(
        "Model hazır değil. Arayüz çalışır durumda; analiz için doğrulanmış veri, model "
        "artifact'i ve validation ile kilitlenmiş threshold gerekir."
    )


def analysis_summary(result: AnalysisResult) -> None:
    """Render decision, provenance, timing, and available visuals."""
    st.subheader(result.prediction.decision.value)
    columns = st.columns(4)
    columns[0].metric("Anomaly score", f"{result.prediction.raw_anomaly_score:.5f}")
    columns[1].metric("Review priority", result.prediction.review_priority.value)
    columns[2].metric("Area ratio", f"{result.prediction.anomaly_area_ratio:.2%}")
    columns[3].metric("Latency", f"{result.timing_ms.total:.1f} ms")
    st.caption("Anomaly score olasılık veya confidence değildir.")
    st.json(result.model_dump(mode="json"))
