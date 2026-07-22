"""Domain Shift — Drift monitoring windows dashboard.

Calls only services/ layer. No direct evaluation/ or models/ imports.
"""

from __future__ import annotations

import streamlit as st

from weavevision.persistence.database import Database
from weavevision.persistence.repositories import DriftWindowRepository
from weavevision.settings import load_settings

st.set_page_config(page_title="WeaveVision — Domain Shift", layout="wide")
st.title("📊 Domain Shift İzleme")
st.caption(
    "Son drift pencerelerini görüntüler. Eşikler `configs/app.yaml → drift` bölümünden okunur."
)

settings = load_settings()
db = Database(settings.resolved_database_path())
db.migrate()
repo = DriftWindowRepository(db)

# --- Filters ----------------------------------------------------------------
col_model, col_limit = st.columns([2, 1])
with col_model:
    model_filter = st.text_input("Model ID filtresi (boş = hepsi)", key="ds_model_filter")
with col_limit:
    limit = st.number_input(
        "Gösterilecek satır sayısı", min_value=10, max_value=500, value=50, step=10
    )

rows = repo.list_recent(limit=int(limit))
if model_filter:
    rows = [r for r in rows if model_filter.lower() in str(r.get("model_id", "")).lower()]

# --- Status summary ---------------------------------------------------------
st.subheader("Özet")
total = len(rows)
if total == 0:
    st.info("Henüz drift penceresi kaydı yok.")
else:
    status_counts: dict[str, int] = {}
    for r in rows:
        s = str(r.get("trend_status", "UNKNOWN"))
        status_counts[s] = status_counts.get(s, 0) + 1

    cols = st.columns(len(status_counts) or 1)
    for i, (status, count) in enumerate(sorted(status_counts.items())):
        color = "🔴" if "BOTH" in status else "🟠" if "ALERT" in status else "🟢"
        cols[i % len(cols)].metric(f"{color} {status}", count)

    # --- Table ---------------------------------------------------------------
    st.subheader(f"Son {total} Pencere")
    display_cols = [
        "window_id",
        "model_id",
        "metric_name",
        "metric_value",
        "ewma_value",
        "cusum_value",
        "psi_value",
        "trend_status",
        "drift_pattern",
        "created_at",
    ]
    table_rows = []
    for r in rows:
        table_rows.append({c: r.get(c) for c in display_cols})

    st.dataframe(table_rows, use_container_width=True)

    # --- Detail expand -------------------------------------------------------
    with st.expander("Ham JSON (son 10)"):
        st.json(rows[:10])
