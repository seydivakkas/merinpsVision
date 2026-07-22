"""Drift Incidents — Open incident triage dashboard.

Calls only services/ layer. No direct evaluation/ or models/ imports.
"""

from __future__ import annotations

import streamlit as st

from weavevision.persistence.database import Database
from weavevision.services.incident_service import IncidentService
from weavevision.settings import load_settings

st.set_page_config(page_title="WeaveVision — Drift Incidents", layout="wide")
st.title("🚨 Drift İncidentleri")
st.caption(
    "2-of-N sinyal kuralıyla açılan incidentleri listeler. "
    "Kapatmak için `resolved_at` güncellenmeli."
)

settings = load_settings()
db = Database(settings.resolved_database_path())
db.migrate()
incident_svc = IncidentService(settings, db)

# --- Priority filter --------------------------------------------------------
priority_options = ["Hepsi", "P0_BLOCKED", "P1_INCIDENT", "P2_REVIEW", "INFO"]
selected_priority = st.selectbox("Öncelik filtresi", priority_options, key="inc_priority")

# --- Open incidents ---------------------------------------------------------
open_rows = incident_svc.list_open()
if selected_priority != "Hepsi":
    open_rows = [r for r in open_rows if r.get("priority") == selected_priority]

# --- Summary metrics --------------------------------------------------------
st.subheader("Açık İncidentler")
if not open_rows:
    st.success("Açık incident yok.")
else:
    p_counts: dict[str, int] = {}
    for r in open_rows:
        p = str(r.get("priority", "UNKNOWN"))
        p_counts[p] = p_counts.get(p, 0) + 1

    cols = st.columns(max(len(p_counts), 1))
    priority_icons = {"P0_BLOCKED": "🔴", "P1_INCIDENT": "🟠", "P2_REVIEW": "🟡", "INFO": "🔵"}
    for i, (p, count) in enumerate(sorted(p_counts.items())):
        icon = priority_icons.get(p, "⚪")
        cols[i % len(cols)].metric(f"{icon} {p}", count)

    # --- Incident table -----------------------------------------------------
    display_cols = [
        "incident_id",
        "priority",
        "drift_pattern",
        "model_id",
        "affected_window_id",
        "root_cause",
        "action_taken",
        "created_at",
    ]
    table_rows = [{c: r.get(c) for c in display_cols} for r in open_rows]
    st.dataframe(table_rows, use_container_width=True)

    # --- Resolve form -------------------------------------------------------
    st.divider()
    st.subheader("İncident Kapat")
    with st.form("resolve_form", clear_on_submit=True):
        incident_ids = [str(r["incident_id"]) for r in open_rows]
        selected_id = st.selectbox("Kapatılacak incident", incident_ids)
        action_taken = st.text_area("Alınan aksiyon", placeholder="Örn: model rollback uygulandı")
        submitted = st.form_submit_button("✅ Kapat")

    if submitted and selected_id:
        try:
            incident_svc.resolve(selected_id, action_taken=action_taken or None)
            st.success(f"Incident {selected_id} kapatıldı.")
            st.rerun()
        except Exception as exc:
            st.error(f"Hata: {exc}")

# --- Raw JSON ---------------------------------------------------------------
with st.expander("Ham JSON"):
    st.json(open_rows)
