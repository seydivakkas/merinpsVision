"""Labeling Queue — Active learning candidate review dashboard.

Calls only services/ layer. No direct evaluation/ or models/ imports.
"""

from __future__ import annotations

import streamlit as st

from weavevision.persistence.database import Database
from weavevision.services.active_learning_service import ActiveLearningService
from weavevision.settings import load_settings

st.set_page_config(page_title="WeaveVision — Labeling Queue", layout="wide")
st.title("🏷️ Aktif Öğrenme — Etiketleme Kuyruğu")
st.caption(
    "Uzman etiketleme için seçilen aday görüntüler. "
    "Seçim: greedy coreset (diversity) + drift score önceliği."
)

settings = load_settings()
db = Database(settings.resolved_database_path())
db.migrate()
svc = ActiveLearningService(settings, db)

# --- Pending items ----------------------------------------------------------
pending = svc.list_pending()

# --- Summary ----------------------------------------------------------------
bucket_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
bucket_counts: dict[str, int] = {}
for r in pending:
    b = str(r.get("priority_bucket", "P3"))
    bucket_counts[b] = bucket_counts.get(b, 0) + 1

st.subheader("Kuyruk Özeti")
if not pending:
    st.info("Kuyruk boş. Aktif öğrenme henüz başlatılmamış ya da tüm öğeler incelendi.")
else:
    bucket_icons = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🔵"}
    cols = st.columns(4)
    for i, b in enumerate(["P0", "P1", "P2", "P3"]):
        count = bucket_counts.get(b, 0)
        cols[i].metric(f"{bucket_icons[b]} {b}", count)

    # --- Table ---------------------------------------------------------------
    st.subheader(f"Bekleyen Öğeler ({len(pending)})")
    display_cols = [
        "item_id",
        "priority_bucket",
        "source_path",
        "drift_score",
        "uncertainty_score",
        "selection_reason",
        "created_at",
    ]
    table_rows = [{c: r.get(c) for c in display_cols} for r in pending]
    st.dataframe(table_rows, use_container_width=True)

    # --- Verdict form -------------------------------------------------------
    st.divider()
    st.subheader("Verdict Kaydet")
    with st.form("verdict_form", clear_on_submit=True):
        item_ids = [str(r["item_id"]) for r in pending]
        selected_item = st.selectbox("İtem ID", item_ids, key="lq_item_select")
        verdict_options = [
            "TRUE_ANOMALY",
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
            "CONFIRMED_NORMAL",
            "CONFIRMED_ANOMALY",
            "UNSURE",
        ]
        verdict = st.selectbox("Verdict", verdict_options, key="lq_verdict")
        reviewer = st.text_input("Uzman ID", key="lq_reviewer", placeholder="Örn: qa_lead_01")
        submitted = st.form_submit_button("💾 Kaydet")

    if submitted and selected_item:
        try:
            svc.record_verdict(selected_item, verdict, reviewer=reviewer or None)
            st.success(f"Verdict kaydedildi: {selected_item} → {verdict}")
            st.rerun()
        except Exception as exc:
            st.error(f"Hata: {exc}")

# --- Raw JSON ---------------------------------------------------------------
with st.expander("Ham JSON (bekleyen)"):
    st.json(pending[:20])
