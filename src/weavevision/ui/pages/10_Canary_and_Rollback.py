"""Canary & Rollback — Champion vs challenger evaluation and rollback audit.

Calls only services/ layer. No direct evaluation/ or models/ imports.
"""

from __future__ import annotations

import streamlit as st

from weavevision.domain.enums import RollbackReason
from weavevision.persistence.database import Database
from weavevision.services.canary_service import CanaryService
from weavevision.services.model_registry_service import ModelRegistryService
from weavevision.settings import load_settings

st.set_page_config(page_title="WeaveVision — Canary & Rollback", layout="wide")
st.title("🐦 Canary & Rollback Paneli")
st.caption(
    "Champion vs challenger karşılaştırma sonuçlarını ve model rollback audit trail'ini görüntüler."
)

settings = load_settings()
db = Database(settings.resolved_database_path())
db.migrate()
canary_svc = CanaryService(settings, db)
registry_svc = ModelRegistryService(settings, db)

tab_canary, tab_rollback, tab_manual_canary = st.tabs(
    ["📊 Canary Sonuçları", "🔄 Rollback Geçmişi", "🧪 Manuel Canary Değerlendirme"]
)

# =============================================================================
# Tab 1: Canary Results
# =============================================================================
with tab_canary:
    st.subheader("Tüm Canary Değerlendirmeleri")
    canaries = canary_svc.list_canaries()

    if not canaries:
        st.info("Henüz canary değerlendirmesi yok.")
    else:
        # --- Summary metrics -------------------------------------------------
        passed = sum(1 for c in canaries if c.get("status") == "PASSED")
        failed = sum(1 for c in canaries if c.get("status") == "FAILED")
        col1, col2, col3 = st.columns(3)
        col1.metric("Toplam", len(canaries))
        col2.metric("✅ PASSED", passed)
        col3.metric("❌ FAILED", failed)

        # --- Table -----------------------------------------------------------
        display_cols = [
            "canary_id",
            "champion_model_id",
            "challenger_model_id",
            "sample_count",
            "disagreement_rate",
            "critical_recall_delta",
            "latency_p95_ms",
            "status",
            "created_at",
        ]
        rows = [{c: c_row.get(c) for c in display_cols} for c_row in canaries]
        st.dataframe(rows, use_container_width=True)

        with st.expander("Politika Eşikleri"):
            st.info(
                f"**Max disagreement rate:** {settings.drift.canary_max_disagreement_rate}  \n"
                f"**Min recall delta:** {settings.drift.canary_min_recall_delta}  \n"
                "Kaynak: `configs/app.yaml → drift`"
            )

# =============================================================================
# Tab 2: Rollback History
# =============================================================================
with tab_rollback:
    st.subheader("Rollback Audit Trail")
    rollbacks = registry_svc.list_rollbacks()

    if not rollbacks:
        st.info("Rollback kaydı yok.")
    else:
        display_cols = [
            "rollback_id",
            "from_model_id",
            "to_model_id",
            "reason",
            "triggered_by",
            "incident_id",
            "created_at",
        ]
        rows = [{c: r.get(c) for c in display_cols} for r in rollbacks]
        st.dataframe(rows, use_container_width=True)

    # --- Rollback form -------------------------------------------------------
    st.divider()
    st.subheader("Model Rollback Başlat")
    st.warning(
        "⚠️ Bu işlem `from_model_id`'yi RETIRED yapar ve `to_model_id`'yi ACTIVE_BENCHMARK'a çeker."
    )

    with st.form("rollback_form", clear_on_submit=True):
        from_model = st.text_input("Mevcut model ID (devre dışı bırakılacak)", key="rb_from")
        to_model = st.text_input("Hedef model ID (geri dönülecek)", key="rb_to")
        reason_options = [r.value for r in RollbackReason]
        reason_val = st.selectbox("Rollback nedeni", reason_options, key="rb_reason")
        triggered_by = st.text_input(
            "Kim başlattı?", placeholder="Örn: ops_engineer_01", key="rb_triggered"
        )
        incident_id_val = st.text_input("Bağlı incident ID (opsiyonel)", key="rb_incident")
        confirm = st.checkbox("Bu işlemi onaylıyorum", key="rb_confirm")
        submitted = st.form_submit_button("🔄 Rollback Başlat", disabled=not confirm)

    if submitted:
        if not from_model or not to_model or not triggered_by:
            st.error("from_model, to_model ve triggered_by zorunlu.")
        else:
            try:
                event = registry_svc.rollback(
                    from_model,
                    to_model,
                    RollbackReason(reason_val),
                    triggered_by,
                    incident_id=incident_id_val or None,
                )
                st.success(f"Rollback tamamlandı: {event.rollback_id}")
                st.rerun()
            except Exception as exc:
                st.error(f"Hata: {exc}")

# =============================================================================
# Tab 3: Manual Canary Evaluation
# =============================================================================
with tab_manual_canary:
    st.subheader("Canary Değerlendirme Kaydet")
    st.info("Harici bir paralel test koşusunun sonuçlarını kaydedin.")

    with st.form("canary_form", clear_on_submit=True):
        champion_id = st.text_input("Champion model ID", key="cv_champion")
        challenger_id = st.text_input("Challenger model ID", key="cv_challenger")
        sample_count = st.number_input("Örnek sayısı", min_value=1, value=100, key="cv_samples")
        disagreement_rate = st.number_input(
            "Disagreement rate [0-1]",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.01,
            key="cv_dis",
        )
        recall_delta = st.number_input(
            "Critical recall delta (negatif = challenger daha kötü)",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key="cv_recall",
        )
        latency = st.number_input(
            "Latency p95 (ms)", min_value=0.0, value=50.0, step=1.0, key="cv_latency"
        )
        submitted_c = st.form_submit_button("📊 Değerlendirmeyi Kaydet")

    if submitted_c:
        if not champion_id or not challenger_id:
            st.error("Champion ve challenger ID zorunlu.")
        else:
            try:
                result = canary_svc.evaluate(
                    champion_id,
                    challenger_id,
                    sample_count=int(sample_count),
                    disagreement_rate=float(disagreement_rate),
                    critical_recall_delta=float(recall_delta),
                    latency_p95_ms=float(latency),
                )
                status_icon = "✅" if result.status.value == "PASSED" else "❌"
                st.success(f"{status_icon} Canary {result.canary_id}: **{result.status.value}**")
                st.json(result.model_dump(mode="json"))
            except Exception as exc:
                st.error(f"Hata: {exc}")
