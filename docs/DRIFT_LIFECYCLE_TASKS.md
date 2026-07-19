# WeaveVision — Drift & Lifecycle Governance: Task Listesi

> Kaynak: `WEAVEVISION_DRIFT_IMPLEMENTATION.md`. Milestone'ları sırayla uygula,
> bir milestone'un DoD'si geçmeden bir sonrakine geçme. Her milestone kendi
> git commit'ini oluşturmalı: `feat(drift): M<n> - <özet>`.

---

## M0 — Hazırlık

- [ ] `feat/drift-lifecycle-governance` branch'i oluştur
- [ ] `uv sync` ile ortamı doğrula
- [ ] `uv run weavevision doctor` çalıştır, GPU/CUDA durumunu doğrula
- [ ] `nvidia-smi` ile RTX 4070 Laptop VRAM'in 8GB göründüğünü teyit et
- [ ] `WEAVEVISION_DRIFT_IMPLEMENTATION.md` dosyasını `docs/DRIFT_LIFECYCLE_IMPLEMENTATION.md` olarak repoya kopyala

**DoD:** `make doctor` hatasız çalışıyor, branch açık.

---

## M1 — Domain Katmanı

- [ ] `domain/enums.py`: `DriftPattern`, `TrendStatus`, `IncidentPriority`, `RetrainingStrategy`, `CanaryStatus`, `RollbackReason` ekle
- [ ] `domain/schemas.py`: `DriftWindow`, `TrendPoint`, `IncidentRecord`, `TriageDecision`, `RetrainingRequest`, `LabelingQueueItem`, `CanaryEvaluation`, `RollbackEvent` ekle (`ConfigDict(extra="forbid")`)
- [ ] `tests/unit/test_domain_drift_schemas.py`: her şema için 1 geçerli + 1 "extra field reddi" testi

**DoD:** `mypy src` strict pass, `ruff check .` pass, yeni testler yeşil.

---

## M2 — Trend & PSI Modülleri

- [ ] `evaluation/trend_monitor.py` → `monitor_downward_drift()`, `classify_sudden_drop()`
- [ ] `evaluation/psi.py` → `population_stability_index()`, `psi_severity()`
- [ ] `tests/unit/test_trend_monitor.py`: stabil / ani düşüş / 3+ hafta kademeli senaryoları, `baseline_std=0` hata testi
- [ ] `tests/unit/test_psi.py`: aynı dağılım ≈0, kaydırılmış dağılım yüksek, boş dizi hata testi

**DoD:** Bu iki modülde ≥%90 coverage; edge case'ler (sıfır std, boş dizi, `ewma_lambda` sınır dışı) test edilmiş.

---

## M3 — Incident Triage

- [ ] `evaluation/incident_triage.py` → `TriageEvidence`, `count_confirming_signals()`, `requires_retraining_request()`
- [ ] `evaluation/alert_policy.py` → eşikleri `configs/app.yaml -> drift` bölümünden okuyan ince katman
- [ ] `tests/unit/test_incident_triage.py`: 0/1/2/6 sinyal kombinasyonları

**DoD:** `minimum_signals` config'ten okunuyor, hardcode değer yok.

---

## M4 — Persistence

- [ ] `persistence/database.py`: `drift_windows`, `incidents`, `labeling_queue`, `canary_runs`, `rollback_events` tabloları (idempotent migration)
- [ ] `persistence/repositories.py`: `DriftRepository`, `IncidentRepository`, `LabelingQueueRepository`, `CanaryRepository`, `RollbackRepository`
- [ ] `tests/unit/test_database.py` genişlet: 5 yeni tablonun migrate sonrası varlığı

**DoD:** WAL modu korunuyor, mevcut `analyses`/`feedback`/`models`/`thresholds` tabloları etkilenmiyor.

---

## M5 — Services: Drift & Incident

- [ ] `services/drift_monitor_service.py` → `DriftMonitorService.evaluate_window()`
- [ ] `services/incident_service.py` → `IncidentService.open()`, `.resolve()`
- [ ] `tests/integration/test_drift_monitor_service.py`: sentetik batch → `DriftWindow` → (gerekiyorsa) `IncidentRecord`

**DoD:** Servis, `evaluation/` ve `persistence/` dışında hiçbir katmana doğrudan bağımlı değil (UI/CLI hariç).

---

## M6 — Aktif Öğrenme

- [ ] `evaluation/active_learning.py` → `LabelingCandidate`, `select_labeling_batch()` (kota bazlı: disagreement/drift/uncertainty/diversity/random)
- [ ] `services/labeling_service.py` → `LabelingService.build_queue()`, `.record_verdict()`
- [ ] `tests/unit/test_active_learning.py`: kota toplamı bütçeyi aşmıyor, aynı `cluster_id`'den tekrar seçim yok

**DoD:** Kotalar `configs/app.yaml`'dan okunuyor; varsayılan kota toplamı 1.0.

---

## M7 — Model Registry Genişletme + Canary/Rollback

- [ ] `models/registry.py`: `parent_model_id`, `training_trigger_id`, `canary_status` alanları eklendi
- [ ] `services/canary_service.py` → `CanaryService.compare()` (champion/challenger paralel, GPU lock ile sıralı)
- [ ] `services/rollback_service.py` → `RollbackService.rollback()` (hash doğrulamalı)
- [ ] `tests/unit/test_registry_health_logging.py` genişlet
- [ ] `tests/unit/test_canary_rollback.py` (yeni)

**DoD:** Rollback işlemi eski model hash'ini doğrulamadan asla aktif hale getirmiyor; her rollback `RollbackEvent` olarak audit trail'e düşüyor.

---

## M8 — CLI

- [ ] `cli.py`: `drift check`, `drift trend`, `incident list`, `labeling-queue export`, `canary run`, `model rollback` komutları
- [ ] `tests/contract/test_cli.py` genişlet

**DoD:** Her komut `WeaveVisionError`'ı yakalayıp mevcut `_emit()` deseniyle JSON döndürüyor.

---

## M9 — UI

- [ ] `ui/pages/8_Domain_Shift.py`
- [ ] `ui/pages/9_Drift_Incidents.py`
- [ ] `ui/pages/10_Labeling_Queue.py`
- [ ] `ui/pages/11_Canary_and_Rollback.py`
- [ ] `ui/components.py`: paylaşılan drift/incident widget'ları

**DoD:** UI sayfaları yalnızca `services/` çağırıyor; hiçbir sayfa `evaluation/` veya `models/` katmanını doğrudan import etmiyor.

---

## M10 — Config & Dokümantasyon

- [ ] `configs/app.yaml`: `drift:` bölümü eklendi
- [ ] `docs/ARCHITECTURE.md` güncellendi
- [ ] WeaveVision skills bilgi tabanı (proje kök dosyası) yeni modüllerle güncellendi
- [ ] `CHANGELOG.md` kaydı eklendi

---

## M11 — Doğrulama (her milestone sonunda tekrar edilir)

- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run mypy src`
- [ ] `uv run pytest -q`
- [ ] `uv run pytest -m smoke` (RTX 4070 üzerinde GPU testleri; VRAM'i `nvidia-smi` ile izleyerek)
- [ ] `make doctor`

---

## Öncelik Notu

M1→M3 (domain + trend/PSI + triage) dış bağımlılığı olmayan saf mantıktır ve
GPU gerektirmez — Cursor'da hızlıca ve düşük riskle tamamlanabilir. M7 (canary)
ve M9 (UI) en çok zaman alacak, en riskli parçalardır; onlardan önce M1–M6'nın
testleri tam yeşil olmalı.
