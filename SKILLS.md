# WEAVEVISION — PROJE BİLGİ TABANI (Skills)

> **Son güncelleme:** 2026-07-20
> **Amaç:** AI asistanların (Cursor, Antigravity vb.) bu projeyi hızlıca kavramasını ve gereksiz token harcamasını önlemesini sağlar. Bu dosyayı oku, ardından kod yaz.

---

## 1. PROJE KİMLİĞİ

| Alan | Değer |
|------|-------|
| **Ürün adı** | WeaveVision |
| **Tam başlık** | WeaveVision — Carpet Anomaly Detection and Quality Analytics |
| **Repo dizini** | `c:\Users\seydieryilmaz\Desktop\merinpsVision` |
| **Paket adı** | `weavevision` |
| **Sürüm** | `0.1.0` |
| **Lisans** | Proprietary — Tüm Hakları Saklıdır (Seydi Eryılmaz / @seydivakkas) |
| **Dil** | Python `>=3.11, <3.12` (kilitli) |
| **Paket yönetici** | `uv` (lock dosyası: `uv.lock`) |
| **Build backend** | `hatchling >=1.27` |
| **Entry point** | `weavevision.cli:app` (Typer CLI) |
| **UI** | Streamlit multi-page app (`src/weavevision/ui/app.py`) |
| **DB** | SQLite3 (WAL modu, `artifacts/weavevision.sqlite3`) |
| **Hedef donanım** | NVIDIA RTX 4070 Laptop GPU; CPU fallback zorunlu |

---

## 2. NE YAPAR / NE YAPMAZ

### ✅ Yapar
- Normal halı/kumaş görsellerinden one-class öğrenme (unsupervised anomaly detection)
- Görüntü düzeyinde anomaly score, piksel düzeyinde ısı haritası
- Kusurlu bölge maskesi, kontur ve bounding box
- Tekil ve batch analiz
- JSON, CSV, HTML rapor üretimi
- Model sürüm yönetimi ve hash bütünlük doğrulaması
- Validation-only threshold kalibrasyonu
- Tiled inference (yüksek çözünürlüklü görseller için)
- Girdi kalite kontrolü (quality gate → PASS/REVIEW/ABSTAIN)
- Uzman geri bildirimi (feedback)
- SQLite audit trail
- Degraded-ready UI (model yokken bile açılır)

### ❌ Yapmaz
- Halı deseni **üretmez** (ayrı ürün: carpet-designer)
- Olasılık kalibrasyonu yapılmadan güven yüzdesi göstermez
- Test setiyle threshold belirlemez (veri sızıntısı yasak)
- Kanıtsız metrik/claim yazmaz (`NOT_RUN` kuralı)
- İnsan uzmanının kararını otomatik geçersiz kılmaz
- Cloud/SaaS, mobil uygulama veya ERP entegrasyonu yoktur

---

## 3. TEKNOLOJİ YIĞINI

### Ana Bağımlılıklar
| Paket | Sürüm | Kullanım |
|-------|-------|----------|
| `anomalib` | `2.5.0` (cu126 + openvino) | PatchCore, EfficientAD, PaDiM model eğitimi ve inference |
| `torch` | `2.8.0` | Derin öğrenme backend |
| `torchvision` | `0.23.0` | Görsel ön işleme |
| `streamlit` | `>=1.49, <2` | Web arayüzü |
| `pydantic` | `>=2.11, <3` | Schema validation (tüm domain modelleri) |
| `typer` | `>=0.16, <1` | CLI framework |
| `opencv-python-headless` | `>=4.10, <5` | Görüntü işleme, blur/laplacian |
| `scikit-learn` | `>=1.6, <2` | Metrikler (AUROC, AP, F1, confusion matrix) |
| `scikit-image` | `>=0.24, <1` | Connected component labeling |
| `numpy` | `>=2.0, <3` | Sayısal hesaplama |
| `pandas` | `>=2.2, <4` | CSV rapor |
| `plotly` | `>=6, <7` | İnteraktif grafikler |
| `jinja2` | `>=3.1, <4` | HTML rapor şablonları |
| `orjson` | `>=3.10, <4` | Hızlı JSON serialization |
| `PyYAML` | `>=6.0, <7` | YAML config okuma |
| `rich` | `>=14, <15` | Terminal çıktısı |
| `filelock` | `>=3.18, <4` | Dosya kilitleme |
| `psutil` | `>=7, <8` | Sistem bilgisi (RAM, disk) |

### Dev Bağımlılıklar
`pytest`, `pytest-cov`, `pytest-xdist`, `ruff`, `mypy`, `pre-commit`, `types-PyYAML`, `types-psutil`

### PyTorch CUDA Kaynağı
```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

---

## 4. MİMARİ

**Tip:** Modüler Monolit (mikroservis yok)

### Katman Hiyerarşisi (Bağımlılık yönü: yukarıdan aşağıya)

```
┌─────────────────────────────────────────────────────┐
│  UI (Streamlit)  +  CLI (Typer)                     │  ← Giriş noktaları
├─────────────────────────────────────────────────────┤
│  Services (orchestration layer)                      │  ← İş akışı koordinasyonu
├───────────┬───────────┬──────────┬──────────────────┤
│ Inference │ Evaluation│ Reporting│ Models + Registry │  ← Özellik modülleri
├───────────┴───────────┴──────────┴──────────────────┤
│  Data (adapters, transforms, tiling, manifest)       │  ← Veri erişim
├─────────────────────────────────────────────────────┤
│  Persistence (SQLite database, repositories)         │  ← Kalıcılık
├─────────────────────────────────────────────────────┤
│  Domain (schemas, enums, errors, protocols)           │  ← Saf iş mantığı (bağımlılık yok)
└─────────────────────────────────────────────────────┘
```

### Temel Kural
- **UI** ve **CLI** yalnızca `services/` katmanını çağırır
- **Services** domain, data, inference, evaluation, reporting, persistence modüllerini orkestra eder
- **Domain** katmanı hiçbir dış katmana bağımlı değildir
- Anomalib framework nesneleri `models/anomalib_adapter.py` arkasında kalır

---

## 5. DİZİN YAPISI VE MODÜL SORUMLULUKLARI

### `src/weavevision/` — Ana Paket

| Dosya/Modül | Sorumluluk |
|-------------|-----------|
| `__init__.py` | Paket sürümü (`__version__ = "0.1.0"`) |
| `__main__.py` | `python -m weavevision` desteği |
| `cli.py` | **Typer CLI** — `doctor`, `serve`, `train`, `calibrate`, `infer`, `batch`, `evaluate`, `benchmark`, `dataset audit`, `model list/show/promote` komutları |
| `settings.py` | `AppConfig`, `PathConfig`, `RuntimeConfig`, `InferenceConfig`, `ReportingConfig`, `Settings` Pydantic modelleri + `load_settings()` + `find_project_root()` |
| `logging_config.py` | `JsonFormatter` (JSONL structured logging) + `configure_logging()` |

### `domain/` — Saf İş Mantığı (Dış bağımlılık YOK)

| Dosya | İçerik |
|-------|--------|
| `enums.py` | `Decision` (NORMAL/ANOMALY/REVIEW/ABSTAIN), `QualityGateStatus`, `ReviewPriority` (P0–P3), `ModelStatus` (CANDIDATE→ACTIVE→RETIRED), `ExperimentStatus`, `FeedbackVerdict`, `DatasetVerificationStatus` |
| `schemas.py` | 16 Pydantic ContractModel: `AnalysisResult`, `BatchResult`, `PredictionResult`, `RegionResult`, `QualityGateResult`, `SourceImageMetadata`, `ArtifactPaths`, `TimingResult`, `ModelIdentity`, `ThresholdIdentity`, `DatasetManifest`, `DatasetFile`, `ModelManifest`, `ThresholdArtifact`, `BenchmarkResult` vb. |
| `errors.py` | `WeaveVisionError` (base) + 12 typed error: `ConfigError`, `DatasetNotFoundError`, `DataLeakageError`, `ModelNotReadyError`, `ModelHashMismatchError`, `ThresholdNotFoundError`, `ImageValidationError`, `InferenceError`, `ReportError`, `DatabaseError` vb. Her hata `WV_*` kodu taşır. |
| `protocols.py` | `AnomalyPredictor` Protocol — `model_id`, `predict_array(image_rgb) → (score, map)`, `export(dest)` |

### `data/` — Veri Erişim ve Yönetişim

| Dosya | İçerik |
|-------|--------|
| `manifest.py` | `DatasetManifestBuilder` — dosya hash'leme ve manifest oluşturma |
| `audit.py` | `require_no_leakage()` — split'ler arası veri sızıntısı kontrolü |
| `split.py` | `parent_image_split()` — aynı kaynak görüntüden tile sızıntısını önleyen group-aware split |
| `tiling.py` | `create_tile_coordinates()`, `extract_tiles()`, `merge_tile_maps()` — tiled inference için tile yönetimi |
| `transforms.py` | `load_image_rgb()` — güvenli görüntü yükleme, format doğrulama |
| `adapters/` | Dataset adaptörleri: `base.py` (ABC), `mvtec_ad.py` (MVTec AD carpet), `fixture.py` (sentetik test fixture), `company.py`, `visa.py`, `aitex.py` |
| `adapters/factory.py` | `adapter_from_config(path)` — YAML'dan adapter seçimi |

### `models/` — Model Yönetimi

| Dosya | İçerik |
|-------|--------|
| `anomalib_adapter.py` | `AnomalibAdapter` — Anomalib 2.5 sarmalayıcı, `fit_mvtec()` ve `predict_array()`, OpenVINO inferencer. Sürüm kontrolü zorlar (`2.5.0`). |
| `factory.py` | `create_anomalib_model(algorithm, config)` — patchcore/efficient_ad/padim model fabrikası |
| `registry.py` | `ModelRegistry` — hash-doğrulamalı manifest CRUD, `register()`, `list()`, `get()`, `active()`, `promote()`, `health()`. Durum geçişleri: CANDIDATE → VALIDATED → ACTIVE_BENCHMARK → RETIRED |
| `export.py` | `sha256_artifact(path)` — model dosyası hash hesaplama |

### `inference/` — Tahmin Pipeline

| Dosya | İçerik |
|-------|--------|
| `quality_gate.py` | `evaluate_quality(image_rgb)` — boyut, aspect ratio, blur (Laplacian varyansı), parlaklık, std kontrolü → PASS/REVIEW/ABSTAIN |
| `predictor.py` | `predict_image()` — doğrudan veya tiled tahmin, `RawPrediction` dataclass döner |
| `postprocess.py` | `postprocess_prediction()` — threshold uygulama, maske üretme, karar belirleme |
| `regions.py` | Connected component analizi — bölge bbox, alan, centroid, kontur çıkarma |
| `overlay.py` | `render_artifacts()` — heatmap ve overlay görsel üretimi |

### `evaluation/` — Değerlendirme ve Kalibrasyon

| Dosya | İçerik |
|-------|--------|
| `metrics.py` | `image_metrics()` → AUROC, AP, F1, precision, recall, specificity, confusion matrix; `pixel_metrics()` → pixel AUROC/AP, IoU, Dice; `recall_at_normal_fpr()` |
| `calibration.py` | `calibrate_image_threshold()` — validation-only kalibrasyon (test verisi YASAK), normal_quantile veya recall_at_fpr_then_f1 yöntemi; `calibrate_pixel_threshold()` |
| `robustness.py` | Gaussian noise, blur, brightness perturbation testleri |
| `benchmark.py` | Benchmark rapor yapısı |
| `plots.py` | Plotly görselleştirmeleri |
| `trend_monitor.py` | **[M2]** `monitor_downward_drift()` — CUSUM+EWMA birlikte (CPU/NumPy); `classify_sudden_drop()` — ani düşüşü IncidentPriority'ye sınıflar |
| `psi.py` | **[M2]** `population_stability_index()` — referans/güncel dağılım PSI; `psi_severity()` → LOW/MEDIUM/HIGH |
| `incident_triage.py` | **[M3]** `TriageEvidence` dataclass, `count_confirming_signals()`, `requires_retraining_request()` — 2-of-N sinyal kuralı |
| `alert_policy.py` | **[M3]** `configs/app.yaml → drift` bölümünden eşikleri okuyan ince katman |
| `active_learning.py` | **[M6]** `LabelingCandidate` dataclass, `select_labeling_batch()` — disagreement/drift/uncertainty/diversity/random kota bazlı seçim |

### `services/` — Uygulama Servisleri (Orkestrasyon)

| Dosya | İçerik |
|-------|--------|
| `analysis_service.py` | `AnalysisService.analyze()` — tek görsel analiz transaction: quality_gate → predict → postprocess → overlay → report → persist |
| `batch_service.py` | `BatchService.analyze()` — klasör/ZIP batch analiz, partial-failure isolation |
| `training_service.py` | `TrainingService.train()` — dataset verify → fit → calibrate → register candidate model + tüm provenance artifacts |
| `evaluation_service.py` | Deney değerlendirme servisi |
| `feedback_service.py` | Uzman geri bildirimi kaydetme |
| `report_service.py` | JSON/HTML/CSV rapor üretimi ve artifact kaydetme |
| `health_service.py` | `HealthService.collect()` — sistem durumu: Python, GPU, CUDA, disk, RAM, registry, SQLite. **[M0+]** VRAM used/total da eklenmeli |
| `factory.py` | `load_active_analysis_service()` — hash-doğrulamalı aktif model + threshold ile AnalysisService kompozisyonu |
| `drift_monitor_service.py` | **[M5]** `DriftMonitorService.evaluate_window()` — batch/gün bazlı metrikler, trend_monitor + psi çağırır, `DriftWindow` üretip kaydeder |
| `incident_service.py` | **[M5]** `IncidentService.open()`, `.resolve()` — `DriftWindow` + kanıtları triage'a verir, `IncidentRecord` yönetir |
| `labeling_service.py` | **[M6]** `LabelingService.build_queue()`, `.record_verdict()` — aday havuzunu skorlar, kota bazlı kuyruk oluşturur |
| `canary_service.py` | **[M7]** `CanaryService.compare()` — champion/challenger paralel inference, GPU lock (`artifacts/.gpu.lock`) ile sıralı, `CanaryEvaluation` üretir |
| `rollback_service.py` | **[M7]** `RollbackService.rollback()` — hash doğrulamalı geri alma, `RollbackEvent` audit trail'e kaydeder |

### `persistence/` — Veritabanı

| Dosya | İçerik |
|-------|--------|
| `database.py` | `Database` — SQLite bağlantı yönetimi, WAL modu, idempotent migrate. Tablolar: `schema_meta`, `analyses`, `feedback`, `models`, `thresholds` |
| `repositories.py` | `AnalysisRepository.save()` + `FeedbackRepository` + **[M4]** `DriftRepository`, `IncidentRepository`, `LabelingQueueRepository`, `CanaryRepository`, `RollbackRepository` |
| `migrations.py` | **[M4]** Gelecek schema güncellemeleri; yeni tablolar idempotent `CREATE TABLE IF NOT EXISTS` olarak eklenir |

### `reporting/` — Rapor Üretimi

| Dosya | İçerik |
|-------|--------|
| `json_report.py` | JSON analiz raporu |
| `csv_report.py` | CSV batch raporu |
| `html_report.py` | Jinja2 HTML raporu |
| `templates/` | `analysis_report.html.j2`, `benchmark_report.html.j2` |

### `ui/` — Streamlit Arayüzü

| Dosya | İçerik |
|-------|--------|
| `app.py` | Ana sayfa: model durumu, system doctor, registry özet |
| `state.py` | Session state yönetimi |
| `components.py` | `model_not_ready_notice()` ve paylaşılan UI bileşenleri; **[M9]** drift/incident widget'ları |
| `pages/1_Single_Analysis.py` | Tekli görsel yükleme ve analiz |
| `pages/2_Batch_Analysis.py` | Toplu klasör/ZIP analizi |
| `pages/3_Review_and_Feedback.py` | Uzman inceleme ve geri bildirim |
| `pages/4_Benchmark.py` | Benchmark sonuçları |
| `pages/5_Model_Registry.py` | Model kayıt defteri görünümü |
| `pages/6_System_Health.py` | Sistem sağlık durumu |
| `pages/8_Domain_Shift.py` | **[M9]** Drift penceresi görselleştirme |
| `pages/9_Drift_Incidents.py` | **[M9]** Incident listesi ve yönetimi |
| `pages/10_Labeling_Queue.py` | **[M9]** Etiketleme kuyruğu görünümü |
| `pages/11_Canary_and_Rollback.py` | **[M9]** Canary karşılaştırma ve rollback kontrolü |

---

## 6. VERİ AKIŞI — TEKİL ANALİZ

```
Kullanıcı görsel yükler
    ↓
AnalysisService.analyze(source_path)
    ↓
load_image_rgb() → (image_rgb, SourceImageMetadata)
    ↓
evaluate_quality(image_rgb) → QualityGateResult
    ↓ ABSTAIN → erken çıkış (decision=ABSTAIN)
    ↓ PASS/REVIEW → devam
    ↓
predict_image(predictor, image_rgb, tiled=True)
  └→ create_tile_coordinates → extract_tiles → predict_array (her tile) → merge_tile_maps
    ↓
postprocess_prediction(score, anomaly_map, quality, thresholds)
  └→ threshold → binary mask → connected components → regions → Decision
    ↓
render_artifacts(image, anomaly_map, mask) → heatmap, overlay
    ↓
ReportService.write_analysis() → JSON + HTML + CSV + overlay/mask/heatmap dosyaları
    ↓
AnalysisRepository.save() → SQLite INSERT
    ↓
AnalysisResult döner
```

---

## 7. VERİ AKIŞI — MODEL EĞİTİMİ

```
TrainingService.train(experiment_config.yaml)
    ↓
adapter_from_config(dataset_config) → MVTecADAdapter.verify() → DatasetManifest
    ↓
require_no_leakage(manifest)
    ↓
AnomalibAdapter.fit_mvtec(dataset_root, "carpet", ...)
  └→ Anomalib MVTecAD datamodule → Engine.fit() → save_checkpoint → export OpenVINO → predict(validation)
    ↓
calibrate_image_threshold(normal_scores, split="validation")
    ↓
ModelRegistry.register(ModelManifest) → hash doğrulama → manifest JSON
    ↓
Provenance artifacts: config.resolved.yaml, dataset_manifest.json, leakage_audit.json,
                       thresholds.json, validation_scores.npy, environment.json, run_manifest.json
```

---

## 8. KONFİGÜRASYON

### `configs/app.yaml` — Uygulama ayarları
```yaml
app:                          # AppConfig
  name: WeaveVision
  organization_name: null     # "Merinos" ancak yetkili kurumsal kullanımda
  environment: local
  language: tr
  max_upload_mb: 200
  telemetry: false
paths:                        # PathConfig
  data_root: data
  artifacts_root: artifacts
  database: artifacts/weavevision.sqlite3
runtime:                      # RuntimeConfig
  device: auto                # auto | cpu | cuda | mps
  precision: auto
  num_workers: 4
  deterministic: true
  seed: 42
inference:                    # InferenceConfig
  quality_gate_enabled: true
  tiling_enabled: true
  tile_size: [512, 512]
  tile_overlap: 0.25
  min_component_area_px: 16
  heatmap_alpha: 0.45
reporting:                    # ReportingConfig
  save_original_copy: false
  save_heatmap: true
  save_mask: true
  save_overlay: true
```

### Ayar yükleme mekanizması
- `load_settings(config_path?)` → `find_project_root()` (`WEAVEVISION_CURSOR_MASTER_BUILD_SPEC.md` marker arar) → YAML parse → Pydantic validation → `Settings` nesnesi
- Tüm yollar `project_root` ile resolve edilir: `resolved_data_root()`, `resolved_artifacts_root()`, `resolved_database()`

### Deney konfigürasyonları
- `configs/experiments/` → `smoke.yaml`, `benchmark_mvtec_carpet.yaml`, `robustness.yaml`, `smoke_efficientad.yaml`
- `configs/models/` → `patchcore_baseline.yaml`, `efficientad_challenger.yaml`, `padim_optional.yaml`
- `configs/datasets/` → `mvtec_carpet.yaml`, `fixture.yaml`, `visa.yaml`, `aitex.yaml`, `company_template.yaml`

---

## 9. CLI KOMUTLARI

```bash
# Mevcut komutlar
uv run weavevision doctor [--json] [--config PATH]   # Sistem sağlık kontrolü
uv run weavevision serve                               # Streamlit UI başlat
uv run weavevision train --config PATH [--json]        # Model eğit ve kaydet
uv run weavevision calibrate --run-id ID [--json]      # Threshold yeniden hesapla
uv run weavevision infer --input PATH --output PATH    # Tek görsel analiz
uv run weavevision batch --input PATH --output PATH    # Toplu analiz
uv run weavevision evaluate --run-id ID [--split test] # Metrikleri göster
uv run weavevision benchmark --config PATH             # Benchmark durumu
uv run weavevision dataset audit --config PATH         # Dataset doğrula
uv run weavevision model list                          # Kayıtlı modeller
uv run weavevision model show --model-id ID            # Model detayı (hash doğrulamalı)
uv run weavevision model promote --model-id ID --reason TEXT  # Model terfi

# [M8] Drift lifecycle komutları
uv run weavevision drift check --model-id ID [--json]            # Güncel drift durumu
uv run weavevision drift trend --model-id ID --metric ap50 [--weeks 4]  # Trend geçmişi
uv run weavevision incident list [--priority P1_INCIDENT]        # Incident listesi
uv run weavevision labeling-queue export --output PATH           # Etiketleme kuyruğunu dışa aktar
uv run weavevision canary run --challenger-id ID [--sample-size 200]  # Canary karşılaştırma
uv run weavevision model rollback --to-model-id ID --reason RECALL_DROP  # Rollback
```

---

## 10. TEST YAPISI

```
tests/
├── conftest.py          # Ortak fixture'lar
├── unit/                # ~11 test dosyası — saf mantık testleri
│   ├── test_calibration_metrics.py
│   ├── test_database.py
│   ├── test_evaluation_extended.py
│   ├── test_manifest.py
│   ├── test_models_extended.py
│   ├── test_quality_regions_overlay.py
│   ├── test_registry_health_logging.py
│   ├── test_safe_zip.py
│   ├── test_settings.py
│   ├── test_split_and_audit.py
│   └── test_tiling.py
├── contract/            # Schema ve CLI sözleşme testleri
│   ├── test_cli.py
│   └── test_result_schema.py
├── integration/         # Uçtan uca pipeline testleri
└── smoke/               # GPU smoke testleri
```

### Test çalıştırma
```bash
uv run pytest -q                    # Tüm testler
uv run pytest tests/unit -q         # Sadece unit
uv run pytest -m smoke              # GPU smoke
uv run pytest -m integration        # Integration
uv run ruff check .                 # Lint
uv run ruff format --check .        # Format kontrolü
uv run mypy src                     # Type check (strict mode)
```

### Pytest markers
- `integration` — Integration testleri
- `smoke` — Smoke testleri
- `gpu` — CUDA gerektiren testler

### Coverage
- Kaynak: `weavevision`
- Hariç: `src/weavevision/ui/*`, `src/weavevision/ui/pages/*`

---

## 11. RUFF ve MYPY KURALLARI

### Ruff
```toml
target-version = "py311"
line-length = 100
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["B008"]                    # Typer default argüman deseni
allowed-confusables = ["ı"]          # Türkçe 'ı' harfi
```

### Mypy
```toml
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]
exclude = ["src/weavevision/ui/pages"]
# Ignored imports: cv2, skimage, sklearn, plotly, anomalib
```

---

## 12. KRİTİK KURALLAR

### Veri Bütünlüğü
1. **Test verisi** eğitim, model seçimi veya threshold kalibrasyonu için **ASLA** kullanılmaz
2. **Threshold** yalnızca `validation` split'inden türetilir (`split != "validation"` → ValueError)
3. **Leakage check** zorunlu: `require_no_leakage(manifest)` her eğitimden önce çağrılır
4. **Model hash** kayıt ve yüklemede doğrulanır (`sha256_artifact`)

### Claim / Kanıt Sözleşmesi
- Çalıştırılmamış deney → `NOT_RUN` (ASLA `PASS`)
- Kanıtsız metrik/yüzde/claim yazmak **yasaktır**
- Anomaly score ≠ olasılık (kalibrasyon yapılmadan güven yüzdesi gösterilemez)

### Kod Standartları
- Her public fonksiyon tip annotasyonlu ve docstring'li olmalı
- Tüm dosya yolları `pathlib.Path` kullanır
- Exception'lar asla sessizce yutulmaz
- Structured JSON logging (`weavevision` logger)
- UI servisleri doğrudan model kodu çağırmaz (services katmanı zorunlu)
- `ConfigDict(extra="forbid")` — tüm Pydantic modellerinde fazla alan yasak

### Git'e Eklenmez
- `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
- `data/external/**`, `data/interim/**`, `data/processed/**`
- `artifacts/models/**`, `artifacts/experiments/**`, `artifacts/reports/**`
- `*.ckpt`, `*.pt`, `*.pth`, `*.onnx`, `*.xml`, `*.bin`

---

## 13. HATA KODLARI

| Kod | Hata Sınıfı | Anlam |
|-----|-------------|-------|
| `WV_UNKNOWN` | `WeaveVisionError` | Bilinmeyen hata |
| `WV_CONFIG_INVALID` | `ConfigError` | Geçersiz konfigürasyon |
| `WV_DATASET_NOT_FOUND` | `DatasetNotFoundError` | Veri seti bulunamadı |
| `WV_DATASET_LICENSE_BLOCKED` | `DatasetLicenseBlockedError` | Lisans kabul edilmedi |
| `WV_DATASET_STRUCTURE_INVALID` | `DatasetStructureError` | Veri seti yapısı hatalı |
| `WV_DATA_LEAKAGE_DETECTED` | `DataLeakageError` | Split'ler arası sızıntı |
| `WV_MODEL_NOT_READY` | `ModelNotReadyError` | Aktif model yok |
| `WV_MODEL_HASH_MISMATCH` | `ModelHashMismatchError` | Model hash doğrulanamadı |
| `WV_THRESHOLD_NOT_FOUND` | `ThresholdNotFoundError` | Threshold yok |
| `WV_IMAGE_INVALID` | `ImageValidationError` | Görsel geçersiz |
| `WV_UNSUPPORTED_FORMAT` | `UnsupportedFormatError` | Desteklenmeyen format |
| `WV_INFERENCE_FAILED` | `InferenceError` | Tahmin başarısız |
| `WV_REPORT_FAILED` | `ReportError` | Rapor üretilemedi |
| `WV_DATABASE_FAILED` | `DatabaseError` | SQLite hatası |

---

## 14. ENUM DEĞERLERİ

| Enum | Değerler |
|------|----------|
| `Decision` | `NORMAL`, `ANOMALY`, `REVIEW`, `ABSTAIN` |
| `QualityGateStatus` | `PASS`, `REVIEW`, `ABSTAIN` |
| `ReviewPriority` | `P0`, `P1`, `P2`, `P3`, `ABSTAIN` |
| `ModelStatus` | `CANDIDATE` → `VALIDATED` → `ACTIVE_BENCHMARK` → `ACTIVE_COMPANY_PILOT` → `RETIRED` / `REJECTED` |
| `ExperimentStatus` | `NOT_RUN`, `PASS`, `FAIL`, `BLOCKED`, `PASS_WITH_RESTRICTIONS` |
| `FeedbackVerdict` | `CONFIRMED_NORMAL`, `CONFIRMED_ANOMALY`, `FALSE_POSITIVE`, `FALSE_NEGATIVE`, `UNSURE` |
| `DatasetVerificationStatus` | `VERIFIED`, `BLOCKED`, `INVALID` |

### Drift Lifecycle Enum'ları (`domain/enums.py` — M1 milestone)

| Enum | Değerler |
|------|----------|
| `DriftPattern` | `STABLE`, `SUDDEN`, `GRADUAL`, `SEMANTIC`, `TECHNICAL` |
| `TrendStatus` | `STABLE`, `EWMA_ALERT`, `CUSUM_ALERT`, `BOTH_ALERT` |
| `IncidentPriority` | `INFO`, `P2_REVIEW`, `P1_INCIDENT`, `P0_BLOCKED` |
| `RetrainingStrategy` | `NONE`, `FINE_TUNE`, `FULL_RETRAIN`, `CONTINUAL` |
| `CanaryStatus` | `NOT_RUN`, `RUNNING`, `PASSED`, `FAILED` |
| `RollbackReason` | `HASH_MISMATCH`, `RECALL_DROP`, `FP_SPIKE`, `LATENCY`, `DRIFT_WORSENING`, `SAFETY_ALARM` |

---

## 15. QUALITY GATE DETAYLARI

`evaluate_quality(image_rgb)` kontrolleri:

| Kontrol | Eşik | Sonuç |
|---------|-------|-------|
| Boyut < 64×64 | `min_width/min_height = 64` | ABSTAIN |
| Aşırı aspect ratio | `< 0.1` veya `> 10.0` | ABSTAIN |
| Neredeyse düz görsel | `std < 2.0` | ABSTAIN |
| Bulanıklık | `laplacian_variance < 20.0` | REVIEW |
| Karanlık | `luminance_mean < 12.0` | REVIEW |
| Çok parlak | `luminance_mean > 243.0` | REVIEW |
| Yanlış kanal/bit derinliği | `ndim≠3, channels≠3, dtype≠uint8` | ABSTAIN |

---

## 16. DOSYA YAPISI (ARTIFACT'LAR)

```
artifacts/
├── experiments/          # Eğitim çıktıları (run_id başına)
│   └── <run_id>/
│       ├── checkpoints/model.ckpt
│       ├── config.resolved.yaml
│       ├── dataset_manifest.json
│       ├── leakage_audit.json
│       ├── thresholds.json
│       ├── validation_scores.npy
│       ├── environment.json
│       ├── run_manifest.json
│       └── metrics.json (değerlendirme sonrası)
├── models/
│   └── manifests/        # ModelManifest JSON dosyaları
├── reports/              # Analiz çıktıları
│   └── <analysis_id>/
│       ├── result.json
│       ├── report.html
│       ├── overlay.png
│       ├── mask.png
│       └── heatmap.png
├── benchmarks/
│   └── system_doctor.json
└── weavevision.sqlite3   # Audit veritabanı
```

---

## 17. DATABASE ŞEMASI (SQLite)

```sql
-- Mevcut tablolar
analyses (analysis_id PK, created_at, source_filename, source_sha256,
          decision, review_priority, raw_score, normalized_score,
          anomaly_area_ratio, region_count, model_id, threshold_id,
          quality_status, total_latency_ms, result_json_path)

feedback  (feedback_id PK, analysis_id FK→analyses, created_at, reviewer,
           verdict, defect_type_optional, comment, corrected_mask_path_optional)

models    (model_id PK, algorithm, status, artifact_path, artifact_sha256,
           metrics_path, created_at)

thresholds (threshold_id PK, model_id, image_threshold, pixel_threshold,
            method, status, created_at)

-- [M4] Drift lifecycle tabloları
drift_windows  (window_id PK, created_at, model_id, threshold_id, metric_name,
                window_start, window_end, metric_value, ewma_value, cusum_value,
                psi_value, bbsd_mmd, uae_p95_error, trend_status, drift_pattern,
                source_manifest_sha256)

incidents      (incident_id PK, created_at, priority, drift_pattern, root_cause,
                affected_window_id FK→drift_windows, model_id, threshold_id,
                action_taken, resolved_at)

labeling_queue (item_id PK, created_at, image_sha256, source_path, priority_bucket,
                selection_reason, drift_score, uncertainty_score, assigned_reviewer,
                verdict, reviewed_at)

canary_runs    (canary_id PK, created_at, champion_model_id, challenger_model_id,
                sample_count, disagreement_rate, critical_recall_delta,
                latency_p95_ms, status)

rollback_events (rollback_id PK, created_at, from_model_id, to_model_id, reason,
                 triggered_by, incident_id FK→incidents)
```

---

## 18. DESTEKLENİLEN ALGORİTMALAR

| Algoritma | Durum | Config |
|-----------|-------|--------|
| **PatchCore** | Baseline (birincil) | `configs/models/patchcore_baseline.yaml` |
| **EfficientAD** | Challenger | `configs/models/efficientad_challenger.yaml` |
| **PaDiM** | Opsiyonel | `configs/models/padim_optional.yaml` |

Modeller `ModelManifest.algorithm` alanında `Literal["patchcore", "efficient_ad", "padim"]` olarak kilitlidir.

---

## 19. HIZLI BAŞVURU: YAYGIN İŞLEMLER

### Yeni modül ekleme
1. `src/weavevision/<modül>/` altında dosya oluştur
2. Domain katmanına bağımlılık ekleme (domain bağımsız kalmalı)
3. Test ekle (`tests/unit/test_<modül>.py`)
4. `ruff check . && mypy src && pytest -q` çalıştır

### Yeni CLI komutu ekleme
1. `cli.py` içinde `@app.command()` veya alt Typer'a ekle
2. İş mantığı `services/` katmanında olmalı
3. Hata: `WeaveVisionError` yakalayıp `_emit()` ile JSON çıktı ver
4. `tests/contract/test_cli.py`'ye test ekle

### Yeni dataset adaptörü ekleme
1. `data/adapters/base.py` ABC'sini miras al
2. `data/adapters/factory.py`'de kaydet
3. `configs/datasets/<name>.yaml` oluştur
4. `verify()` → `DatasetManifest` döndür

### Yeni Pydantic schema ekleme
1. `domain/schemas.py`'de `ContractModel`'den türet
2. `ConfigDict(extra="forbid")` — fazla alan yasak
3. Field validasyonları ekle (`Field(gt=0)`, `Field(ge=0.0, le=1.0)` vb.)

---

## 20. DOCKER

```bash
# CPU
docker build -f Dockerfile.cpu -t weavevision:cpu .
# CUDA
docker build -f Dockerfile.cuda -t weavevision:cuda .
# Compose
docker compose up          # CPU
docker compose -f compose.gpu.yaml up  # GPU
```

---

## 21. MERINOS AI SUITE SINIRI

Bu proje (`weavevision`) ile `carpet-designer` projesi tamamen ayrıdır:
- **Paylaşılan**: Hiçbir Python environment, model registry, veri seti veya veritabanı paylaşılmaz
- **İzin verilen**: Yalnızca bağımsız, model-agnostic yardımcı kütüphaneler (güvenli I/O, hash, logging) `shared-libs` olarak paylaşılabilir
- Repository içi göreli import ile iki ürünü bağlamak **yasaktır**

---

## 22. MAKEFILE KISA YOLLARI

```makefile
make doctor      # uv run weavevision doctor
make test        # uv run pytest -q
make lint        # uv run ruff check . && ruff format --check .
make typecheck   # uv run mypy src
make serve       # uv run weavevision serve
```

---

## 23. ÖNEMLİ DOSYA REFERANSLARI

| Dosya | Neden önemli |
|-------|-------------|
| `WEAVEVISION_CURSOR_MASTER_BUILD_SPEC.md` | **73KB** ana spec — tüm fazlar, kurallar, kabul kapıları. Mimari değişiklik öncesi oku. |
| `.cursor/rules/weavevision.mdc` | Cursor Agent kuralları |
| `configs/app.yaml` | Runtime ayarları (drift bölümü de buraya eklenecek — M10) |
| `pyproject.toml` | Bağımlılıklar, CLI entry, lint/test/mypy config |
| `docs/BLOCKERS.md` | Bilinen engeller |
| `docs/EXECUTION_LOG.md` | Yürütme kaydı |
| `docs/ARCHITECTURE.md` | Mimari özet |
| `docs/DRIFT_LIFECYCLE_IMPLEMENTATION.md` | **Drift teknik spesifikasyonu** — CUSUM/EWMA/PSI kod örnekleri, RTX 4070 kısıtları, yeni şemalar/tablolar/config |
| `docs/DRIFT_LIFECYCLE_TASKS.md` | **M0–M11 milestone listesi** — her biri dosya listesi + DoD ile |
| `docs/CURSOR_PROMPT_drift_lifecycle.md` | Cursor Agent'a yapıştırılacak hazır prompt (drift lifecycle) |
| `CHANGELOG.md` | Sürüm geçmişi |

---

---

## 24. DRIFT LİFECYCLE — ÖZET (M0–M11)

> Detay: [`docs/DRIFT_LIFECYCLE_IMPLEMENTATION.md`](docs/DRIFT_LIFECYCLE_IMPLEMENTATION.md) ve [`docs/DRIFT_LIFECYCLE_TASKS.md`](docs/DRIFT_LIFECYCLE_TASKS.md)

### RTX 4070 Laptop (8GB VRAM) Kısıtları

| Kural | Pratik Etki |
|-------|-------------|
| PSI/CUSUM/EWMA → CPU/NumPy | GPU sadece embedding/feature çıkarımı için |
| Embedding → hemen `.detach().cpu().numpy()` | Tensörler GPU'da biriktirilmez |
| Eğitim + canary → asla eş zamanlı | `artifacts/.gpu.lock` (filelock) ile sıralı |
| Mixed precision → `bfloat16` (float16 değil) | Ada Lovelace native, gradient scaler gereksiz |
| Her GPU işlemi sonrası `torch.cuda.empty_cache() + gc.collect()` | VRAM fragmentasyon önlemi |
| PatchCore coreset → %1–2 subsampling | %10 varsayılan OOM riski taşır |
| Arka plan drift kontrolleri → OpenVINO INT8 (CPU) | GPU üretim inference için serbest |

### Yeni `evaluation/` Modülleri

| Modül | Fonksiyon | Açıklama |
|-------|-----------|----------|
| `trend_monitor.py` | `monitor_downward_drift()` | CUSUM + EWMA aynı taramada; baseline yalnızca operational-validation'dan |
| `trend_monitor.py` | `classify_sudden_drop()` | Ani düşüş → `P0_BLOCKED` / `P1_INCIDENT` / `P2_REVIEW` |
| `psi.py` | `population_stability_index()` | Referans kantiline dayalı PSI |
| `psi.py` | `psi_severity()` | PSI < 0.10 → LOW, 0.10–0.25 → MEDIUM, > 0.25 → HIGH |
| `incident_triage.py` | `TriageEvidence` + `requires_retraining_request()` | 2-of-6 sinyal kuralı |
| `active_learning.py` | `select_labeling_batch()` | disagreement/drift/uncertainty/diversity/random/expert kota seçimi |

### Yeni `domain/schemas.py` Şemaları

`DriftWindow`, `TrendPoint`, `IncidentRecord`, `TriageDecision`, `RetrainingRequest`, `LabelingQueueItem`, `CanaryEvaluation`, `RollbackEvent`

Tümü `ConfigDict(extra="forbid")` taşır.

### `configs/app.yaml` — Drift Bölümü (M10)

```yaml
drift:
  ewma_lambda: 0.25           # ⚠️ Başlangıç politikası — operational-validation ile kalibre et
  ewma_limit_sigma: 3.0
  cusum_k_sigma: 0.25
  cusum_h_sigma: 4.0          # ⚠️ Test setiyle kalibre etme (proje kuralı)
  sudden_review_drop_pp: 2.0
  sudden_incident_drop_pp: 5.0
  sudden_block_drop_pp: 10.0
  gradual_window_weeks: 4
  gradual_weekly_drop_pp: 0.5
  gradual_min_consecutive_weeks: 3
  psi_medium_threshold: 0.10
  psi_high_threshold: 0.25
  retraining_min_confirming_signals: 2
  retraining_min_target_images: 200
  retraining_min_labeled_validation_images: 100
  retraining_cooldown_days: 7
```

### Milestone Öncelik Özeti

| Milestone | İçerik | GPU Gereksinimi |
|-----------|--------|-----------------|
| M0 | Hazırlık, branch, doctor | Hayır |
| M1 | Domain enum + şema | Hayır |
| M2 | trend_monitor + psi | Hayır |
| M3 | incident_triage + alert_policy | Hayır |
| M4 | Persistence (5 yeni tablo) | Hayır |
| M5 | drift_monitor_service + incident_service | Hayır |
| M6 | active_learning + labeling_service | Hayır |
| M7 | canary_service + rollback_service | **Evet** (GPU lock) |
| M8 | CLI komutları | Hayır |
| M9 | UI sayfaları 8–11 | Hayır |
| M10 | configs/app.yaml + docs güncelleme | Hayır |
| M11 | Tam doğrulama (ruff+mypy+pytest+smoke) | **Evet** |

> **⚠️ Eşik uyarısı:** `ewma_lambda`, `cusum_h_sigma`, PSI eşikleri ve drop yüzdeleri başlangıç politikasıdır. Test setiyle kalibre edilemez (proje kuralı). Yalnızca ayrı bir operational-validation dönemiyle güncellenmelidir.

---

> **Bu dosya değiştiğinde güncelle.** Yeni modül, enum, hata kodu veya mimari değişiklik olduğunda bu dosyayı da güncelle.
