# WeaveVision — Drift İzleme, Model Yaşam Döngüsü, Aktif Öğrenme
## Implementasyon Spesifikasyonu (Cursor için)

> Bu belge, WeaveVision skills bilgi tabanındaki mimari kurallara (domain bağımsız,
> services orkestrasyon, UI/CLI sadece services çağırır) sadık kalarak yazılmıştır.
> Hedef donanım: **RTX 4070 Laptop GPU, 8GB VRAM**, CPU fallback zorunlu.

---

## 0. Kapsam

Önceki araştırma turlarında (drift eşik matrisi, PSI/BBSD/UAE, CUSUM/EWMA,
model registry/rollback, aktif öğrenme) çıkan tasarımı dört uygulanabilir katmana
indirgedim:

| # | Bileşen | Neden önce bu |
|---|---------|----------------|
| 1 | Trend/PSI hesaplama (`evaluation/`) | Her şeyin girdisi; dış bağımlılığı yok, saf matematik |
| 2 | Incident triage + alert policy | Trend çıktısını karara çevirir |
| 3 | Persistence + services orkestrasyonu | Audit trail zorunluluğu (proje kuralı) |
| 4 | Aktif öğrenme + registry genişletme + canary/rollback | Diğer üçüne bağımlı, en pahalı |

UI sayfaları (8–11) ve CLI komutları en sona bırakıldı; onlar sadece services'i sarar.

---

## 1. RTX 4070 Laptop GPU — Donanım Kısıtları ve Tasarım Kararları

RTX 4070 Laptop GPU, Ada Lovelace mimarisinde **8GB GDDR6 VRAM**'e sahiptir (TGP
35–140W arası OEM'e göre değişir, ama VRAM sabittir). Bu proje için pratik sonuçları:

1. **PatchCore coreset subsampling** — 512×512 tile'larda memory bank'ı %1–2
   subsampling oranıyla tutun; %10 varsayılanı 8GB'ta OOM riski taşır.
2. **EfficientAD/PaDiM eğitimi** — `batch_size` 8–16 aralığından başlayın,
   `torch.backends.cudnn.benchmark = True` açın, OOM alırsanız yarıya indirin.
3. **Mixed precision** — Ada Lovelace `bfloat16`'ı native destekler:
   `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. `float16` yerine
   `bfloat16` tercih edin (gradient scaler gerektirmez, daha stabildir).
4. **Drift hesaplamaları GPU'yu meşgul etmemeli.** PSI, CUSUM, EWMA, BBSD-MMD gibi
   istatistiksel hesaplar CPU/NumPy'de çalışmalı. Sadece embedding/feature
   çıkarımı (UAE reconstruction, backbone features) GPU'da batch'lenip hemen
   `.detach().cpu().numpy()` ile aktarılmalı — tensörleri GPU'da biriktirmeyin.
5. **Eş zamanlılık yasağı** — training ve canary/shadow inference'ı aynı anda
   çalıştırmayın (VRAM fragmentasyonu). `filelock` (zaten bağımlılıkta var) ile
   GPU kullanan CLI komutlarını sıralı hale getirin: `artifacts/.gpu.lock`.
6. **Bellek temizliği disiplini** — her eğitim/kalibrasyon adımı sonrası
   `torch.cuda.empty_cache()` + `gc.collect()`.
7. **Arka plan görevleri CPU/OpenVINO'da** — periyodik drift-check gibi düşük
   öncelikli işler OpenVINO INT8 export edilmiş modelle CPU'da çalışsın, GPU
   üretim inference'ı için serbest kalsın.
8. **`health_service.py` genişletmesi** — `nvidia-smi`/`torch.cuda.mem_get_info()`
   ile VRAM used/total'ı doctor çıktısına ekleyin; canary/training başlamadan
   önce eşik altı VRAM varsa erken uyarı verin.

---

## 2. Domain Katmanı

### 2.1 `domain/enums.py` — yeni enum'lar

```python
class DriftPattern(str, Enum):
    STABLE = "STABLE"
    SUDDEN = "SUDDEN"
    GRADUAL = "GRADUAL"
    SEMANTIC = "SEMANTIC"
    TECHNICAL = "TECHNICAL"


class TrendStatus(str, Enum):
    STABLE = "STABLE"
    EWMA_ALERT = "EWMA_ALERT"
    CUSUM_ALERT = "CUSUM_ALERT"
    BOTH_ALERT = "BOTH_ALERT"


class IncidentPriority(str, Enum):
    INFO = "INFO"
    P2_REVIEW = "P2_REVIEW"
    P1_INCIDENT = "P1_INCIDENT"
    P0_BLOCKED = "P0_BLOCKED"


class RetrainingStrategy(str, Enum):
    NONE = "NONE"
    FINE_TUNE = "FINE_TUNE"
    FULL_RETRAIN = "FULL_RETRAIN"
    CONTINUAL = "CONTINUAL"


class CanaryStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"


class RollbackReason(str, Enum):
    HASH_MISMATCH = "HASH_MISMATCH"
    RECALL_DROP = "RECALL_DROP"
    FP_SPIKE = "FP_SPIKE"
    LATENCY = "LATENCY"
    DRIFT_WORSENING = "DRIFT_WORSENING"
    SAFETY_ALARM = "SAFETY_ALARM"
```

### 2.2 `domain/schemas.py` — yeni `ContractModel`'ler

Tümü `ConfigDict(extra="forbid")` taşımalı, proje kuralı gereği fazla alan yasak.

| Şema | Kritik alanlar |
|------|-----------------|
| `DriftWindow` | `window_id`, `model_id`, `threshold_id`, `metric_name`, `window_start/end`, `metric_value`, `ewma_value`, `cusum_value`, `psi_value`, `bbsd_mmd`, `uae_p95_error`, `trend_status: TrendStatus`, `drift_pattern: DriftPattern`, `source_manifest_sha256` |
| `TrendPoint` | `index`, `value`, `ewma`, `ewma_lower_limit`, `cusum_down`, `ewma_alert: bool`, `cusum_alert: bool`, `status: TrendStatus` |
| `IncidentRecord` | `incident_id`, `priority: IncidentPriority`, `drift_pattern: DriftPattern`, `root_cause: str \| None`, `affected_window_id`, `model_id`, `threshold_id`, `action_taken`, `resolved_at: datetime \| None` |
| `TriageDecision` | `decision_id`, `incident_id`, `confirming_signal_count: int = Field(ge=0, le=6)`, `evidence: dict[str, bool]`, `reviewer: str \| None` |
| `RetrainingRequest` | `request_id`, `trigger_id`, `strategy: RetrainingStrategy`, `target_model_family`, `min_target_images: int = Field(gt=0)`, `min_labeled_validation_images: int = Field(gt=0)`, `status: ExperimentStatus` |
| `LabelingQueueItem` | `item_id`, `image_sha256`, `source_path`, `priority_bucket: Literal["P0","P1","P2","P3"]`, `selection_reason`, `drift_score: float \| None`, `uncertainty_score: float \| None`, `assigned_reviewer`, `verdict: FeedbackVerdict \| None` |
| `CanaryEvaluation` | `canary_id`, `champion_model_id`, `challenger_model_id`, `sample_count: int = Field(ge=0)`, `disagreement_rate: float = Field(ge=0.0, le=1.0)`, `critical_recall_delta: float`, `latency_p95_ms: float = Field(ge=0.0)`, `status: CanaryStatus` |
| `RollbackEvent` | `rollback_id`, `from_model_id`, `to_model_id`, `reason: RollbackReason`, `triggered_by`, `incident_id: str \| None` |

Her şema için `tests/unit/test_domain_drift_schemas.py` içinde en az bir geçerli
ve bir `extra field` reddi testi yazılmalı.

---

## 3. `evaluation/` — Saf Hesaplama Modülleri

### 3.1 `evaluation/trend_monitor.py`

CUSUM ve EWMA'yı aynı taramada birlikte hesaplayan tek fonksiyon. Referans
(`baseline_mean`, `baseline_std`) **yalnızca** sabit operational-validation
setinden gelmeli — canlı veriden türetilmez.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

TrendStatusLiteral = Literal["STABLE", "EWMA_ALERT", "CUSUM_ALERT", "BOTH_ALERT"]


@dataclass(frozen=True)
class TrendPoint:
    """Bir gözlem penceresinin drift-trend değerlendirmesi."""

    index: int
    value: float
    ewma: float
    ewma_lower_limit: float
    cusum_down: float
    ewma_alert: bool
    cusum_alert: bool
    status: TrendStatusLiteral


def monitor_downward_drift(
    values: np.ndarray,
    baseline_mean: float,
    baseline_std: float,
    *,
    ewma_lambda: float = 0.25,
    ewma_limit_sigma: float = 3.0,
    cusum_k_sigma: float = 0.25,
    cusum_h_sigma: float = 4.0,
) -> list[TrendPoint]:
    """AP50/recall gibi 'yüksek iyi' metriklerde aşağı yönlü drift izler.

    Parametreler yalnızca configs/app.yaml -> drift bölümünden okunmalı,
    kodda sabitlenmemeli.
    """
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values boş olmayan tek boyutlu bir dizi olmalıdır.")
    if baseline_std <= 0.0:
        raise ValueError("baseline_std sıfırdan büyük olmalıdır.")
    if not 0.0 < ewma_lambda <= 1.0:
        raise ValueError("ewma_lambda (0, 1] aralığında olmalıdır.")

    ewma = baseline_mean
    cusum_down = 0.0
    points: list[TrendPoint] = []

    for index, value in enumerate(values, start=1):
        ewma = ewma_lambda * float(value) + (1.0 - ewma_lambda) * ewma

        ewma_sigma = baseline_std * np.sqrt(
            (ewma_lambda / (2.0 - ewma_lambda))
            * (1.0 - (1.0 - ewma_lambda) ** (2 * index))
        )
        lower_limit = baseline_mean - ewma_limit_sigma * ewma_sigma
        ewma_alert = ewma < lower_limit

        standardized_drop = (baseline_mean - float(value)) / baseline_std
        cusum_down = max(0.0, cusum_down + standardized_drop - cusum_k_sigma)
        cusum_alert = cusum_down > cusum_h_sigma

        if ewma_alert and cusum_alert:
            status: TrendStatusLiteral = "BOTH_ALERT"
        elif ewma_alert:
            status = "EWMA_ALERT"
        elif cusum_alert:
            status = "CUSUM_ALERT"
        else:
            status = "STABLE"

        points.append(
            TrendPoint(
                index=index,
                value=float(value),
                ewma=float(ewma),
                ewma_lower_limit=float(lower_limit),
                cusum_down=float(cusum_down),
                ewma_alert=ewma_alert,
                cusum_alert=cusum_alert,
                status=status,
            )
        )

    return points


def classify_sudden_drop(
    current_value: float,
    prior_value: float,
    *,
    review_drop_pp: float = 2.0,
    incident_drop_pp: float = 5.0,
    block_drop_pp: float = 10.0,
) -> str:
    """İki ardışık pencere arasındaki ani düşüşü IncidentPriority string'ine sınıflar."""
    drop_pp = prior_value - current_value
    if drop_pp >= block_drop_pp:
        return "P0_BLOCKED"
    if drop_pp >= incident_drop_pp:
        return "P1_INCIDENT"
    if drop_pp >= review_drop_pp:
        return "P2_REVIEW"
    return "STABLE"
```

### 3.2 `evaluation/psi.py`

```python
from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Referans ve güncel dağılım arasındaki PSI'yi hesaplar.

    Kova sınırları referans dağılımın kantillerinden türetilir, böylece
    her referans kovası yaklaşık eşit ağırlık taşır.
    """
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference ve current boş olamaz.")

    edges = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_ratio = ref_counts / reference.size + epsilon
    cur_ratio = cur_counts / current.size + epsilon

    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))


def psi_severity(psi_value: float) -> str:
    """PSI değerini WeaveVision politika seviyesine eşler (config'ten okunmalı)."""
    if psi_value < 0.10:
        return "LOW"
    if psi_value < 0.25:
        return "MEDIUM"
    return "HIGH"
```

### 3.3 `evaluation/incident_triage.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriageEvidence:
    """2-of-3 doğrulama kuralı için kanıt setinin anlık görüntüsü."""

    ewma_alert: bool = False
    cusum_alert: bool = False
    psi_high: bool = False
    bbsd_significant: bool = False
    uae_above_p99: bool = False
    labeled_metric_confirmed: bool = False


def count_confirming_signals(evidence: TriageEvidence) -> int:
    return sum(
        (
            evidence.ewma_alert,
            evidence.cusum_alert,
            evidence.psi_high,
            evidence.bbsd_significant,
            evidence.uae_above_p99,
            evidence.labeled_metric_confirmed,
        )
    )


def requires_retraining_request(
    evidence: TriageEvidence, *, minimum_signals: int = 2
) -> bool:
    """En az `minimum_signals` bağımsız kanıt doğrulanmadan retraining talebi açılmaz."""
    return count_confirming_signals(evidence) >= minimum_signals
```

---

## 4. `evaluation/active_learning.py` — İskelet

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelingCandidate:
    image_sha256: str
    uncertainty_score: float
    drift_score: float
    disagreement: bool
    cluster_id: str


def select_labeling_batch(
    candidates: list[LabelingCandidate],
    *,
    budget: int,
    quotas: dict[str, float] | None = None,
) -> list[LabelingCandidate]:
    """Kota bazlı seçim: disagreement -> drift -> uncertainty -> diversity -> random.

    Varsayılan kotalar (config'ten okunmalı):
      disagreement=0.20, drift=0.20, uncertainty=0.15, diversity=0.15,
      random=0.05, critical_expert=0.25
    Her cluster_id'den en fazla bir medoid seçilerek aynı doku/kameranın
    tekrar seçilmesi önlenmeli (diversity guard).
    """
    raise NotImplementedError("Cursor: proje eşik/kota politikasına göre doldurun.")
```

`services/labeling_service.py`, bu fonksiyonu `LabelingQueueRepository` ile
sarmalayıp `LabelingQueueItem` kayıtları üretmeli.

---

## 5. `services/` Sorumlulukları

| Servis | Sorumluluk | Bağımlılık |
|--------|------------|------------|
| `drift_monitor_service.py` | Batch/gün bazlı metrikleri toplar, `trend_monitor` + `psi` çağırır, `DriftWindow` üretip kaydeder | `evaluation/`, `persistence/` |
| `incident_service.py` | `DriftWindow` + kanıtları `incident_triage`'a verir, `IncidentRecord` açar/kapatır | `evaluation/`, `persistence/` |
| `labeling_service.py` | Aday havuzunu skorlar, kota bazlı kuyruk oluşturur, uzman verdict'ini kaydeder | `evaluation/active_learning.py` |
| `canary_service.py` | Champion/challenger'ı aynı görüntülerde paralel çalıştırır, `CanaryEvaluation` üretir | `models/registry.py`, GPU lock |
| `rollback_service.py` | `RollbackReason` tetiklerini değerlendirir, hash doğrulamalı geri alma yapar | `models/registry.py`, `persistence/` |

Tüm servisler `WeaveVisionError` alt sınıflarını fırlatmalı; sessiz `except` yasak.

---

## 6. Persistence — Yeni Tablolar

```sql
CREATE TABLE IF NOT EXISTS drift_windows (
  window_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  model_id TEXT NOT NULL,
  threshold_id TEXT,
  metric_name TEXT NOT NULL,
  window_start TEXT NOT NULL,
  window_end TEXT NOT NULL,
  metric_value REAL,
  ewma_value REAL,
  cusum_value REAL,
  psi_value REAL,
  bbsd_mmd REAL,
  uae_p95_error REAL,
  trend_status TEXT NOT NULL,
  drift_pattern TEXT NOT NULL,
  source_manifest_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
  incident_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  priority TEXT NOT NULL,
  drift_pattern TEXT NOT NULL,
  root_cause TEXT,
  affected_window_id TEXT REFERENCES drift_windows(window_id),
  model_id TEXT,
  threshold_id TEXT,
  action_taken TEXT,
  resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS labeling_queue (
  item_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  image_sha256 TEXT NOT NULL,
  source_path TEXT NOT NULL,
  priority_bucket TEXT NOT NULL,
  selection_reason TEXT NOT NULL,
  drift_score REAL,
  uncertainty_score REAL,
  assigned_reviewer TEXT,
  verdict TEXT,
  reviewed_at TEXT
);

CREATE TABLE IF NOT EXISTS canary_runs (
  canary_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  champion_model_id TEXT NOT NULL,
  challenger_model_id TEXT NOT NULL,
  sample_count INTEGER,
  disagreement_rate REAL,
  critical_recall_delta REAL,
  latency_p95_ms REAL,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rollback_events (
  rollback_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  from_model_id TEXT NOT NULL,
  to_model_id TEXT NOT NULL,
  reason TEXT NOT NULL,
  triggered_by TEXT NOT NULL,
  incident_id TEXT REFERENCES incidents(incident_id)
);
```

`migrations.py` içine idempotent `CREATE TABLE IF NOT EXISTS` bloğu olarak eklenmeli;
mevcut `database.py` migrate akışına dahil edilmeli.

---

## 7. `configs/app.yaml` — Yeni Bölüm

```yaml
drift:
  ewma_lambda: 0.25
  ewma_limit_sigma: 3.0
  cusum_k_sigma: 0.25
  cusum_h_sigma: 4.0
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

> **Kritik:** Bu sayılar başlangıç politikasıdır. Test setiyle kalibre edilemez
> (proje kuralı #2). Yalnızca ayrı bir operational-validation dönemiyle
> sürümlenip güncellenmelidir.

---

## 8. CLI Eklemeleri (`cli.py`)

```
uv run weavevision drift check --model-id ID [--json]
uv run weavevision drift trend --model-id ID --metric ap50 [--weeks 4]
uv run weavevision incident list [--priority P1_INCIDENT]
uv run weavevision labeling-queue export --output PATH
uv run weavevision canary run --challenger-id ID [--sample-size 200]
uv run weavevision model rollback --to-model-id ID --reason RECALL_DROP
```

Her komut `services/` katmanını çağırmalı, `WeaveVisionError` yakalayıp
mevcut `_emit()` deseniyle JSON çıktı vermeli.

---

## 9. Test Gereksinimleri

| Dosya | Asgari senaryo |
|-------|-----------------|
| `tests/unit/test_trend_monitor.py` | stabil seri (alarm yok), ani düşüş, 3+ hafta kademeli düşüş, `baseline_std=0` → `ValueError` |
| `tests/unit/test_psi.py` | aynı dağılım → PSI≈0, kaydırılmış dağılım → PSI yüksek, boş dizi → `ValueError` |
| `tests/unit/test_incident_triage.py` | 0/1/2/6 sinyal kombinasyonları için `requires_retraining_request` |
| `tests/unit/test_active_learning.py` | kota toplamının bütçeyi aşmaması, aynı `cluster_id`'den tekrar seçilmemesi |
| `tests/unit/test_database.py` (genişletme) | yeni 5 tablonun migrate sonrası var olduğu |
| `tests/integration/test_drift_monitor_service.py` | uçtan uca: sentetik batch → `DriftWindow` → gerekiyorsa `IncidentRecord` |

---

## 10. Kabul Kriterleri (DoD)

- `uv run ruff check .` ve `uv run ruff format --check .` temiz
- `uv run mypy src` strict modda hatasız (yeni modüller dahil)
- `uv run pytest -q` yeşil; yeni modüllerde ≥%90 satır kapsamı
- Her public fonksiyon tip annotasyonlu + docstring'li
- Hiçbir eşik değeri kod içine sabit yazılmamış, tümü `configs/app.yaml` üzerinden okunuyor
- Yeni Pydantic şemaları `ConfigDict(extra="forbid")` taşıyor
- Audit trail: her `IncidentRecord`/`RollbackEvent` ilgili model/threshold hash'ini saklıyor
