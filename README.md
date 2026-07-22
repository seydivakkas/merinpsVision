# 🧵 WeaveVision — Halı ve Tekstil Görsel Anomali Tespiti ve Kalite Analitiği
### Bitirme Projesi ve Endüstriyel Yapay Zeka Sistem Dokümantasyonu

![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11.15-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2Bcu126-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Anomalib](https://img.shields.io/badge/Anomalib-2.5.0-009688?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA%20RTX%204070-8GB%20VRAM-76B900?style=flat-square&logo=nvidia&logoColor=white)

---

## 📌 1. Proje Özeti ve Vizyon

**WeaveVision**, halı ve tekstil imalat sektöründe dokuma ve baskı hatalarını (çözgü kopması, atkı kaçığı, renk lekesi, iplik birikmesi, desen kayması vb.) insan müdahalesine gerek kalmaksızın, **yalnızca kusursuz (normal) referans görsellerden öğrenen (One-Class Learning)** modüler, uçtan uca yerel bir yapay zeka ve kalite analitiği sistemidir.

### 🎯 Temel Vizyon
Geleneksel kalite kontrol süreçleri insan gözünün yorulması, usta bağımlılığı ve yüksek hatalı kabul/red oranları nedeniyle yüksek maliyet yaratır. WeaveVision:
- **Tek Sınıflı Öğrenme (One-Class Anomaly Detection):** Hatalı numune toplamaya ihtiyaç duymadan, yalnızca normal dokuma örnekleriyle eğitilir.
- **Yerel ve Güvenli (Local-First & Offline):** Tüm veriyi, model ağırlıklarını ve denetim kayıtlarını yerel sistemde tutar; dış servislere bağımlılık yaratmaz.
- **Yönetişim ve Sürüklenme Yönetimi (Drift Lifecycle Governance):** Üretim hattındaki ışık değişimleri, iplik lotu farklılıkları ve kamera açısı kaymalarını (Concept / Data Drift) istatistiksel yöntemlerle (EWMA, CUSUM, PSI) anlık izler ve otomatik alarm / rollback süreçlerini yönetir.

---

## 💡 2. Tasarım ve Üretim Ekipleri İçin Değer Önerisi

| Ekip / Rol | Karşılaşılan Problem | WeaveVision Çözümü & Katma Değer |
|---|---|---|
| **🎨 Tasarım Ekipleri** | Yeni desen/ilme tasarımlarında hatalı alanların otomatik tespiti ve piksel seviyesinde ısı haritası ihtiyacı. | **Piksel Düzeyinde Anomali Haritalama:** Anomali skorunun ötesinde, görüntünün tam olarak neresinde desen kayması veya ilme hatası olduğunu gösteren piksel ısı haritası ve kaplama (overlay) sunar. |
| **🏭 Üretim & Tezgah Operatörleri** | Dokuma esnasında çözgü/atkı kopuklarının geç fark edilmesi nedeniyle metrelerce zayiat oluşması. | **Gerçek Zamanlı Tiled Çıkarım:** 512x512 pencerelerle (tiling) yüksek çözünürlüklü dokuma görüntülerini milisaniyeler içinde işler, anında uyarır. |
| **🔬 Kalite Güvence (QA) Uzmanları** | Şüpheli ürünlerin kararsızlık durumu ve yanlış alarm (False Positive) yükü. | **Kalite Kapısı (Quality Gate):** Sistem sadece `PASS` / `FAIL` vermez. Kararsız durumlarda `REVIEW` veya veri güvenilmezse `ABSTAIN` kararı vererek uzman onayına yönlendirir. |
| **📊 MLOps & Sistem Yöneticileri** | Zamanla değişen ip/ışık koşullarında modelin başarım kaybetmesi (Model Drift). | **Aktif Öğrenme & Otomatik Canary/Rollback:** Kayma tespit edildiğinde en değerli numuneleri greedy-coreset ile etiketleme kuyruğuna atar; yeni modeli canary testine sokar ve gerektiğinde eski modele otomatik rollback yapar. |

---

## 🔄 3. Baştan Sona Proje Yolculuğu (End-to-End Journey)

```
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  1. VERİ KABUL  │───▶│ 2. EĞİTİM & EXPORT│───▶│ 3. THRESHOLD KAL. │───▶│4. MODEL REGISTRY │
│ Manifest & Audit│    │ PatchCore/Eff.AD │    │  Validation-Only  │    │ Integrity Check  │
└─────────────────┘    └──────────────────┘    └───────────────────┘    └──────────────────┘
                                                                                 │
┌─────────────────┐    ┌──────────────────┐    ┌───────────────────┐             │
│ 7. CANARY/ROLLB.│◀───│ 6. ETİKETLEME KUY│◀───│ 5. DRIFT MONITOR  │◀────────────┘
│  Güvenli Geri   │    │ Active Learning  │    │ EWMA+CUSUM+PSI    │    4. ÇIKARIM (INFERENCE)
└─────────────────┘    └──────────────────┘    └───────────────────┘    Tiling + Overlay + Gate
```

1. **Veri Hazırlığı & Yönetişimi:** Ham görüntüler doğrulanır, SHA256 özetleri çıkarılır ve veri sızıntısını engelleyen manifest oluşturulur.
2. **Model Eğitimi (One-Class):** PatchCore (WideResNet50 backbone) veya EfficientAD algoritmalarıyla yalnız normal veriler eğitilir. Model OpenVINO IR formatına export edilir.
3. **Eşik Kalibrasyonu:** Doğrulama seti skorları üzerinden güvenli karar eşiği (threshold) hesaplanır. Sealed test seti kalibrasyona dahil edilmez.
4. **Çıkarım (Inference & Quality Gate):** Yüksek çözünürlüklü dokuma görüntüleri kaydırmalı pencerelerle (tiling) bölünür, çıkarım yapılır, birleştirilir ve kalite kapısından geçirilir.
5. **Sürüklenme İzleme (Drift Monitoring):** Üretim hattından gelen skorlar EWMA ve CUSUM kontrol grafikleriyle ile PSI (Population Stability Index) dağılım analizine tabi tutulur.
6. **Aktif Öğrenme (Active Learning):** Sürüklenme tespit edildiğinde Greedy Coreset algoritması ile en çeşitli ve yüksek riskli numuneler uzman etiketleme kuyruğuna alınır.
7. **Canary & Rollback:** Yeni model canary aşamasında canlı trafikte test edilir. Başarısızlık durumunda GPU kilitli güvenli rollback tetiklenir.

---

## 🚩 4. Fazlar ve Kabul Kapıları (FAZ 0 — FAZ 10)

Proje 11 ardışık faz ve sıkı kabul kriterleri ile tamamlanmıştır:

- [x] **FAZ 0 — Ortam & Donanım Hazırlığı:** UV paket yöneticisi, Python 3.11.15, PyTorch 2.8.0+cu126, CUDA runtime ve RTX 4070 8GB VRAM doğrulaması (`weavevision doctor` PASS).
- [x] **FAZ 1 — Domain Katmanı:** Strict Pydantic veri modelleri (`extra="forbid"`), tip tanımları ve 12 özel typed hata sınıfı.
- [x] **FAZ 2 — Persistence Katmanı:** SQLite WAL modunda çalışan, 5 sürüklenme yönetişim tablosu içeren idempotent veritabanı altyapısı.
- [x] **FAZ 3 — Data Katmanı:** Görüntü dönüşümleri, sızıntısız manifest yönetimi, MVTec AD ve fixture adaptörleri, 512x512 tiling modülü.
- [x] **FAZ 4 — Models Katmanı:** Anomalib 2.5.0 adaptörleri, PatchCore coreset VRAM optimizasyonu (%2 coreset, 8GB VRAM OOM koruması), OOM retry politikası.
- [x] **FAZ 5 — Inference Katmanı:** Kalite kapısı (`PASS`, `REVIEW`, `FAIL`, `ABSTAIN`), mask birleştirme, bölgesel analiz ve görsel overlay üretimi.
- [x] **FAZ 6 — Evaluation Katmanı:** AUROC/PRO metrikleri, validation-only kalibrasyon, EWMA/CUSUM trend izleme, PSI dağılım hesabı, 2-of-N olay triyajı.
- [x] **FAZ 7 — Services Katmanı:** Analysis, Batch, Training, Evaluation, Feedback, Drift Monitor, Incident, Labeling, Canary ve Rollback servisleri.
- [x] **FAZ 8 — Reporting Katmanı:** Şablon tabanlı JSON, CSV ve HTML analiz raporlama motoru.
- [x] **FAZ 9 — CLI ve UI Katmanı:** Typer tabanlı kapsamlı CLI komut seti ve 10 modüler Streamlit kontrol sayfası.
- [x] **FAZ 10 — Tam Doğrulama:** 262/262 Pytest testi (GPU smoke dahil), 0 Ruff lint hatası, 0 Mypy tip hatası ve Execution Log doğrulaması.

---

## 🏗️ 5. Teknik Mimari (Technical Architecture)

Sistem **Local Modular Monolith** mimarisinde tasarlanmıştır. Arayüz ve CLI doğrudan iş mantığını çalıştırmaz; `services/` katmanı üzerinden haberleşir.

```
UI (Streamlit Pages 1-10)  ──┐
CLI (Typer Commands)       ──┼──▶  services/          ──▶  evaluation/    (Saf Mat, CPU)
                                  drift_monitor_service   trend_monitor  (EWMA + CUSUM)
                                  incident_service         psi            (PSI Hesabı)
                                  active_learning_service  incident_triage (2-of-N Triyaj)
                                  canary_service            alert_policy   (Alarm Eşikleri)
                                  model_registry_service
                                 
                               ──▶  persistence/       ──▶  SQLite (WAL Mode)
                                    DriftWindowRepository    drift_windows
                                    IncidentRepository       drift_incidents
                                    LabelingQueueRepository  labeling_queue
                                    CanaryRepository         canary_runs
                                    RollbackRepository       rollback_events
```

### 🎮 RTX 4070 Laptop GPU / VRAM Yönetimi
- **Sürüklenme Matematiği (EWMA, CUSUM, PSI):** %100 CPU/NumPy üzerinde çalışır, GPU VRAM tüketmez.
- **PatchCore Coreset Subsampling:** %10 olan varsayılan değer RTX 4070 8GB VRAM sınırı için `%2` (`0.02`) seviyesine çekilmiştir.
- **Eşzamanlılık Koruması (GPU Lock):** Eğitim ve Canary testlerinin aynı anda GPU'ya yüklenmesini engellemek için `filelock` ile `artifacts/.gpu.lock` mekanizması kullanılır.
- **OOM Retry Policy:** Eğitim esnasında VRAM dolarsa otomatik olarak batch size yarıya düşürülerek eğitim yeniden denenir.

---

## 🖥️ 6. Uygulama Ekranları ve Arayüz Kullanımı

Streamlit arayüzü 10 ana modüler sayfadan oluşur:

| Sayfa No | Sayfa Adı | İşlevi ve Kullanım Senaryosu |
|---|---|---|
| **Sayfa 1** | 🏠 **Ana Sayfa / Sistem Durumu** | Sistem doctor durumu, aktif model/threshold bilgileri ve genel çalışma parametreleri. |
| **Sayfa 2** | 📁 **Veri Seti Yönetimi** | Veri setlerinin denetimi, manifest oluşturma ve sızıntı (leakage) kontrolü. |
| **Sayfa 3** | ⚙️ **Model Eğitimi & Deneyler** | PatchCore veya EfficientAD modellerinin eğitimi, OpenVINO IR export ve VRAM takibi. |
| **Sayfa 4** | 🔍 **Tekil Görüntü Analizi** | Yüklenen dokuma fotoğrafının anomali skoru, kalite kapısı kararı, piksel maskesi ve ısı haritası overlay gösterimi. |
| **Sayfa 5** | 📦 **Toplu (Batch) Analiz** | Klasör veya ZIP biçimindeki çoklu dokuma görsellerinin toplu işlenmesi ve raporlanması. |
| **Sayfa 6** | 📊 **Metrikler ve Değerlendirme** | AUROC, Pixel AP, IoU grafikleri ve kalibrasyon eğrileri. |
| **Sayfa 7** | 📈 **Domain Shift & Drift İzleme** | EWMA/CUSUM grafiklerı, PSI zaman serisi ve dağılım kaymaları. |
| **Sayfa 8** | 🚨 **Drift Olayları (Incidents)** | 2-of-N triyaj kuralı ile tetiklenen P0/P1/P2 seviyeli alarm ve olay yönetimi. |
| **Sayfa 9** | 🏷️ **Etiketleme Kuyruğu (Active Learning)** | Greedy-coreset ile seçilen en riskli görsellerin uzman etiketleme paneli. |
| **Sayfa 10**| 🐥 **Canary Testi ve Rollback** | Yeni modellerin canlı trafik kopyasında testi ve tek tıkla/otomatik sürüm geri alma (rollback). |

---

## 🚀 7. Kurulum ve Kullanım Kılavuzu

### 💻 Sistem Gereksinimleri
- **İşletim Sistemi:** Windows 10/11 veya Linux (Ubuntu 22.04+)
- **Python:** 3.11.x (UV paket yöneticisi önerilir)
- **GPU (Opsiyonel ama Önerilir):** NVIDIA GPU (CUDA 12.6 uyumlu, min 8GB VRAM)

### 🛠️ Kurulum Adımları

1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/seydivakkas/merinpsVision.git
   cd merinpsVision
   ```

2. **Ortamı kurun ve bağımlılıkları yükleyin:**
   ```powershell
   # Windows (PowerShell)
   uv sync --extra dev
   ```

3. **Sistem Sağlık Kontrolünü (Doctor) çalıştırın:**
   ```bash
   uv run weavevision doctor
   ```

4. **Kullanıcı Arayüzünü (Streamlit) Başlatın:**
   ```bash
   uv run weavevision serve
   ```
   Arayüz otomatik olarak `http://localhost:8501` adresinde açılacaktır.

### 💻 CLI Komut Satırı Kullanımı

- **Model Eğitimi:**
  ```bash
  uv run weavevision train --config configs/experiments/smoke.yaml
  ```
- **Tekil Görsel Çıkarımı:**
  ```bash
  uv run weavevision infer --input data/test_image.png --output artifacts/reports/result.json
  ```
- **Drift Durum Kontrolü:**
  ```bash
  uv run weavevision drift status <model_id>
  ```
- **Açık Incident Listesi:**
  ```bash
  uv run weavevision drift incidents
  ```
- **Model Rollback İşlemi:**
  ```bash
  uv run weavevision model rollback <current_model_id> <previous_model_id> --reason DRIFT_WORSENING
  ```

---

## 📜 8. Lisans ve Kullanım Koşulları

![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)

```
ÖZEL LİSANS — TÜM HAKLAR SAKLIDIR

Telif Hakkı (c) 2026 Seydi Eryılmaz (@seydivakkas)

Bu yazılım ve ilgili tüm dosyalar ("Yazılım") yalnızca görüntüleme ve eğitim
amaçlı olarak paylaşılmıştır.

YASAKLAR:
  1. Kopyalanamaz, çoğaltılamaz, dağıtılamaz veya yeniden yayınlanamaz.
  2. Ticari veya ticari olmayan hiçbir projede kullanılamaz, değiştirilemez.
  3. Alt lisanslanamaz, satılamaz veya devredilemez.
  4. Tersine mühendislik yapılamaz.

İZİN VERİLEN KULLANIM:
  - GitHub üzerinde görüntüleme ve okuma.
  - Kişisel öğrenim amacıyla kodu inceleme (kopyalamadan).

YAZARIN AÇIK YAZILI İZNİ OLMAKSIZIN HİÇBİR KULLANIM HAKKI TANINMAZ.
İzin talepleri için: GitHub @seydivakkas
```
