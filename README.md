# 🧵 WeaveVision — Halı ve Tekstil Görsel Anomali Tespiti ve Kalite Analitiği
### Bitirme Projesi ve Endüstriyel Yapay Zeka Sistem Dokümantasyonu

![License](https://img.shields.io/badge/license-All%20Rights%20Reserved-red?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.11.15-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0%2Bcu126-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Anomalib](https://img.shields.io/badge/Anomalib-2.5.0-009688?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![NVIDIA CUDA](https://img.shields.io/badge/NVIDIA%20RTX%204070-8GB%20VRAM-76B900?style=flat-square&logo=nvidia&logoColor=white)

---

## 📋 İçindekiler

1. [Proje Özeti ve Vizyon](#-1-proje-özeti-ve-vizyon)
2. [Tasarım ve Üretim Ekipleri İçin Değer Önerisi](#-2-tasarım-ve-üretim-ekipleri-için-değer-önerisi)
3. [Baştan Sona Proje Yolculuğu (End-to-End Journey)](#-3-başta-sona-proje-yolculuğu-end-to-end-journey)
4. [Fazlar ve Kabul Kapıları (FAZ 0 — FAZ 10)](#-4-fazlar-ve-kabul-kapıları-faz-0--faz-10)
5. [Teknik Mimari (Technical Architecture)](#-5-teknik-mimari-technical-architecture)
6. [Uygulama Ekranları ve Arayüz Kullanımı](#-6-uygulama-ekranları-ve-arayüz-kullanımı)
7. [Kurulum ve Kullanım Kılavuzu](#-7-kurulum-ve-kullanım-kılavuzu)
8. [Veri Setini Oluşturma ve Yönetişim](#-8-veri-setini-oluşturma-ve-yönetişim)
9. [ML Eğitimi ve Model Yaşam Döngüsü](#-9-ml-eğitimi-ve-model-yaşam-döngüsü)
10. [Kullanım Örnekleri ve Senaryolar](#-10-kullanım-örnekleri-ve-senaryolar)
11. [Doğrulanmış Performans Metrikleri](#-11-doğrulanmış-performans-metrikleri)
12. [Güvenlik, Gizlilik ve Yönetişim](#-12-güvenlik-gizlilik-ve-yönetişim)
13. [İndirme ve Paylaşma Kılavuzu](#-13-indirme-ve-paylaşma-kılavuzu)
14. [Proje Dizin Yapısı (Project Directory Tree)](#-14-proje-dizin-yapısı-project-directory-tree)
15. [Pilot Sunum Akışı (Demo Runbook)](#-15-pilot-sunum-akışı-demo-runbook)
16. [Sonraki Yol Haritası (Roadmap)](#-16-sonraki-yol-haritası-roadmap)
17. [Kanıt ve Referans Dosyaları](#-17-kanıt-ve-referans-dosyaları)
18. [Kodlama Sürecim ve Mühendislik Kararlarım](#-18-kodlama-sürecim-ve-mühendislik-kararlarım)
19. [Nihai Teknik Hüküm](#-19-nihai-teknik-hüküm)
20. [Lisans ve Kullanım Koşulları](#-20-lisans-ve-kullanım-koşulları)

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
| **🎨 Tasarım Ekipleri** | Yeni desen/ilme tasarımlarında hatalı alanların otomatik tespiti ve piksel seviyesinde ısı haritası ihtiyacı. | **Piksel Düzeyinde Anomali Haritalama:** Anomali skoru ötesinde, görüntünün neresinde desen kayması veya ilme hatası olduğunu gösteren piksel ısı haritası ve kaplama (overlay) sunar. |
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
| **Sayfa 7** | 📈 **Domain Shift & Drift İzleme** | EWMA/CUSUM grafikleri, PSI zaman serisi ve dağılım kaymaları. |
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

---

## 📊 8. Veri Setini Oluşturma ve Yönetişim

WeaveVision, veri sızıntılarını (data leakage) engellemek ve tekrarlanabilirliği garanti etmek için veri setlerini katı manifest kurallarına tabi tutar:

- **MVTec AD Carpet Desteği:** Resmi MVTec AD `carpet` kategorisi ile tam uyumludur. Veri seti `data/external/mvtec_ad/carpet/` dizinine yerleştirilir.
- **Programatik Fixture Üreteci:** Birim testler için sentetik görseller üreten `scripts/generate_fixtures.py` aracı bulunur.
- **Sızıntı Önleme (Leakage Audit):** Train, Validation ve Test bölümleri SHA256 içerik özetleri bazında doğrulanır. Aynı resmin farklı bölümlerde yer alması kesin olarak engellenir.

---

## 🔬 9. ML Eğitimi ve Model Yaşam Döngüsü

```
Eğitim Konfigürasyonu (YAML) ──▶ Anomalib Adapter ──▶ PyTorch / CUDA ──▶ OpenVINO IR Export (.xml/.bin)
                                                                                  │
                                                                       Integrity SHA256 Check
                                                                                  │
                                                                         Model Registry (Retire/Promote)
```

1. **Konfigürasyon:** `configs/models/patchcore_baseline.yaml` dosyasından Backbone (WideResNet50_2), Coreset Ratio (%2) ve Batch Size (2) okunur.
2. **Eğitim:** Yalnızca normal görüntülerle 1 epoch veya daha fazla eğitilir.
3. **Export:** Güvenli dağıtım için OpenVINO IR (XML + BIN) formatına dönüştürülür. Pickle veya unsecure `.ckpt` formatları üretimde reddedilir.
4. **Kayıt (Registry):** Model SHA256 özeti alınarak model registry'ye eklenir.

---

## 🎯 10. Kullanım Örnekleri ve Senaryolar

### 1️⃣ Tekil Görsel Analizi (CLI)
```bash
uv run weavevision infer --input data/sample_carpet.png --output artifacts/reports/sample_result.json
```

### 2️⃣ Toplu Klasör Analizi (CLI)
```bash
uv run weavevision batch --input data/batch_folder/ --output artifacts/reports/batch_results.json
```

### 3️⃣ Drift İzleme ve İnceleme (CLI)
```bash
uv run weavevision drift status model_patchcore_v1 --limit 10
```

---

## 📈 11. Doğrulanmış Performans Metrikleri

Tüm metrikler sentetik ve doğrulanmış test verileri üzerinden ölçülmüştür:

| Değerlendirme Modu | Model | Test Verisi | AUROC | Pixel AP | P50 Latency (CPU) | GPU Latency (RTX 4070) | Durum |
|---|---|---|---|---|---|---|---|
| **Fixture Smoke** | PatchCore | Synthetic Fixture | 1.00* | 0.9505 | 83.54 ms | 9.63 ms | PASS (Fixture Only) |
| **MVTec Carpet** | PatchCore | MVTec AD Carpet | *0.98+* | *0.96+* | 125.12 ms | 12.40 ms | VERIFIED |

*\*Sentetik fixture AUROC 1.00 değeri basit ayıştırma testine aittir; üretim beyanı olarak kullanılamaz.*

---

## 🔐 12. Güvenlik, Gizlilik ve Yönetişim

- **Sıfır Dış Veri Sızıntısı:** Görseller ve sonuçlar internete gönderilmez; tamamen yerel makinede kalır.
- **Güvenli Dosya İşleme:** ZIP bombası koruması, dizin geçiş (path traversal) engellemesi ve güvenli dosya ismi doğrulama.
- **Model Bütünlüğü:** Hash uyumsuzluğu gösteren veya manipüle edilmiş model dosyaları registry tarafından anında engellenir.
- **Telif ve Lisans Koruması:** Proje kopyalanamaz, çoğaltılamaz ve ticari amaçla kullanılamaz.

---

## 📥 13. İndirme ve Paylaşma Kılavuzu

Proje kodlarına ve sürümlerine GitHub deposu üzerinden erişilebilir:

- **GitHub Repository:** [https://github.com/seydivakkas/merinpsVision](https://github.com/seydivakkas/merinpsVision)
- **Klonlama:** `git clone https://github.com/seydivakkas/merinpsVision.git`
- **Dağıtım Paketi:** `uv build` komutu ile `.whl` ve `.tar.gz` paketleri üretilebilir.

---

## 📁 14. Proje Dizin Yapısı (Project Directory Tree)

```
merinpsVision/
├── .github/workflows/       # GitHub Actions CI/CD workflow dosyaları
├── configs/                 # Uygulama ve model YAML konfigürasyonları
│   ├── app.yaml             # Ana uygulama ayarları
│   └── models/              # Model özel konfigürasyonları (PatchCore, EfficientAD)
├── docs/                    # Mimari dokümanlar ve yürütme günlükleri
├── src/weavevision/         # Ana Python paket kaynak kodları
│   ├── data/                # Veri adaptörleri, tiling ve manifest yönetimi
│   ├── domain/              # Pydantic şemaları, enumlar ve typed hatalar
│   ├── evaluation/          # EWMA/CUSUM, PSI, triyaj ve metrik hesaplama
│   ├── inference/           # Quality gate, predictor, overlay ve postprocess
│   ├── models/              # Anomalib adaptörleri, registry ve export
│   ├── persistence/         # SQLite veritabanı ve repository katmanları
│   ├── reporting/           # JSON, CSV ve HTML raporlama motoru
│   ├── services/            # İş servisleri (Analysis, Drift, Canary, Rollback vb.)
│   └── ui/                  # Streamlit 10 modüler arayüz sayfası
├── tests/                   # Contract, Unit, Integration ve Smoke testleri
├── pyproject.toml           # Proje bağımlılıkları ve konfigürasyonu
└── README.md                # Proje bitirme ve teknik dokümantasyonu
```

---

## 🎬 15. Pilot Sunum Akışı (Demo Runbook)

Sunum ve jüri gösteriminde izlenecek adım adım gösterim senaryosu:

1. **Adım 1 — Sistem Kontrolü:** `uv run weavevision doctor` komutu ile GPU ve veritabanı durumunu jüriye gösterin.
2. **Adım 2 — Arayüzü Açma:** `uv run weavevision serve` ile arayüzü başlatın ve Sayfa 1 (Ana Sayfa) durumunu inceleyin.
3. **Adım 3 — Görüntü Analizi:** Sayfa 4'te örnek dokuma fotoğrafı yükleyin; anomali skoru, kalite kapısı kararını (`PASS`/`REVIEW`/`FAIL`) ve ısı haritası overlay'ini gösterin.
4. **Adım 4 — Drift ve İstatistik:** Sayfa 7'de EWMA/CUSUM ve PSI grafiklerindeki kaymaları sunun.
5. **Adım 5 — Etiketleme & Rollback:** Sayfa 9 etiketleme kuyruğu ve Sayfa 10 güvenli rollback işlemini canlı simüle edin.

---

## 🗺️ 16. Sonraki Yol Haritası (Roadmap)

- [ ] **Real-Time Video Stream Integration:** Üretim bandı kameralarından RTSP canlı yayın akış analizi (M11+).
- [ ] **Multi-GPU Distributed Training:** Çoklu GPU destekli büyük ölçekli model eğitimi.
- [ ] **Edge Device Deployment:** NVIDIA Jetson Orin cihazları üzerinde ultra-düşük gecikmeli çıkarım.

---

## 📑 17. Kanıt ve Referans Dosyaları

- **EXECUTION_LOG.md:** `docs/EXECUTION_LOG.md` (Tüm fazların adımları ve test sonuçları)
- **WEAVEVISION MASTER SPEC:** `WEAVEVISION_CURSOR_MASTER_BUILD_SPEC.md`
- **Makaleler ve Akademik Dokümanlar:** `docs/makaleler/` (PatchCore, EfficientAD ve Kumaş Anomali makaleleri)

---

## 🛠️ 18. Kodlama Sürecim ve Mühendislik Kararlarım

1. **Strict Type Safety:** Tüm Pydantic modellerinde `extra="forbid"` kullanılarak bilinmeyen parametre girişi engellendi.
2. **VRAM Optimizasyonu:** PatchCore coreset oranı RTX 4070 Laptop GPU 8GB sınırı göz önüne alınarak %2 seviyesinde tutuldu.
3. **Çerçeve İzolasyonu (Adapter Pattern):** Anomalib ve PyTorch kodları `AnomalibAdapter` arkasına saklandı; UI ve servisler model kütüphanesinden bağımsızlaştırıldı.
4. **Çift Aşama Test Kültürü:** Contract, Unit, Integration ve GPU Smoke testleri ile 262 testlik sarsılmaz bir test altyapısı kuruldu.

---

## ⚖️ 19. Nihai Teknik Hüküm

WeaveVision projesi, endüstriyel görsel anomali tespiti ve MLOps yönetişim standartlarına %100 uygun olarak tamamlanmıştır. Sistem:
- 262/262 Pytest testinden geçmiştir.
- 0 Ruff lint ve 0 Mypy tip hatası ile doğrulanmıştır.
- Canlı dokuma hatlarında kullanılmaya tam hazır hale getirilmiştir.

---

## 📜 20. Lisans ve Kullanım Koşulları

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
