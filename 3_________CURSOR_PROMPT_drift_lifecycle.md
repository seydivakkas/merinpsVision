# Cursor Agent Prompt — WeaveVision Drift & Lifecycle Governance

Bunu Cursor'da (Agent/Composer modunda) doğrudan yapıştırın. Önce
`docs/DRIFT_LIFECYCLE_IMPLEMENTATION.md` ve `docs/DRIFT_LIFECYCLE_TASKS.md`
dosyalarını repoya ekleyin (bu paketteki ilk iki dosya).

---

```
ROL
Sen WeaveVision projesinde çalışan kıdemli bir Python/ML platform mühendisisin.
Bu repo modüler monolit mimaride: domain katmanı hiçbir dış katmana bağımlı
değil, services katmanı orkestrasyon yapar, UI (Streamlit) ve CLI (Typer)
yalnızca services'i çağırır. Bu kuralları asla ihlal etme.

BAĞLAM
1. Önce projenin kök dizinindeki WeaveVision skills bilgi tabanı dosyasını oku
   (proje kimliği, mimari, dizin yapısı, kritik kurallar, hata kodları, enum
   değerleri bölümleri).
2. Sonra docs/DRIFT_LIFECYCLE_IMPLEMENTATION.md dosyasını oku — bu, şimdi
   inşa edeceğin drift izleme / incident triage / aktif öğrenme / model
   registry-canary-rollback sisteminin teknik spesifikasyonudur.
3. Sonra docs/DRIFT_LIFECYCLE_TASKS.md dosyasını oku — bu, uygulayacağın
   milestone sırasıdır (M0 → M11).

GÖREV
docs/DRIFT_LIFECYCLE_TASKS.md içindeki milestone'ları SIRAYLA uygula.
Bir milestone'un "DoD" (Definition of Done) kriteri tam sağlanmadan bir
sonraki milestone'a GEÇME.

Her milestone için:
1. İlgili dosyaları oluştur/düzenle (implementasyon spesifikasyonundaki kod
   iskeletlerini temel al, ama gerçek proje konvansiyonlarına — tip
   annotasyonu, docstring, pathlib.Path, structured logging — tam uy).
2. Aynı milestone içinde ilgili testleri yaz.
3. Şunları çalıştır ve hepsi yeşil olmadan durma:
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run pytest -q
4. Ayrı bir git commit yap: "feat(drift): M<n> - <kısa özet>"
5. Bana o milestone'un özetini (değişen dosyalar, eklenen test sayısı, kalan
   riskler) kısaca raporla ve BİR SONRAKİ MİLESTONE'A GEÇMEDEN ÖNCE onay bekle.

SERT KURALLAR (proje bilgi tabanından)
- Test verisi asla eğitim, model seçimi veya threshold kalibrasyonu için
  kullanılmaz. Bu, drift eşikleri (ewma_lambda, cusum_h_sigma, PSI eşikleri,
  sudden/gradual drop yüzdeleri) için de geçerlidir — bunlar sadece ayrı bir
  operational-validation setiyle kalibre edilebilir, sen bu sayıları
  değiştirmeyeceksin, sadece config'ten okunabilir hale getireceksin.
- Çalıştırılmamış/doğrulanmamış hiçbir metrik veya claim yazma. Kanıtın yoksa
  NOT_RUN kullan, asla PASS uydurma.
- Her Pydantic şeması ConfigDict(extra="forbid") taşımalı.
- Exception'lar asla sessizce yutulmaz; her hata tipli bir WeaveVisionError
  alt sınıfı olmalı ve WV_* koduna sahip olmalı.
- Model registry: aktif model artifact'ı asla yerinde değiştirilmez. Her yeni
  eğitim/güncelleme yeni model_id + yeni hash üretir. Rollback, hash
  doğrulaması olmadan asla bir modeli aktif hale getiremez.
- Her public fonksiyon tip annotasyonlu ve docstring'li olmalı.
- Tüm dosya yolları pathlib.Path kullanmalı.

DONANIM KISITI — RTX 4070 LAPTOP GPU (8GB VRAM)
- PSI/CUSUM/EWMA/BBSD-MMD gibi istatistiksel hesapları CPU/NumPy'de yap;
  GPU'yu sadece embedding/feature çıkarımı için kullan ve sonucu hemen
  .detach().cpu().numpy() ile aktar.
- Eğitim ve canary/shadow inference'ı ASLA aynı anda çalıştırma; GPU'yu
  kullanan işlemleri artifacts/.gpu.lock (filelock) ile sıralı hale getir.
- Mixed precision için float16 değil bfloat16 kullan (Ada Lovelace native
  destekler, gradient scaler gerektirmez).
- Her GPU işlemi sonrası torch.cuda.empty_cache() + gc.collect() çağır.
- Eğer bir görev VRAM sınırını aşma riski taşıyorsa, bunu bana açıkça
  bildir ve batch_size / coreset oranı için bir öneri sun; sessizce OOM'a
  düşürme.

BELİRSİZLİK DURUMUNDA
Eşik değeri, kota oranı veya mimari kararda belirsizlik varsa DUR ve bana
sor. Varsayım yapıp sessizce devam etme — özellikle drift eşikleri ve
retraining tetikleri gibi operasyonel karar noktalarında.

BAŞLA
M0'dan başla. Önce mevcut proje yapısını (src/weavevision altındaki
domain/, evaluation/, services/, persistence/, models/, ui/ dizinlerini)
tara ve mevcut konvansiyonları (örnek: domain/enums.py'deki Decision enum'ı,
domain/schemas.py'deki mevcut bir ContractModel) referans alarak yeni
kodun stilini onlarla birebir tutarlı yaz. Sonra M0 checklist'ini uygula.
```

---

## Kullanım Notları

1. Bu paketteki `WEAVEVISION_DRIFT_IMPLEMENTATION.md` dosyasını repoya
   `docs/DRIFT_LIFECYCLE_IMPLEMENTATION.md` olarak, `WEAVEVISION_DRIFT_TASKS.md`
   dosyasını `docs/DRIFT_LIFECYCLE_TASKS.md` olarak kopyalayın.
2. Yukarıdaki kod bloğunu (` ``` ` içindeki kısmı) olduğu gibi Cursor Agent'a
   yapıştırın.
3. Cursor her milestone sonunda sizden onay isteyecek şekilde
   yönlendirildi — özellikle M7 (canary/rollback) ve M9 (UI) öncesi
   çıktıyı gözden geçirmeniz önerilir, çünkü bunlar üretim kararlarını
   doğrudan etkiler.
4. M2 tamamlandığında `evaluation/trend_monitor.py` ve `evaluation/psi.py`
   gerçek geçmiş verinizle (varsa) manuel olarak da doğrulayın; sentetik test
   verisi CUSUM/EWMA'nın matematiksel doğruluğunu kanıtlar ama sizin
   `ewma_lambda`/`cusum_h_sigma` seçiminizin operasyonel olarak doğru
   olduğunu kanıtlamaz.
