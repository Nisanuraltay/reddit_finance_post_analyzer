# 🚀 Reddit Finance Post Analyzer

> *"Yatırım topluluklarında etkileşim ve manipülasyonun anatomisi"*

[![Streamlit App](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://your-streamlit-link.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![BigQuery](https://img.shields.io/badge/BigQuery-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/bigquery)
[![XGBoost R²=0.76](https://img.shields.io/badge/XGBoost-R²_0.76-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

**Ekip:** Nisa Nur Altay · İrem Yaren Rodop · Enes Metehan Özçelik

---

## 📋 İçindekiler

- [Proje Özeti](#-proje-özeti)
- [Araştırma Sorusu](#-araştırma-sorusu)
- [Veri Seti](#-veri-seti)
- [Analiz Pipeline](#-analiz-pipeline)
- [Temel Bulgular](#-temel-bulgular)
- [ML Modeli](#-ml-modeli--xgboost)
- [Canlı Uygulama](#-canlı-uygulama)
- [Looker Studio Dashboard](#-looker-studio-dashboard)
- [Proje Yapısı](#-proje-yapısı)
- [GitHub'a Ne Yüklemeli?](#-githuba-ne-yüklemeli--adım-adım-rehber)
- [Teknolojiler](#-teknolojiler)
- [Ekip](#-ekip)

---

## 🎯 Proje Özeti

Reddit'teki finans ve yatırım topluluklarında paylaşılan **425.700+ gönderiyi** analiz eden, etkileşimi tahmin eden ve manipülasyon riskini ölçen bir karar destek sistemi.

Projenin iki çıktısı var:

| Çıktı | Ne Yapar |
|-------|----------|
| 📊 **Looker Studio Dashboard** | 6 analiz sayfası — zaman, içerik türü, hype, anomali, GME case study |
| 🖥️ **Streamlit App** | Gerçek zamanlı 2 modlu analiz aracı |

### Uygulama Kullanım Modları

```
🔍 YATIRIMCI MODU          ✨ İÇERİK ÜRETİCİ MODU
────────────────────        ──────────────────────
Reddit URL gir         →    Taslak başlık yaz
Manipülasyon riski al  →    Viral öneriler al
Hype kelime tespiti    →    Zamanlama tavsiyesi
```

---

## 🎯 Araştırma Sorusu

> *"Bir Reddit gönderisinin etkileşimi önceden tahmin edilebilir mi ve bu etkileşimin organik mi manipülatif mi olduğu ayırt edilebilir mi?"*

**Neden bu soru önemli?**
- Yüksek score ≠ güvenilir içerik
- Yüksek yorum ≠ olumlu topluluk tepkisi
- Hype dili yatırımcıyı yanıltabilir
- Zamanlama yanlışsa içerik görünmez olur

---

## 📊 Veri Seti

**Kaynak:** Google BigQuery — Reddit Public Dataset (Pushshift)

| Özellik | Değer |
|---------|-------|
| Toplam Gönderi | 425.700+ |
| Subreddit Sayısı | 14 finans topluluğu |
| Zaman Aralığı | Ocak–Aralık 2021 (tam yıl) |
| Ham Özellik | 12 sütun |
| Engineered Feature | +6 yeni metrik |

**Kapsanan Subredditler:**
`wallstreetbets` · `stocks` · `investing` · `finance` · `gme` · `options` · `forex` · `pennystocks` · `personalfinance` · `financialindependence` · `robinhood` · `securityanalysis` · `stockmarket` · `robinhoodpennystock`

**Her gönderi için kullanılan metrikler:**

```
Yazar · Tarih/Gün/Ay/Yıl/Saat · Başlık · Yorum Sayısı · Score
Upvote Oranı · Crosspost Sayısı · Metin mi? · Video var mı?
Subreddit · Sabitlendi mi? · Arşivlendi mi?
```

---

## 🔬 Analiz Pipeline

```
Google BigQuery (14 tablo, 425K+ satır)
            ↓
  Temizleme & Feature Engineering
  (Zaman · İçerik · Hype · Duygu · Ödül)
            ↓
    Keşifsel Veri Analizi (EDA)
    ├── Zaman Analizi
    ├── Upvote Ratio Kalite Sınıflandırması
    ├── İçerik Türü Karşılaştırması
    ├── Crosspost Viralite Analizi
    ├── Hype & Anomali Tespiti
    └── Ödül Enflasyonu
            ↓
      ML Modeli (XGBoost)
      Log-dönüşümlü score tahmini
            ↓
    Karar Destek Sistemi
    ├── Looker Studio Dashboard
    └── Streamlit App (2 mod)
```

---

## 🔍 Temel Bulgular

### 1 · Zaman Analizi — Ne Zaman Paylaşmalı?

En yüksek etkileşim: **Salı–Perşembe, saat 10:00–11:00**

| Gün/Saat | Ort. Yorum |
|----------|-----------|
| Çarşamba 10:00 | **321** |
| Pazartesi 10:00 | **314** |
| Salı 10:00 | **287** |
| Cumartesi 10:00 | 36 *(çok düşük)* |

> Hafta sonu etkileşimi hafta içinin ~%10'u — pazartesi sabahı yeniden başlıyor.

---

### 2 · Upvote Ratio ve Topluluk Kalitesi

| Kategori | Oran | Post Sayısı |
|----------|------|------------|
| ✅ Fikir Birliği (≥0.90) | 42.7% | 299.574 |
| ⚠️ Tartışmalı (0.75–0.90) | 42.2% | 112.665 |
| ❌ Düşük Güven (<0.75) | 15.1% | 12.730 |
| 🥊 Kaos/Kavga | — | 712 |

> "Kaos" kategorisi: Beğenilmemiş ama çok konuşulmuş — manipülasyon şüphelisi.

---

### 3 · İçerik Türü — Format Farkı Yarıyor

| Format | Yorum Üretimi | Beğeni |
|--------|--------------|--------|
| 📝 Metin (%73.9) | ~5× daha fazla | Orta |
| 🔗 Link (%26.1) | Düşük | ~140 puan |
| 🎥 Video (%1.5) | En az | En yüksek |

> **Kural:** Tartışma istiyorsan metin, görünürlük istiyorsan link/video.

---

### 4 · Crosspost Etkisi — Viralite Nasıl Çalışıyor?

| Crosspost | Ort. Score |
|-----------|-----------|
| 0 kez | 63.77 |
| 1 kez | 662.95 |
| 2 kez | 1.687.15 |
| 3+ kez | **6.228.58** |

> 3+ crosspost → score **~98× artıyor.** Ama upvote ratio düşüyor — kalite ile viralite çelişiyor.

---

### 5 · Hype & Anomali Tespiti

**Hype kelime sözlüğü** (`app.py`'dan):
```
moon · rocket · yolo · squeeze · diamond · hands · ape
short · buy · hold · lfg · gem · pump
```

**Anomali Skoru formülü:**
```
anomaly = normalize(hype_score) + |sentiment_score| + (1 - upvote_ratio)
```

**GME Case Study:**
- Ocak 2021 short squeeze döneminde anomali skoru 0.30'a ulaştı
- Normal seviye: 0.15
- Anomali modeli, GME çılgınlığını **trend haline gelmeden önce** işaretleyebilir

**Hype–güven korelasyonu:** −0.02 → Reddit topluluğu hype kelimelere karşı görece bağışıklı, ama manipülatif gönderiler var.

---

### 6 · Ödül Enflasyonu

`award_inflation_score = total_awards / (|score| + num_comments + 1)`

Bazı subredditlerde organik etkileşimden bağımsız biçimde yoğun ödül veriliyor — koordineli görünürlük artırma işareti.

---

## 🤖 ML Modeli — XGBoost

**Hedef:** Reddit post score'unu tahmin et (log-dönüşüm uygulandı)

### Model Karşılaştırması

| Model | R² | MAE |
|-------|----|-----|
| Linear Regression | 0.4324 | 0.9042 |
| Random Forest | 0.7589 | 0.5962 |
| **XGBoost ✓** | **0.7620** | **0.5932** |

### XGBoost Detay

| Metrik | Değer | Yorum |
|--------|-------|-------|
| R² Score | 0.7620 | Varyansın %76'sını açıklıyor |
| MAE | 0.5932 | Düşük ortalama hata |
| RMSE | 0.8051 | Uç değerlere dayanıklı |

**Model input özellikleri** (`app.py`'dan):
```python
features = [
    'sentiment_score',   # VADER duygu skoru
    'hype_count',        # Hype kelime sayısı
    'title_len',         # Başlık uzunluğu
    'saat',              # Paylaşım saati
    'emoji_count',       # Emoji sayısı
    'sub_wallstreetbets', 'sub_stocks', ...  # Subreddit one-hot
]
```

> Neden XGBoost? Gradient boosting yapısı, hype skoru + sentiment + subreddit gibi farklı skalalardaki özellikleri birlikte en iyi şekilde ele alıyor. Model sadece score tahmini için değil, **hype ve manipülasyonu işaretleyen erken uyarı sinyali** olarak tasarlandı.

---

## 🖥️ Canlı Uygulama

**Demo:** [Streamlit linkinizi buraya ekleyin](https://your-streamlit-link.streamlit.app)

### Akış Diyagramı

```
Kullanıcı Girişi                  Sistem Çıktısı
─────────────────                 ──────────────────────────────
📝 URL veya taslak gir     →      📈 Tahmini upvote sayısı
🎯 Subreddit seç           →      🚨 Manipülasyon risk skoru (%0–100)
⏰ Saat seç (0–23)         →      🔥 Hype kelime tespiti (badge'ler)
🔘 "Analiz Et" tıkla       →      💡 Başlık optimizasyon önerileri
                                   ⏰ Saatlik aktivite grafiği
```

### Risk Seviyeleri

| Risk Skoru | Seviye | Anlamı |
|-----------|--------|--------|
| 0–40 | ✅ Düşük | Minimal manipülasyon işareti |
| 41–70 | ⚠️ Orta | Ek kaynaklarla teyit gerekli |
| 71–100 | 🚨 Yüksek | FOMO/pump&dump ihtimali |

---

## 📊 Looker Studio Dashboard

**Dashboard:** [Looker Studio linkinizi buraya ekleyin](#)

Dashboard'u paylaşmak için: *Share → Manage access → "Anyone with link can view"*

| Sayfa | Soru | Görseller |
|-------|------|-----------|
| Genel Bakış | Topluluk nasıl çalışıyor? | KPI'lar + anomali zaman serisi |
| Zaman Analizi | Ne zaman paylaşmalı? | Saat/gün grafiği + ısı haritası |
| Etkileşim Kalitesi | Organik mi, kavga mı? | Konsensus dağılımı + wordcloud |
| Manipülasyon Tespiti | Topluluk manipüle ediliyor mu? | Hype vs upvote + video farkı |
| GME Case Study | Anomali piyasayla örtüşüyor mu? | GME zaman serisi + sentiment |
| İçerik Türü | Hangi format nasıl performans verir? | Text/link/video + crosspost |

---

## 📂 Proje Yapısı

```
reddit-finance-post-analyzer/
│
├── app.py                        ✅ Streamlit uygulaması (2 modlu)
├── final_reddit_model.pkl        ✅ Eğitilmiş XGBoost modeli
├── final_features.pkl            ✅ Model feature listesi
├── metrics.pkl                   ✅ Model metrikleri
├── requirements.txt              ✅ Bağımlılıklar
│
├── notebooks/
│   └── Reddit_Data.ipynb         ⬆️ YÜKLENMELİ — EDA + Feature Engineering
│
├── reports/
│   └── REDDIT_POST_ANALYZER.pdf  ⬆️ YÜKLENMELİ — Looker Studio sunum PDF'i
│
├── data/
│   └── README.md                 ⬆️ YÜKLENMELİ — Veri erişim talimatları
│
├── .devcontainer/                ✅ Mevcut
├── .gitignore                    ⬆️ GÜNCELLENMELİ
└── README.md                     ✅ Bu dosya
```

---

## 🔧 GitHub'a Ne Yüklemeli? — Adım Adım Rehber

### Mevcut Durum (GitHub'da şu an var)

```
✅ app.py
✅ final_reddit_model.pkl
✅ final_features.pkl
✅ metrics.pkl
✅ requirements.txt
✅ .devcontainer/
❌ README.md       → eklenecek (bu dosya)
❌ notebooks/      → klasör oluşturulacak
❌ reports/        → klasör oluşturulacak
❌ data/           → klasör oluşturulacak
```

---

### Adım 1 — `notebooks/` klasörü oluştur ve notebook'u yükle

GitHub'da:
1. **"Add file" → "Create new file"** tıkla
2. Dosya adı: `notebooks/Reddit_Data.ipynb`
3. Elindeki `Reddit_Data.ipynb` dosyasını aç, tüm içeriği kopyala
4. Yapıştır ve **"Commit changes"**

> Ya da lokal Git ile:
> ```bash
> mkdir notebooks
> cp Reddit_Data.ipynb notebooks/
> git add notebooks/
> git commit -m "Add analysis notebook"
> git push
> ```

---

### Adım 2 — `reports/` klasörü oluştur ve PDF'i yükle

1. **"Add file" → "Upload files"** tıkla
2. Dosyayı sürükle: `REDDIT_POST_ANALYZER.pdf`
3. Klasörü belirtmek için dosya adını şöyle yaz: `reports/REDDIT_POST_ANALYZER.pdf`
4. **"Commit changes"**

---

### Adım 3 — `data/README.md` oluştur

Bu dosya veriyi açıklar (CSV yükleme — 425K satır çok büyük):

```markdown
# Data

Veri Google BigQuery üzerinden çekildi.

**BigQuery Projesi:** odev-482215

**Tablolar (14 adet):**
- finance_clean
- gme_clean
- stocks_clean
- wallstreetbets_clean
- ... (diğerleri)

**Erişim:** BigQuery API veya Google Colab authentication gerektirir.
Notebook'ta `from google.colab import auth` ile authenticate olunabilir.

**Zaman Aralığı:** Ocak 2021 – Aralık 2021
**Toplam Kayıt:** ~425.700 gönderi
```

---

### Adım 4 — `.gitignore` güncelle

`.gitignore` dosyasını düzenle, şunları ekle:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.env
venv/

# Büyük veri dosyaları
*.csv
*.parquet
*.json
data/raw/

# Streamlit secrets
.streamlit/secrets.toml

# Jupyter checkpoints
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
```

---

### Adım 5 — `requirements.txt` kontrol et

Şu paketlerin hepsinin olduğundan emin ol:

```
streamlit
pandas
numpy
joblib
plotly
matplotlib
wordcloud
vaderSentiment
xgboost
scikit-learn
```

---

### Adım 6 — README ve badge linklerini güncelle

Bu README'de iki yeri kendi linklerin ile değiştir:

```markdown
# Streamlit linki:
[![Streamlit App](https://img.shields.io/badge/...)](https://YOUR-APP.streamlit.app)

# Looker Studio linki:
**Dashboard:** [Looker Studio](https://lookerstudio.google.com/YOUR-LINK)
```

---

### Final Kontrol Listesi

```
□ README.md yüklendi
□ notebooks/Reddit_Data.ipynb yüklendi
□ reports/REDDIT_POST_ANALYZER.pdf yüklendi
□ data/README.md oluşturuldu
□ .gitignore güncellendi
□ requirements.txt kontrol edildi
□ Streamlit linki badge'e eklendi
□ Looker Studio linki eklendi
```

---

## 🚀 Çalıştırma

```bash
git clone https://github.com/Nisanuraltay/<repo-adı>.git
cd <repo-adı>
pip install -r requirements.txt
streamlit run app.py
```

> **Not:** Notebook'lar Google Colab + BigQuery ortamı için yazılmıştır. Lokal çalıştırmak için BigQuery proje erişimi veya veriyi lokal CSV olarak indirip `file_path` değişkenini güncellemeniz gerekir.

---

## ⚠️ Sınırlamalar

- **BigQuery erişimi:** Ham veri herkese açık değil — `odev-482215` projesine erişim gerektirir
- **VADER sınırları:** İngilizce için optimize — Reddit slang bazen yanlış sınıflandırılabilir
- **UTC zaman dilimi:** Saat verileri UTC tabanlı, lokal saat dilimine göre yorumlanmalı
- **Finansal tavsiye değildir:** Manipülasyon risk skoru bir araştırma çıktısıdır, yatırım kararı olarak kullanılamaz

---

## 🛠️ Teknolojiler

| Kategori | Araç |
|----------|------|
| Dil | Python 3.10+ |
| Veri Kaynağı | Google BigQuery |
| Veri İşleme | Pandas, NumPy |
| NLP | VADER Sentiment |
| ML | XGBoost, Scikit-learn |
| Görselleştirme | Matplotlib, Seaborn, Plotly |
| Dashboard | Looker Studio (Google) |
| Web App | Streamlit |
| Deployment | Streamlit Cloud |

---

## 👥 Ekip

| İsim | Katkı |
|------|-------|
| **Nisa Nur Altay** | EDA · Feature Engineering · Streamlit App |
| **İrem Yaren Rodop** | Analiz · Looker Studio Dashboard |
| **Enes Metehan Özçelik** | ML Modelleme · BigQuery Pipeline |

---

## 📄 Lisans

MIT License — detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

*⭐ Projeyi faydalı bulduysan yıldız vermeyi unutma!*
