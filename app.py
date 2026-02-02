import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 1. GEREKLİ PAKETİ YÜKLE (VADER)
!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

# 2. GELİŞMİŞ ÖZELLİK MÜHENDİSLİĞİ (Kod 3'ün Revize Hali)
print("⚙️ Özellikler çıkarılıyor...")

# VADER Duygu Analizi (TextBlob'dan daha güçlüdür)
full_df['sentiment_score'] = full_df['title'].apply(lambda x: vader_analyzer.polarity_scores(str(x))['compound'])

# Emoji Sayacı
full_df['emoji_count'] = full_df['title'].apply(lambda x: len(re.findall(r'[^\w\s,.]', str(x))))

# Hype Sayacı
hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
full_df['hype_count'] = full_df['title'].apply(lambda x: sum(1 for word in hype_words if word in str(x).lower()))

# Diğer Metrikler
full_df['title_len'] = full_df['title'].apply(lambda x: len(str(x)))
full_df['is_all_caps'] = full_df['title'].apply(lambda x: 1 if str(x).isupper() else 0)

# 3. VERİ SIZINTISI ÖNLEME VE SEÇİM
exclude_cols = [
    'score', 'num_comments', 'upvote_ratio', 'created', 'retrieved', 'edited',
    'id', 'title', 'selftext', 'author', 'link_flair_text', 'thumbnail', 'shortlink', 'subreddit'
]
# Dinamik olarak sayısal ve kategori tabanlı kolonları seç (Kod 4 Revize)
features = [col for col in full_df.columns if col not in exclude_cols]

X = full_df[features]
y = np.log1p(full_df['score']) # Hedef değişkeni normalize et

# 4. VERİYİ BÖLME
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. YÜKSEK PERFORMANSLI MODEL EĞİTİMİ (Kod 4'ün Optimize Hali)
# Parametreler %70 başarısına odaklanacak şekilde güncellendi
final_model = xgb.XGBRegressor(
    n_estimators=1000, 
    learning_rate=0.05, 
    max_depth=6, 
    subsample=0.8, 
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

print("🚀 Güçlendirilmiş XGBoost eğitiliyor...")
final_model.fit(X_train, y_train)

# 6. SONUÇLAR
y_pred = final_model.predict(X_test)
print(f"🔥 Yeni R2 Skoru (Başarı): %{r2_score(y_test, y_pred)*100:.2f}")

# 7. MODELİ VE ÖZELLİKLERİ KAYDET
joblib.dump(final_model, 'final_reddit_model.pkl')
joblib.dump(features, 'final_features.pkl')
print("✅ İşlem tamam! 'final_reddit_model.pkl' ve 'final_features.pkl' dosyalarını indirebilirsin.")
