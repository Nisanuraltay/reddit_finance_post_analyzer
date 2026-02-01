import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import datetime
import os

# TextBlob için gerekli dil paketini indir (Streamlit Cloud için şart)
os.system('python -m textblob.download_corpora')

# 1. Modeli ve Özellik Listesini Yükle
# final_features.pkl: Modelin eğitimde gördüğü SÜTUN SIRALAMASINI tutar.
model = joblib.load('final_reddit_model.pkl') 
model_features = joblib.load('final_features.pkl')

# 2. Yardımcı Fonksiyonlar (Kişi B'nin işleri)
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in text.lower())

# 3. Arayüz Tasarımı (Kişi C'nin işleri)
st.set_page_config(page_title="Reddit Finance Analyzer", page_icon="📈")
st.title("📈 Reddit Finance Post Analyzer")
st.markdown("### Engagement & Hype Risk Engine")

# Kullanıcı Girişleri
user_title = st.text_input("Reddit Başlığını Girin:", "GME to the moon! 🚀")
selected_subreddit = st.selectbox("Subreddit Seçin:", ["wallstreetbets", "stocks", "investing", "finance"])
posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 12)

if st.button("Analiz Et"):
    # --- ÖZELLİK ÇIKARIMI ---
    sentiment = get_sentiment(user_title)
    hype = get_hype_count(user_title)
    title_len = len(user_title)
    
    # Modelin beklediği TÜM sütunları (features) 0 ile hazırla
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    
    # Manuel özellikleri doldur
    input_data['sentiment_score'] = sentiment
    input_data['hype_count'] = hype
    input_data['title_len'] = title_len
    input_data['saat'] = posted_time
    
    # Subreddit encoding'i doldur
    sub_col = f"sub_{selected_subreddit}"
    if sub_col in input_data.columns:
        input_data[sub_col] = 1

    # --- KRİTİK ADIM: SÜTUN HİZALAMA ---
    # Modelin sütunları hangi sırada beklediğini XGBoost'a aynen gönderiyoruz.
    input_data = input_data[model_features]

    # --- TAHMİN ---
    try:
        log_pred = model.predict(input_data)[0]
        final_score = np.expm1(log_pred) # Log'dan gerçek skora dön

        # --- SONUÇLARI GÖSTER ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Beklenen Etkileşim (Score)", f"{int(final_score)} Upvote")
        
        with col2:
            # Risk Mantığı
            if hype > 2 or sentiment > 0.5:
                st.error("🚨 RİSK: YÜKSEK")
                st.write("Aşırı spekülatif içerik!")
            elif hype > 0:
                st.warning("⚠️ RİSK: ORTA")
                st.write("Bazı hype kelimeleri tespit edildi.")
            else:
                st.success("✅ RİSK: DÜŞÜK")
                st.write("Dengeli ve doğal görünüm.")

        # Detaylı Analiz Notu
        st.info(f"**Analiz Özeti:** Bu başlıkta {hype} hype kelimesi ve %{sentiment*100:.1f} duygu yoğunluğu tespit edildi.")
        
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
        st.write("Lütfen model ve özellik dosyalarının GitHub'da güncel olduğundan emin olun.")
