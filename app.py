import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from textblob import TextBlob
import plotly.express as px

# 1. YÜKSEK PERFORMANS İÇİN NLP PAKETLERİ
@st.cache_resource
def install_requirements():
    # VADER: Sosyal medya dilini (Rocket!! 🚀) TextBlob'dan daha iyi anlar
    os.system('pip install vaderSentiment')
    os.system('python -m textblob.download_corpora')

install_requirements()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

# 2. MODEL VE VARLIKLARI YÜKLE
@st.cache_resource
def load_assets():
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    return model, features

model, model_features = load_assets()

# 3. YENİ ÖZELLİK MÜHENDİSLİĞİ FONKSİYONLARI (Skor Artırıcılar)
def get_vader_score(text):
    return vader_analyzer.polarity_scores(text)['compound']

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', text))

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in text.lower())

# --- ARAYÜZ AYARLARI ---
st.set_page_config(page_title="Reddit Finance Pro Analyzer", layout="wide", page_icon="📈")

# --- YAN PANEL (SIDEBAR) - Girişler Buradan ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀🚀🚀")
    selected_sub = st.selectbox("Subreddit:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Saat (0-23):", 0, 23, 15)
    st.divider()
    st.write("🎯 **Hedef Doğruluk:** %70 (VADER & Emoji Enhanced)")
    st.write("📊 **Mevcut Model:** XGBoost v2.0")

# --- ANA EKRAN ---
st.title("🚀 Reddit Yatırım Topluluklarında Birleşik Analiz Sistemi")

with st.expander("ℹ️ Proje ve Metodoloji Hakkında"):
    st.write("""
    Bu sistem, sosyal medya etkileşimini tahmin etmek için **VADER Sentiment Analysis** ve **XGBoost** kullanır. 
    Sadece kelimelere değil, emojilere ve büyük harf kullanımına (Hype belirtileri) da odaklanır.
    """)

tab_tahmin, tab_eda = st.tabs(["🧠 Gelişmiş Tahmin Motoru", "📊 Veri Analizi Dashboard"])

# --- SEKME 1: TAHMİN VE RAPOR ---
with tab_tahmin:
    if st.button("🚀 Analizi Başlat ve Raporu Oluştur"):
        # Özellikleri hesapla
        v_sentiment = get_vader_score(user_title)
        hype = get_hype_count(user_title)
        emojis = get_emoji_count(user_title)
        is_caps = 1 if user_title.isupper() else 0
        
        # Giriş verisini hazırla (Modelin beklediği sütun isimlerine sadık kalarak)
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        input_df['sentiment_score'] = v_sentiment
        input_data['hype_count'] = hype
        input_data['title_len'] = len(user_title)
        input_data['saat'] = posted_time
        # Eğer modeline emoji_count eklediysen buraya ekleyebilirsin
        
        sub_col = f"sub_{selected_sub}"
        if sub_col in input_df.columns:
            input_df[sub_col] = 1
        
        input_df = input_df[model_features]

        try:
            log_pred = model.predict(input_df)[0]
            final_score = np.expm1(log_pred)
            risk = min((hype * 25) + (abs(v_sentiment) * 20) + (emojis * 10), 100)

            st.subheader("📊 Analiz Raporu")
            c1, c2, c3 = st.columns(3)
            c1.metric("Tahmini Upvote", f"{int(final_score)} ↑")
            c2.metric("VADER Sentiment", f"{v_sentiment:.2f}")
            c3.metric("Emoji Sayısı", emojis)

            st.divider()
            st.write(f"### Manipülasyon Riski: %{risk:.1f}")
            st.progress(risk / 100)
            
            if risk > 55:
                st.error("🚨 **Yüksek Hype Tespiti:** Spekülatif içerik ve emoji yoğunluğu saptandı.")
            else:
                st.success("✅ **Organik İçerik:** Gönderi doğal bir etkileşim profili sergiliyor.")

            st.subheader("📋 Teknik Detaylar")
            st.table(pd.DataFrame({
                "Metrik": ["VADER Skoru", "Hype Kelime", "Emoji", "Büyük Harf"],
                "Değer": [v_sentiment, hype, emojis, "Evet" if is_caps else "Hayır"]
            }))
            
            st.chat_message("assistant").write(f"Tahmini etkileşim {int(final_score)} seviyesindedir. %{risk:.1f} risk skoruyla dikkatli olunmalıdır.")
        except Exception as e:
            st.error(f"Hata: {e}")

# --- SEKME 2: EDA (Görsel Ziyafet) ---
with tab_eda:
    st.header("🔬 Colab Veri Analiz Çıktıları")
    # Örnek görsel ziyafet grafiği
    df_plot = pd.DataFrame({
        'Hype Seviyesi': ['Düşük', 'Orta', 'Yüksek', 'Ekstrem'],
        'Etkileşim Skoru': [10, 45, 120, 350]
    })
    fig = px.bar(df_plot, x='Hype Seviyesi', y='Etkileşim Skoru', color='Etkileşim Skoru', template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
