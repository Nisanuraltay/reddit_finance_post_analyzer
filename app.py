import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import re
import os

# --- 1. NLP VE MODEL KURULUMU ---
def setup_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        os.system('pip install vaderSentiment')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

vader_analyzer = setup_vader()

@st.cache_resource
def load_ml_assets():
    try:
        model = joblib.load('final_reddit_model.pkl')
        features = joblib.load('final_features.pkl')
        return model, features
    except:
        return None, None

model, model_features = load_ml_assets()

# --- 2. SAYFA TASARIMI ---
st.set_page_config(page_title="Reddit Finance Hub", layout="wide")
st.title("📈 Reddit Yatırım Toplulukları Analiz Merkezi")

# Tab isimlerini senin istediğin başlıklara göre düzenledim
tab_tahmin, tab_zaman, tab_icerik, tab_hype = st.tabs([
    "🧠 Etkileşim Tahmini", 
    "🕒 Zaman Analizi", 
    "🎥 İçerik Tipi & Kalite", 
    "🚨 Hype & Anomali Tespiti"
])

# --- ÖRNEK VERİ SETİ (Hata almamak için sütunları eşitliyoruz) ---
@st.cache_data
def get_clean_data():
    sub_list = ["finance", "forex", "gme", "investing", "options", "pennystocks", "stocks", "wallstreetbets"]
    n = 500
    df = pd.DataFrame({
        'subreddit': np.random.choice(sub_list, n),
        'score': np.random.randint(1, 5000, n),
        'upvote_ratio': np.random.uniform(0.6, 1.0, n),
        'saat': np.random.randint(0, 24, n),
        'is_video': np.random.choice([0, 1], n),
        'baslik_uzunlugu': np.random.randint(10, 250, n),
        'sentiment_score': np.random.uniform(-1, 1, n),
        'hype_count': np.random.randint(0, 8, n),
        'num_comments': np.random.randint(5, 1000, n)
    })
    return df

data = get_clean_data()

# --- SEKME 1: ETKİLEŞİM TAHMİNİ ---
with tab_tahmin:
    st.subheader("⭐ Gönderi Etkileşim Analizi")
    utitle = st.text_input("Analiz edilecek başlık:", "GME to the moon! 🚀")
    
    if st.button("Analiz Et"):
        v_score = vader_analyzer.polarity_scores(utitle)['compound']
        st.write(f"**Duygu Skoru:** {v_score:.2f}")
        # Model tahmini buraya eklenebilir

# --- SEKME 2: ZAMAN ANALİZİ (Senin 1. Başlığın) ---
with tab_zaman:
    st.subheader("🕒 Günün Saatlerine Göre Etkileşim")
    # Gruplanmış veri ile çizgi grafik
    hourly_avg = data.groupby('saat')['score'].mean().reset_index()
    fig_time = px.line(hourly_avg, x='saat', y='score', markers=True, 
                       title="Saatlik Ortalama Beğeni (Score) Yoğunluğu",
                       template="plotly_dark")
    st.plotly_chart(fig_time, use_container_width=True)

# --- SEKME 3: İÇERİK TİPİ VE KALİTE (Senin 2. ve 3. Başlığın) ---
with tab_icerik:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Popülarite Kalitesi (Upvote Ratio)")
        fig_up = px.histogram(data, x="upvote_ratio", nbins=20, 
                              title="Topluluk Kalite Eşikleri", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_up, use_container_width=True)
    
    with c2:
        st.subheader("🎥 İçerik Türü Etkisi")
        # HATALI KISIM DÜZELTİLDİ: 'not_ched' silindi, 'notched' eklendi
        fig_box = px.box(data, x="is_video", y="score", color="is_video",
                         title="Video vs Metin İçerik Skoru",
                         notched=False, points="all", template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

# --- SEKME 4: HYPE VE ANOMALİ (Senin Hype Başlığın) ---
with tab_hype:
    st.subheader("🚨 Anomali ve Hype Denetimi")
    # Başlık uzunluğu dağılımı (Hata veren diğer grafik)
    fig_dist = px.histogram(data, x='baslik_uzunlugu', 
                            title="İçerik Uzunluğu Dağılımı",
                            color_discrete_sequence=['#AB63FA'],
                            template="plotly_dark")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.subheader("🔍 Şüpheli Hype Kelime Dağılımı")
    fig_hype = px.scatter(data, x="sentiment_score", y="hype_count", size="score", 
                          color="subreddit", title="Duygu vs Hype Yoğunluğu")
    st.plotly_chart(fig_hype, use_container_width=True)
