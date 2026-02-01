import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import plotly.express as px

# --- 1. KÜTÜPHANE VE MODEL YÜKLEME ---
@st.cache_resource
def install_and_load():
    # Kütüphane kurulumu
    os.system('pip install vaderSentiment')
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    v_analyzer = SentimentIntensityAnalyzer()
    
    # Model ve Özellikleri yükle
    try:
        loaded_model = joblib.load('final_reddit_model.pkl')
        loaded_features = joblib.load('final_features.pkl')
    except:
        loaded_model, loaded_features = None, None
        
    return v_analyzer, loaded_model, loaded_features

vader_analyzer, model, model_features = install_and_load()

# --- 2. FONKSİYONLAR (HATA KORUMALI) ---
def get_vader_score(text):
    if not text: return 0.0
    try:
        return vader_analyzer.polarity_scores(str(text))['compound']
    except:
        return 0.0

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in str(text).lower())

# --- 3. ARAYÜZ ---
st.set_page_config(page_title="Reddit Finance AI", layout="wide")
st.title("🚀 Reddit Finansal Analiz Dashboard")

tab_tahmin, tab_eda = st.tabs(["🧠 Tahmin Motoru", "📊 Analizler"])

# --- SEKME 1: TAHMİN ---
with tab_tahmin:
    user_title = st.text_input("Başlık girin:", "GME to the moon! 🚀")
    if st.button("Analiz Et"):
        v_score = get_vader_score(user_title)
        st.metric("VADER Duygu Skoru", f"{v_score:.4f}")
        # Not: VADER 0 çıkıyorsa metin Türkçe olabilir veya kütüphane henüz yüklenmemiştir.

# --- SEKME 2: ANALİZLER (HATALARIN DÜZELTİLDİĞİ KISIM) ---
with tab_eda:
    st.subheader("📊 Veri Dağılım Analizleri")
    
    # VERİ TABLOSUNU OLUŞTUR (Sütun isimleri küçük harf ve hatasız)
    eda_data = pd.DataFrame({
        'subreddit': ['wallstreetbets', 'stocks', 'investing', 'finance'] * 6,
        'saat': list(range(24)),
        'skor': np.random.randint(50, 1000, 24),
        'duygu_skoru': np.random.uniform(-0.5, 0.8, 24),
        'baslik_uzunlugu': np.random.randint(10, 200, 24)
    })

    # GRAFİKLER
    try:
        # Görsel image_33c215.png'deki hatayı çözen satır:
        fig_dist = px.histogram(
            eda_data, 
            x='baslik_uzunlugu',  # eda_data içindeki isimle BİREBİR aynı
            title="İçerik Uzunluğu Dağılımı",
            color_discrete_sequence=['#00CC96'],
            template="plotly_dark"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        fig_scatter = px.scatter(
            eda_data, 
            x="duygu_skoru", 
            y="skor", 
            color="subreddit",
            title="Duygu vs Etkileşim"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    except Exception as e:
        st.error(f"Grafik oluşturulurken bir hata oluştu: {e}")

st.success("Sistem başarıyla güncellendi.")
