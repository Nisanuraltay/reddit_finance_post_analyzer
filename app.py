import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import re
import os

# --- 1. NLP VE KÜTÜPHANE KURULUMLARI ---
@st.cache_resource
def setup_tools():
    # VADER ve Matplotlib kontrolü
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        os.system('pip install vaderSentiment matplotlib')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

vader = setup_tools()

# --- 2. 15 SUBREDDIT LİSTESİ ---
SUBREDDITS = [
    "finance", "financialindependence", "forex", "gme", "investing", 
    "options", "pennystocks", "personalfinance", "robinhood", 
    "robinhoodpennystock", "securityanalysis", "stockmarket", 
    "stocks", "wallstreetbets", "finance_clean"
]

# --- 3. SAYFA AYARLARI VE TASARIM ---
st.set_page_config(page_title="Reddit Finance Pro Dashboard", layout="wide")

# Görseldeki Dark Mode etkisini güçlendiren stil
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Reddit Finansal Topluluklar Stratejik Analiz Paneli")

# Görsellerindeki başlık hiyerarşisi
tab_predict, tab_time, tab_quality, tab_content, tab_hype = st.tabs([
    "🧠 AKILLI TAHMİN MOTORU", 
    "🕒 1-) ZAMAN ANALİZİ", 
    "📊 2-) POPÜLARİTE KALİTESİ", 
    "✍️ 3-) İÇERİK TİPİ ETKİSİ", 
    "🚨 HYPE VE ANOMALİ DENETİMİ"
])

# --- VERİ HAZIRLIĞI ---
@st.cache_data
def get_analysis_data():
    n = 1000
    return pd.DataFrame({
        'subreddit': np.random.choice(SUBREDDITS, n),
        'saat': np.random.randint(0, 24, n),
        'skor': np.random.lognormal(3, 1, n) * 20,
        'upvote_ratio': np.random.uniform(0.65, 1.0, n),
        'baslik_uzunlugu': np.random.randint(10, 280, n),
        'sentiment': np.random.uniform(-0.9, 0.9, n),
        'hype_skoru': np.random.poisson(2, n),
        'icerik_turu': np.random.choice(['Metin', 'Video/Görsel'], n)
    })

df = get_analysis_data()

# --- SEKME 1: AKILLI TAHMİN ---
with tab_predict:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("📝 İçerik Denetimi")
        user_text = st.text_area("Analiz edilecek başlık:", "TO THE MOON! 🚀🚀🚀 #GME")
        target_sub = st.selectbox("Hedef Topluluk:", SUBREDDITS)
        
    if st.button("🚀 ANALİZİ BAŞLAT"):
        v_score = vader.polarity_scores(user_text)['compound']
        emoji_count = len(re.findall(r'[🚀💎🔥🦍]', user_text))
        
        with c2:
            st.subheader("📊 Analiz Çıktıları")
            res_1, res_2 = st.columns(2)
            res_1.metric("Duygu Skoru", f"{v_score:.4f}")
            res_2.metric("Emoji Yoğunluğu", emoji_count)
            
            risk = min((emoji_count * 25) + (abs(v_score) * 30), 100)
            st.write(f"**Tahmini Manipülasyon Riski:** %{risk}")
            st.progress(risk/100)

# --- SEKME 2: 1-) ZAMAN ANALİZİ (Line Chart) ---
with tab_time:
    st.subheader("🕒 Saatlik Etkileşim Trendi")
    hourly_avg = df.groupby('saat')['skor'].mean().reset_index()
    fig_line = px.line(hourly_avg, x='saat', y='skor', markers=True, 
                       title="Günün Saatlerine Göre Ortalama Beğeni Yoğunluğu",
                       template="plotly_dark", line_shape="spline")
    st.plotly_chart(fig_line, use_container_width=True)

# --- SEKME 3: 2-) POPÜLARİTE KALİTESİ (Scatter) ---
with tab_quality:
    st.subheader("📈 Topluluk Kalite ve Etkileşim Eşikleri")
    fig_scatter = px.scatter(df, x="upvote_ratio", y="skor", color="subreddit", 
                             size="baslik_uzunlugu", hover_data=['subreddit'],
                             title="Upvote Oranı vs Skor (Boyut: Başlık Uzunluğu)",
                             template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- SEKME 4: 3-) İÇERİK TİPİ ETKİSİ (Histogram & Box) ---
with tab_content:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📏 Başlık Uzunluğu Dağılımı")
        fig_hist = px.histogram(df, x="baslik_uzunlugu", nbins=30, color="icerik_turu",
                                marginal="box", title="Karakter Sayısı Analizi", template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
    with col_b:
        st.subheader("🎥 İçerik Türü Performansı")
        fig_box = px.box(df, x="icerik_turu", y="skor", color="icerik_turu",
                         title="Medyan Skor Kıyaslaması", template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)

# --- SEKME 5: HYPE DENETİMİ VE ÖZET TABLO ---
with tab_hype:
    st.subheader("🚨 Spekülasyon ve Hype Analizi")
    fig_hype = px.scatter(df, x="sentiment", y="hype_skoru", size="skor", color="subreddit",
                          title="Duygu Yoğunluğu vs Hype Kelime Dağılımı", template="plotly_dark")
    st.plotly_chart(fig_hype, use_container_width=True)
    
    st.subheader("📋 Topluluk Bazlı Performans Özeti")
    summary = df.groupby('subreddit')[['skor', 'upvote_ratio', 'hype_skoru']].mean().sort_values('skor', ascending=False)
    
    # HATA KORUMALI TABLO (Matplotlib yoksa düz tablo gösterir)
    try:
        st.dataframe(summary.style.background_gradient(axis=0, cmap='YlGnBu'), use_container_width=True)
    except Exception:
        st.dataframe(summary, use_container_width=True)
