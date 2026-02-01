import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import re
import os

# --- 1. NLP VE MODEL AYARLARI ---
@st.cache_resource
def setup_nlp():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except:
        os.system('pip install vaderSentiment')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()

vader = setup_nlp()

# --- 2. SENİN 15 SUBREDDIT LİSTEN ---
SUBREDDITS = [
    "finance", "financialindependence", "forex", "gme", "investing", 
    "options", "pennystocks", "personalfinance", "robinhood", 
    "robinhoodpennystock", "securityanalysis", "stockmarket", 
    "stocks", "wallstreetbets", "finance_clean"
]

# --- 3. GÖRSEL STİL VE SAYFA AYARLARI ---
st.set_page_config(page_title="Reddit Finance Dashboard", layout="wide")

# Görsellerdeki "Dark Mode" etkisini pekiştirmek için CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    stTabs [data-baseweb="tab-list"] { gap: 24px; }
    stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("📈 Reddit Finansal Topluluklar Stratejik Analiz Paneli")

# Sekmeleri görsellerindeki başlık sırasına göre dizdim
tab_predict, tab_time, tab_quality, tab_content, tab_hype = st.tabs([
    "🧠 AKILLI TAHMİN", 
    "🕒 1-) ZAMAN ANALİZİ", 
    "📊 2-) POPÜLARİTE KALİTESİ", 
    "✍️ 3-) İÇERİK TİPİ ETKİSİ", 
    "🚨 HYPE DENETİMİ"
])

# --- VERİ SİMÜLASYONU (Görsellerdeki dağılımlara uygun) ---
@st.cache_data
def get_data():
    n = 1200
    return pd.DataFrame({
        'subreddit': np.random.choice(SUBREDDITS, n),
        'saat': np.random.randint(0, 24, n),
        'skor': np.random.lognormal(mean=4, sigma=1, size=n) * 10,
        'upvote_ratio': np.random.uniform(0.7, 1.0, n),
        'baslik_uzunlugu': np.random.normal(70, 30, n).clip(10, 300),
        'sentiment': np.random.uniform(-0.8, 0.8, n),
        'hype_skoru': np.random.poisson(2, n),
        'icerik_turu': np.random.choice(['Yazı', 'Video/Görsel'], n, p=[0.7, 0.3])
    })

df = get_data()

# --- SEKME 1: AKILLI TAHMİN (Gelişmiş Arayüz) ---
with tab_predict:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("🔍 İçerik Girişi")
        user_text = st.text_area("Analiz edilecek başlık:", "TO THE MOON! 🚀🚀🚀 #GME")
        target_sub = st.selectbox("Hedef Topluluk:", SUBREDDITS)
        hour = st.slider("Tahmini Paylaşım Saati:", 0, 23, 12)
        
    if st.button("🚀 ANALİZİ BAŞLAT"):
        v_score = vader.polarity_scores(user_text)['compound']
        risk = min((len(re.findall(r'[🚀💎🔥]', user_text)) * 20) + (abs(v_score) * 30), 100)
        
        with col2:
            st.subheader("📊 Analiz Sonuçları")
            m1, m2 = st.columns(2)
            m1.metric("Duygu Tonu", f"{v_score:.4f}")
            m2.metric("Hype Skoru", f"{risk/10:.1f}/10")
            st.write(f"**Manipülasyon Riski:** %{risk}")
            st.progress(risk/100)
            st.info(f"💡 Öneri: Bu başlık {target_sub} topluluğunda yüksek etkileşim potansiyeline sahip.")

# --- SEKME 2: 1-) ZAMAN ANALİZİ (Görsel 1 & 2 Uyumu) ---
with tab_time:
    st.subheader("🕒 Saatlik Etkileşim ve Paylaşım Yoğunluğu")
    
    # Görseldeki Line Chart (Saatlik Skor)
    hourly_data = df.groupby('saat')['skor'].mean().reset_index()
    fig_line = px.line(hourly_data, x='saat', y='skor', markers=True, 
                       title="Günün Saatlerine Göre Ortalama Beğeni",
                       template="plotly_dark", line_shape="spline", color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig_line, use_container_width=True)
    

# --- SEKME 3: 2-) POPÜLARİTE KALİTESİ (Görsel 3 & 4 Uyumu) ---
with tab_quality:
    st.subheader("📈 Topluluk Kalite ve Beğeni Eşikleri")
    
    # Görseldeki Scatter Plot (Sentiment vs Score)
    fig_scatter = px.scatter(df, x="upvote_ratio", y="skor", color="subreddit", 
                             size="hype_skoru", hover_data=['subreddit'],
                             title="Upvote Oranı vs Etkileşim Skoru",
                             template="plotly_dark", color_continuous_scale="Viridis")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- SEKME 4: 3-) İÇERİK TİPİ ETKİSİ (Görsel 5 Uyumu) ---
with tab_content:
    st.subheader("✍️ Başlık Yapısı ve Tür Analizi")
    c_a, c_b = st.columns(2)
    
    with c_a:
        # Görseldeki Histogram (Başlık Uzunluğu)
        fig_hist = px.histogram(df, x="baslik_uzunlugu", nbins=30, color="icerik_turu",
                                marginal="box", title="Başlık Uzunluğu Dağılımı",
                                template="plotly_dark", color_discrete_sequence=['#636EFA', '#EF553B'])
        st.plotly_chart(fig_hist, use_container_width=True)
        
        
    with c_b:
        # Görseldeki Box Plot (İçerik Türü Performansı)
        fig_box = px.box(df, x="icerik_turu", y="skor", color="icerik_turu",
                         title="İçerik Türüne Göre Skor Dağılımı",
                         template="plotly_dark", points="outliers")
        st.plotly_chart(fig_box, use_container_width=True)

# --- SEKME 5: HYPE DENETİMİ (Görsel 6 Uyumu) ---
with tab_hype:
    st.subheader("🚨 Spekülasyon ve Anomali Takibi")
    fig_hype = px.scatter(df, x="sentiment", y="hype_skoru", size="skor", color="subreddit",
                          title="Duygu Yoğunluğu ve Hype Kelime İlişkisi",
                          template="plotly_dark")
    st.plotly_chart(fig_hype, use_container_width=True)
    
    # Alt tarafa 15 subredditlik bir özet tablo
    st.write("### 📋 Alt Topluluk Performans Özeti")
    summary = df.groupby('subreddit')[['skor', 'upvote_ratio', 'hype_skoru']].mean().sort_values('skor', ascending=False)
    st.dataframe(summary.style.background_gradient(axis=0, cmap='YlGnBu'), use_container_width=True)
