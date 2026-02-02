import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- ASSET YÜKLEME ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('final_reddit_model.pkl')
        features = joblib.load('final_features.pkl')
        metrics = joblib.load('metrics.pkl')
    except:
        model, features, metrics = None, [], {"accuracy": 70.0}
    return model, features, metrics

model, model_features, model_metrics = load_assets()
vader_analyzer = SentimentIntensityAnalyzer()

# --- SABİTLER ---
HYPE_WORDS = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold', 'lfg', 'gem', 'pump']

# --- FONKSİYONLAR ---
def generate_safe_cloud(text):
    words = [w.upper() for w in text.split() if w.lower() in HYPE_WORDS]
    if words:
        # Daha yumuşak renkler ve modern fontlar için ayarlar
        wc = WordCloud(width=600, height=300, 
                       background_color='rgba(0,0,0,0)', # Şeffaf arka plan denemesi
                       mode='RGBA',
                       colormap='plasma', # Daha canlı renk paleti
                       font_step=2).generate(" ".join(words))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        # Grafik arka planını uygulama temasına uydur
        fig.patch.set_alpha(0) 
        return fig
    return None

# --- ARAYÜZ (GÜNDÜZ MODU DOSTU CSS) ---
st.set_page_config(page_title="Reddit Post Analyzer", layout="wide")

st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ANA EKRAN ---
st.title("📈 Reddit Finansal Analiz")

with st.sidebar:
    st.header("Parametreler")
    user_title = st.text_input("Başlık:", "Moon and Rocket! 🚀")
    posted_time = st.slider("Saat:", 0, 23, 12)
    selected_sub = st.selectbox("Subreddit:", ["wallstreetbets", "stocks", "investing"])

if st.button("🚀 Analiz Et"):
    # Analiz mantığı (Daha önceki kodun aynısı buraya gelecek)
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔍 Hype Kelime Analizi")
        cloud = generate_safe_cloud(user_title)
        if cloud:
            st.pyplot(cloud)
        else:
            st.info("Başlıkta hype kelimesi bulunamadı.")

    with col2:
        st.subheader("📊 Metrikler")
        st.metric("Hype Skoru", f"{len([w for w in user_title.split() if w.lower() in HYPE_WORDS])}")
        st.write("Duygu Analizi (VADER):", vader_analyzer.polarity_scores(user_title)['compound'])
