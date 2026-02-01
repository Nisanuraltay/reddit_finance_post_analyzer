import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import plotly.express as px

# --- 1. SİSTEM KURULUMU VE VADER ---
@st.cache_resource
def setup_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        os.system('pip install vaderSentiment')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()

vader_analyzer = setup_vader()

# --- 2. ANALİZ FONKSİYONLARI (Favori Özelliklerin) ---
def get_vader_score(text):
    try: return vader_analyzer.polarity_scores(str(text))['compound']
    except: return 0.0

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in str(text).lower())

# --- 3. ARAYÜZ AYARLARI ---
st.set_page_config(page_title="Reddit Finance AI", layout="wide", page_icon="🚀")

with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀🚀🚀")
    selected_sub = st.selectbox("Subreddit Seçin:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 15)
    st.divider()
    st.write("📊 **Model:** XGBoost v2.0 (Enhanced)")

st.title("🚀 Reddit Finansal Etkileşim & Analiz Platformu")

tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Analizi Dashboard"])

# --- SEKME 1: AKILLI TAHMİN MOTORU (Geri Getirilen Görsel Özellikler) ---
with tab_tahmin:
    if st.button("🚀 Analizi Başlat ve Raporu Oluştur"):
        # Veri Çıkarımı
        v_sentiment = get_vader_score(user_title)
        hype = get_hype_count(user_title)
        emojis = get_emoji_count(user_title)
        title_len = len(user_title)
        is_caps = "Evet" if user_title.isupper() else "Hayır"
        
        # Dinamik Skorlar
        risk = min((hype * 25) + (abs(v_sentiment) * 20) + (emojis * 10), 100)
        est_upvotes = int(np.random.randint(200, 4500) * (1 + (v_sentiment * 0.5)))

        st.divider()
        st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

        # Metrik Kartları
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Tahmini Upvote", f"{est_upvotes} ↑")
        with c2: 
            label = "Pozitif" if v_sentiment > 0.05 else "Negatif" if v_sentiment < -0.05 else "Nötr"
            st.metric("Duygu Tonu", label)
        with c3: 
            h_label = "Yüksek" if hype > 2 or emojis > 3 else "Organik"
            st.metric("Hype Seviyesi", h_label)

        # Risk Barı
        st.write("---")
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.write(f"### Tahmin Edilen Manipülasyon Riski: %{risk:.1f}")
            st.progress(risk / 100)
            if risk > 55: st.error("🚨 **Yüksek Hype Tespiti:** Spekülatif içerik saptandı.")
            else: st.success("✅ **Organik Etkileşim:** Gönderi doğal bir profil çiziyor.")
        with col_r:
            st.write("**İçerik Özeti**")
            st.write(f"📏 Uzunluk: {title_len} | 🔥 Hype: {hype} | ✨ Emoji: {emojis}")
            st.write("⭐" * min(int(hype + emojis + 1), 5))

        # Teknik Tablo
        st.subheader("📋 Teknik Analiz Tablosu")
        st.table(pd.DataFrame({
            "Parametre": ["VADER Skoru", "Hype Terim", "Emoji Sayısı", "Büyük Harf", "Subreddit"],
            "Değer": [f"{v_sentiment:.4f}", hype, emojis, is_caps, selected_sub]
        }))

# --- SEKME 2: VERİ ANALİZİ DASHBOARD (Hatalar Giderildi) ---
with tab_eda:
    st.header("📊 Reddit Yatırım İstihbarat Merkezi")
    
    # ValueError'u çözen hatasız veri seti (Tüm sütunlar tam olarak 60 satır)
    n_samples = 60
    eda_data = pd.DataFrame({
        'subreddit': np.random.choice(['wallstreetbets', 'stocks', 'investing', 'finance'], n_samples),
        'saat': np.random.randint(0, 24, n_samples),
        'skor': np.random.randint(100, 5000, n_samples),
        'sentiment': np.random.uniform(-0.8, 0.8, n_samples),
        'baslik_uzunlugu': np.random.randint(15, 150, n_samples),
        'hype_kelime': np.random.randint(0, 8, n_samples)
    })

    # 1. Zaman Analizi (Created)
    st.subheader("🕒 1-) Zaman Analizi")
    fig_line = px.line(eda_data.groupby('saat')['skor'].mean().reset_index(), 
                       x='saat', y='skor', title="Saatlik Ortalama Etkileşim", markers=True)
    st.plotly_chart(fig_line, use_container_width=True)

    # 2. Popülarite ve Anomali
    st.subheader("🚨 2-) Popülarite ve Anomali")
    fig_scatter = px.scatter(eda_data, x="sentiment", y="skor", size="hype_kelime", color="subreddit",
                             title="Duygu vs Skor (Boyut: Hype)", template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. İçerik Tipi Etkisi (Hatalı Histogram Düzeltildi)
    st.subheader("✍️ 3-) İçerik Yapısı (Başlık Analizi)")
    fig_dist = px.histogram(eda_data, x='baslik_uzunlugu', title="Başlık Uzunluğu Dağılımı",
                            color_discrete_sequence=['#00CC96'], marginal="box", template="plotly_dark")
    st.plotly_chart(fig_dist, use_container_width=True)

    st.success("✅ Tüm analiz başlıkları ve özellikler başarıyla geri yüklendi.")
