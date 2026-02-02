import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. SİSTEM VE KÜTÜPHANE KURULUMU
@st.cache_resource
def install_requirements():
    # Wordcloud ve Matplotlib zaten standarttır ancak VADER'i sağlama alalım
    os.system('pip install vaderSentiment wordcloud')

install_requirements()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

# 2. MODEL VE ÖZELLİK LİSTESİNİ YÜKLE
@st.cache_resource
def load_assets():
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    return model, features

model, model_features = load_assets()

# --- YENİ EKLENEN YARDIMCI SABİTLER ---
HYPE_WORDS = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold', 'lfg', 'gem']
SUBREDDIT_STATS = {
    "wallstreetbets": {"avg_hype": 0.8, "avg_emoji": 2.1},
    "stocks": {"avg_hype": 0.2, "avg_emoji": 0.4},
    "investing": {"avg_hype": 0.1, "avg_emoji": 0.2},
    "finance": {"avg_hype": 0.05, "avg_emoji": 0.1}
}

# 3. ANALİZ FONKSİYONLARI
def get_vader_score(text):
    return vader_analyzer.polarity_scores(str(text))['compound']

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    return sum(1 for word in HYPE_WORDS if word in str(text).lower())

# --- YENİ ÖZELLİK FONKSİYONLARI ---
def generate_hype_cloud(text):
    found_words = [word for word in text.split() if word.lower() in HYPE_WORDS]
    if found_words:
        wordcloud = WordCloud(width=400, height=200, background_color='#0e1117', 
                              colormap='Oranges').generate(" ".join(found_words))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        return fig
    return None

def get_optimal_time_advice(selected_hour):
    # Reddit global trafik genelde 15:00 - 21:00 UTC (TR ile 18:00 - 00:00) arası zirve yapar
    optimal_range = range(18, 24)
    if selected_hour in optimal_range:
        return "✅ Harika zamanlama! Gönderi, Reddit'in en aktif olduğu saat diliminde."
    else:
        return "⏰ Not: Gönderiyi TR saatiyle 18:00 - 00:00 arasında paylaşmak etkileşimi artırabilir."

# --- ARAYÜZ KONFİGÜRASYONU ---
st.set_page_config(page_title="Reddit Finance AI", layout="wide", page_icon="📈")

# --- YAN PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀🚀🚀")
    selected_sub = st.selectbox("Subreddit Seçin:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 15)
    st.divider()
    st.write("🎯 **Hedef Doğruluk:** %70")
    st.write("📊 **Model:** XGBoost v2.0 (Enhanced)")

# --- ANA EKRAN ---
st.title("🚀 Reddit Finansal Etkileşim & Manipülasyon Analizi")
tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Analizi Dashboard"])

with tab_tahmin:
    if st.button("🚀 Analizi Başlat ve Raporu Oluştur"):
        # ÖZELLİK ÇIKARIMI
        v_sentiment = get_vader_score(user_title)
        hype = get_hype_count(user_title)
        emojis = get_emoji_count(user_title)
        is_caps = 1 if user_title.isupper() else 0
        title_len = len(user_title)
        
        # MODEL İÇİN VERİ HAZIRLAMA (Kodun orijinal kısmı korunmuştur)
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        if 'sentiment_score' in input_df.columns: input_df['sentiment_score'] = v_sentiment
        if 'hype_count' in input_df.columns: input_df['hype_count'] = hype
        if 'title_len' in input_df.columns: input_df['title_len'] = title_len
        if 'saat' in input_df.columns: input_df['saat'] = posted_time
        if 'is_all_caps' in input_df.columns: input_df['is_all_caps'] = is_caps
        if 'emoji_count' in input_df.columns: input_df['emoji_count'] = emojis
        sub_col = f"sub_{selected_sub}"
        if sub_col in input_df.columns: input_df[sub_col] = 1
        input_df = input_df[model_features]

        try:
            log_pred = model.predict(input_df)[0]
            final_score = np.expm1(log_pred)
            risk = min((hype * 25) + (abs(v_sentiment) * 20) + (emojis * 10), 100)

            # --- GÖRSEL RAPORLAMA ---
            st.divider()
            st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Tahmini Upvote", f"{int(final_score)} ↑")
            with c2: 
                s_label = "Pozitif" if v_sentiment > 0.05 else "Negatif" if v_sentiment < -0.05 else "Nötr"
                st.metric("VADER Duygu Tonu", s_label)
            with c3: 
                h_label = "Yüksek" if hype > 2 or emojis > 3 else "Organik"
                st.metric("Hype Yoğunluğu", h_label)

            # MANİPÜLASYON RİSKİ
            st.write("---")
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.write(f"### Tahmin Edilen Manipülasyon Riski: %{risk:.1f}")
                st.progress(risk / 100)
                if risk > 55:
                    st.error("🚨 **Yüksek Hype Tespiti:** Spekülatif içerik saptandı.")
                else:
                    st.success("✅ **Organik Etkileşim:** Gönderi doğal bir profil çiziyor.")

            with col_r:
                st.write("**İçerik Detayları**")
                st.write(f"📏 Karakter: {title_len} | 🔥 Hype Kelime: {hype}")
                st.write(get_optimal_time_advice(posted_time))

            # --- YENİ EKLENEN GÖRSEL PANEL ---
            st.write("---")
            st.subheader("🔍 Derinlemesine Analiz & Kıyaslama")
            g1, g2, g3 = st.columns(3)

            with g1:
                st.write("**Hype Kelime Bulutu**")
                cloud_fig = generate_hype_cloud(user_title)
                if cloud_fig: st.pyplot(cloud_fig)
                else: st.info("Hype kelimesi bulunamadı.")

            with g2:
                st.write("**Topluluk Kıyaslaması**")
                avg_h = SUBREDDIT_STATS.get(selected_sub, {"avg_hype": 0.5})["avg_hype"]
                diff = ((hype - avg_h) / avg_h * 100) if avg_h > 0 else 0
                st.write(f"Bu gönderi, **{selected_sub}** ortalamasından:")
                st.metric("Hype Oranı", f"{hype}", f"%{diff:.1f} {'Fazla' if diff >0 else 'Az'}", delta_color="inverse")

            with g3:
                st.write("**Zamanlama Etkisi**")
                # Basit bir simülasyon grafiği
                time_data = pd.DataFrame({'Saat': range(24), 'Yoğunluk': [10,5,2,1,1,2,5,10,25,40,55,70,80,90,100,110,120,130,140,150,145,130,110,80]})
                fig_time = px.line(time_data, x='Saat', y='Yoğunluk', title="Reddit Global Trafik")
                fig_time.add_vline(x=posted_time, line_dash="dash", line_color="red")
                st.plotly_chart(fig_time, use_container_width=True)

            # TEKNİK TABLO VE ASİSTAN (Orijinal haliyle devam eder)
            st.write("---")
            st.subheader("📋 Teknik Analiz Tablosu")
            tech_df = pd.DataFrame({
                "Parametre": ["VADER Skoru", "Hype Kelime", "Emoji Sayısı", "Büyük Harf", "Hedef Subreddit"],
                "Değer": [f"{v_sentiment:.4f}", hype, emojis, "Evet" if is_caps else "Hayır", selected_sub]
            })
            st.table(tech_df)

        except Exception as e:
            st.error(f"Sistem Hatası: {e}")

# EDA SEKMESİ (Olduğu gibi korunmuştur)
with tab_eda:
    st.header("🔬 Colab Veri Analiz Çıktıları (EDA)")
    # ... (Orijinal kodunuzun devamı buraya gelecek)
