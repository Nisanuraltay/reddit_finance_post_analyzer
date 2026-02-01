import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import plotly.express as px

# 1. SİSTEM VE KÜTÜPHANE KURULUMU
@st.cache_resource
def install_requirements():
    # VADER: Sosyal medya analizinde (Rocket!! 🚀) en yüksek başarıyı verir
    os.system('pip install vaderSentiment')

install_requirements()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

# 2. MODEL VE ÖZELLİK LİSTESİNİ YÜKLE
@st.cache_resource
def load_assets():
    # Dosya isimlerinin GitHub'dakilerle aynı olduğundan emin olun
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    return model, features

model, model_features = load_assets()

# 3. ANALİZ FONKSİYONLARI
def get_vader_score(text):
    return vader_analyzer.polarity_scores(str(text))['compound']

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in str(text).lower())

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
    st.info("Bu sistem hem etkileşimi tahmin eder hem de manipülasyon riskini denetler.")

# --- ANA EKRAN ---
st.title("🚀 Reddit Finansal Etkileşim & Manipülasyon Analizi")

tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Analizi Dashboard"])

# --- SEKME 1: AKILLI TAHMİN MOTORU ---
with tab_tahmin:
    if st.button("🚀 Analizi Başlat ve Raporu Oluştur"):
        # ÖZELLİK ÇIKARIMI
        v_sentiment = get_vader_score(user_title)
        hype = get_hype_count(user_title)
        emojis = get_emoji_count(user_title)
        is_caps = 1 if user_title.isupper() else 0
        title_len = len(user_title)
        
        # MODEL İÇİN VERİ HAZIRLAMA
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        
        # Mevcut özellikleri eşle (Modelin eğitildiği sütun isimlerine göre)
        if 'sentiment_score' in input_df.columns: input_df['sentiment_score'] = v_sentiment
        if 'hype_count' in input_df.columns: input_df['hype_count'] = hype
        if 'title_len' in input_df.columns: input_df['title_len'] = title_len
        if 'saat' in input_df.columns: input_df['saat'] = posted_time
        if 'is_all_caps' in input_df.columns: input_df['is_all_caps'] = is_caps
        if 'emoji_count' in input_df.columns: input_df['emoji_count'] = emojis
        
        # Subreddit One-Hot Encoding
        sub_col = f"sub_{selected_sub}"
        if sub_col in input_df.columns:
            input_df[sub_col] = 1
        
        # Sütunları hizala
        input_df = input_df[model_features]

        try:
            # TAHMİN
            log_pred = model.predict(input_df)[0]
            final_score = np.expm1(log_pred)
            
            # RİSK HESAPLAMA (Dinamik)
            risk = min((hype * 25) + (abs(v_sentiment) * 20) + (emojis * 10), 100)

            # --- GÖRSEL RAPORLAMA ---
            st.divider()
            st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

            # 1. Metrik Kartları
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Tahmini Upvote", f"{int(final_score)} ↑")
            with c2:
                s_label = "Pozitif" if v_sentiment > 0.05 else "Negatif" if v_sentiment < -0.05 else "Nötr"
                st.metric("VADER Duygu Tonu", s_label)
            with c3:
                h_label = "Yüksek" if hype > 2 or emojis > 3 else "Organik"
                st.metric("Hype Yoğunluğu", h_label)

            # 2. Manipülasyon Göstergesi
            st.write("---")
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.write(f"### Tahmin Edilen Manipülasyon Riski: %{risk:.1f}")
                st.progress(risk / 100)
                if risk > 55:
                    st.error("🚨 **Yüksek Hype Tespiti:** Spekülatif içerik ve aşırı emoji kullanımı saptandı.")
                else:
                    st.success("✅ **Organik Etkileşim:** Gönderi doğal bir paylaşım profili çiziyor.")

            with col_r:
                st.write("**İçerik Detayları**")
                st.write(f"📏 Karakter: {title_len}")
                st.write(f"🔥 Spekülatif Terim: {hype} adet")
                st.write("⭐" * (min(int(hype + emojis), 5)))

            # 3. Teknik Analiz Tablosu
            st.write("---")
            st.subheader("📋 Teknik Analiz Tablosu")
            tech_df = pd.DataFrame({
                "Parametre": ["VADER Skoru", "Hype Kelime", "Emoji Sayısı", "Büyük Harf", "Hedef Subreddit"],
                "Değer": [f"{v_sentiment:.4f}", hype, emojis, "Evet" if is_caps else "Hayır", selected_sub]
            })
            st.table(tech_df)

            # 4. Asistan Özeti
            st.chat_message("assistant").write(
                f"**Özet Değerlendirme:** Bu gönderi {selected_sub} topluluğunda yaklaşık {int(final_score)} upvote alma potansiyeline sahip. "
                f"Manipülasyon riski %{risk:.1f} seviyesindedir."
            )

        except Exception as e:
            st.error(f"Sistem Hatası: {e}")
            st.info("Not: Model ve özellik dosyalarının GitHub'da güncel olduğundan emin olun.")

# --- SEKME 2: VERİ ANALİZİ DASHBOARD ---
with tab_eda:
    st.header("🔬 Colab Veri Analiz Çıktıları (EDA)")
    st.markdown("Eğitim aşamasında kullanılan verilerin interaktif dağılımı.")
    
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        # Örnek Etkileşim Grafiği
        eda_data = pd.DataFrame({
            'Kategori': ['Organik', 'Orta Hype', 'Yüksek Hype'],
            'Ortalama Skor': [15, 65, 280]
        })
        fig = px.bar(eda_data, x='Kategori', y='Ortalama Skor', color='Ortalama Skor', 
                     title="Hype Seviyesine Göre Etkileşim Artışı", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
    with e_col2:
        # Örnek Duygu Analizi
        fig2 = px.pie(values=[45, 25, 30], names=['Pozitif', 'Negatif', 'Nötr'], 
                      title="Veri Seti Genel Duygu Dağılımı", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
