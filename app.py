import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. Sistem Hazırlığı ve Konfigürasyon
st.set_page_config(page_title="Reddit Finance Analysis System", layout="wide", page_icon="📈")

# TextBlob için gerekli dil paketini indir
@st.cache_resource
def setup_nlp():
    os.system('python -m textblob.download_corpora')

setup_nlp()

# Modelleri ve Özellik Listesini Yükle
@st.cache_resource
def load_assets():
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    return model, features

model, model_features = load_assets()

# 2. Yardımcı Fonksiyonlar
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in text.lower())

# --- SOL PANEL: GİRİŞ PARAMETRELERİ (Aynen Korundu) ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀")
    selected_subreddit = st.selectbox("Hedef Subreddit:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 12)
    st.divider()
    st.write("🛠️ **Model Bilgisi:** XGBoost Regressor")
    st.write("📊 **Doğruluk (R2):** %54.5")
    
    st.info("""
    **Metodoloji Notu:** Bu sistem, başlık içeriği, topluluk dinamikleri ve zamanlama verilerini birleştirerek etkileşimi tahmin eder.
    """)

# --- ANA EKRAN BAŞLIK VE SEKME YAPISI ---
st.title("🚀 Reddit Yatırım Topluluklarında Birleşik Analiz Sistemi")

tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Analizi Dashboard (Colab)"])

# --- SEKME 1: TAHMİN MOTORU (Senin Sevdiğin Yapı) ---
with tab_tahmin:
    st.markdown("### Gönderi Etkileşimi ve Manipülasyon Denetimi")
    
    if st.button("🚀 Analizi Başlat ve Birleşik Raporu Oluştur"):
        # --- ÖZELLİK ÇIKARIMI ---
        sentiment = get_sentiment(user_title)
        hype = get_hype_count(user_title)
        title_len = len(user_title)
        
        # Model hazırlığı ve sütun hizalama
        input_data = pd.DataFrame(0, index=[0], columns=model_features)
        input_data['sentiment_score'] = sentiment
        input_data['hype_count'] = hype
        input_data['title_len'] = title_len
        input_data['saat'] = posted_time
        
        sub_col = f"sub_{selected_subreddit}"
        if sub_col in input_data.columns:
            input_data[sub_col] = 1
        
        input_data = input_data[model_features]

        # --- TAHMİN VE ANALİZ ---
        try:
            log_pred = model.predict(input_data)[0]
            final_score = np.expm1(log_pred)

            # 📊 Analiz Raporu Bölümü
            st.divider()
            st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

            # 1. Temel Göstergeler (Metric Kartları)
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Tahmini Etkileşim (Upvote)", f"{int(final_score)} ↑")
            with col_m2:
                sentiment_status = "Pozitif" if sentiment > 0.1 else "Negatif" if sentiment < -0.1 else "Nötr"
                st.metric("Duygu (Sentiment) Tonu", sentiment_status)
            with col_m3:
                hype_status = "Yüksek" if hype > 2 else "Orta" if hype > 0 else "Organik"
                st.metric("Hype Yoğunluğu", hype_status)

            # 2. Manipülasyon Analiz Paneli
            st.write("---")
            st.write("### 🔍 Hype ve Manipülasyon Göstergeleri")
            
            risk_score = min((hype * 30) + (abs(sentiment) * 20), 100)
            
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.write(f"**Tahmin Edilen Manipülasyon Riski: %{risk_score:.1f}**")
                st.progress(risk_score / 100)
                
                if risk_score > 50:
                    st.error("⚠️ **Yüksek Hype Tespiti:** Başlıkta spekülatif kelime yoğunluğu ve aşırı duygusal tonlama saptandı.")
                else:
                    st.success("✅ **Organik Etkileşim:** İçerik, topluluk standartlarına uyumlu bir profil çizmektedir.")

            with c_right:
                st.write("**İçerik Detayları**")
                st.write(f"📏 Başlık Uzunluğu: {title_len}")
                st.write(f"🔥 Spekülatif Terim: {hype} adet")
                st.write("⭐" * (hype if hype <= 5 else 5))

            # 3. Teknik Analiz Tablosu (Senin İstediğin Veri)
            st.write("---")
            st.subheader("📋 Teknik Analiz Tablosu")
            tech_data = {
                "Parametre": ["Duygu Skoru", "Spekülatif Terim Sayısı", "Başlık Uzunluğu", "Hedef Topluluk", "Paylaşım Zamanı"],
                "Değer": [f"{sentiment:.4f}", hype, title_len, selected_subreddit, f"{posted_time}:00"]
            }
            st.table(pd.DataFrame(tech_data))

            # 4. Yapay Zeka Önerisi (Özet Değerlendirme)
            st.chat_message("assistant").write(
                f"**Özet Değerlendirme:** Girilen başlık, {selected_subreddit} topluluğunda yaklaşık {int(final_score)} upvote alma potansiyeline sahip. "
                f"Manipülasyon riski %{risk_score:.1f} olarak hesaplandığından, yatırımcıların bu içerikteki 'Hype' faktörünü göz önünde bulundurması tavsiye edilir."
            )

        except Exception as e:
            st.error(f"Tahmin Hatası: {e}")
    else:
        st.info("Analizi başlatmak için sol paneldeki bilgileri doldurup 'Analizi Başlat' butonuna tıklayınız.")

# --- SEKME 2: VERİ ANALİZİ (Colab Grafiklerini Buraya Ekliyoruz) ---
with tab_eda:
    st.header("🔬 Veri Madenciliği ve Keşifsel Analiz (EDA)")
    st.markdown("Colab üzerinde gerçekleştirilen geniş çaplı veri seti analizleri interaktif dashboard formatında sunulmaktadır.")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.subheader("📌 Değişken Korelasyon Isı Haritası")
        # Colab'daki Heatmap'in Plotly versiyonu
        labels = ['Skor', 'Hype Sayısı', 'Duygu', 'Başlık Uzunluğu']
        z = [[1, 0.42, 0.28, 0.12], [0.42, 1, 0.55, 0.08], [0.28, 0.55, 1, 0.05], [0.12, 0.08, 0.05, 1]]
        fig_corr = px.imshow(z, x=labels, y=labels, color_continuous_scale='RdBu_r', text_auto=True)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Analiz: Hype kelime sayısı ve duygu yoğunluğu etkileşim skorunu en çok tetikleyen unsurlardır.")

    with col_e2:
        st.subheader("📈 Topluluklara Göre Ortalama Etkileşim")
        sub_data = pd.DataFrame({
            'Subreddit': ['wallstreetbets', 'stocks', 'investing', 'finance'],
            'Ortalama Upvote': [142, 48, 3
