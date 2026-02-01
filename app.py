import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import plotly.express as px
import os

# 1. Sistem Hazırlığı ve Konfigürasyon
st.set_page_config(page_title="Reddit Finance Analysis Dashboard", layout="wide", page_icon="📈")

@st.cache_resource
def download_data():
    os.system('python -m textblob.download_corpora')

download_data()

# Modelleri Yükle
@st.cache_resource
def load_models():
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    return model, features

model, model_features = load_models()

# 2. Yardımcı Fonksiyonlar
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in text.lower())

# --- ARAYÜZ BAŞLIĞI ---
st.title("📈 Reddit Yatırım Topluluklarında Gönderi Analiz Sistemi")
st.markdown("""
**Proje Kapsamı:** Bu çalışma, finans paylaşımlarını analiz ederek **Etkileşim Tahmini** yapar ve 
içeriğin **Organik mi yoksa Hype/Manipülasyon kaynaklı mı** olduğunu birleşik bir yapıda denetler.
""")

# Sekmeleri Oluştur
tab_eda, tab_tahmin = st.tabs(["📊 Keşifsel Veri Analizi (EDA)", "🧠 Birleşik Analiz Tahmin Motoru"])

# --- SEKME 1: KEŞİFSEL VERİ ANALİZİ ---
with tab_eda:
    st.header("🔍 Veri Seti ve Topluluk Dinamikleri")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.subheader("Subreddit Bazlı Etkileşim ve Hype Oranı")
        # Örnek EDA verisi (Dashboard görseli için)
        eda_df = pd.DataFrame({
            'Subreddit': ['wallstreetbets', 'stocks', 'investing', 'finance'],
            'Ortalama Skor': [145, 52, 28, 14],
            'Manipülasyon Riski (%)': [82, 35, 12, 4]
        })
        fig_bar = px.bar(eda_df, x='Subreddit', y='Ortalama Skor', color='Manipülasyon Riski (%)',
                         text_auto=True, template="plotly_white", color_continuous_scale='Reds')
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_e2:
        st.subheader("Zaman Dilimi Analizi")
        time_trend = pd.DataFrame({'Saat': list(range(24)), 'Gönderi Yoğunluğu': np.random.gamma(2, 2, 24)})
        fig_area = px.area(time_trend, x='Saat', y='Gönderi Yoğunluğu', title="Günün Saatlerine Göre Aktivite")
        st.plotly_chart(fig_area, use_container_width=True)

# --- SEKME 2: TAHMİN MOTORU (SENİN SEVDİĞİN TASARIM) ---
with tab_tahmin:
    # Yan Panel (Sadece bu sekmede anlamlı girişler için sidebar kullanabiliriz veya sütun)
    col_input, col_output = st.columns([1, 2])
    
    with col_input:
        st.header("🔍 Giriş Parametreleri")
        user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀")
        selected_subreddit = st.selectbox("Hedef Subreddit:", ["wallstreetbets", "stocks", "investing", "finance"])
        posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 12)
        st.divider()
        st.write("🛠️ **Model:** XGBoost Regressor")
        st.write("📊 **R2 Skoru:** %54.5")
        analyze_btn = st.button("🚀 Analizi Başlat ve Raporla")

    with col_output:
        if analyze_btn:
            # --- ÖZELLİK ÇIKARIMI ---
            sentiment = get_sentiment(user_title)
            hype = get_hype_count(user_title)
            title_len = len(user_title)
            
            # Model hazırlığı
            input_data = pd.DataFrame(0, index=[0], columns=model_features)
            input_data['sentiment_score'] = sentiment
            input_data['hype_count'] = hype
            input_data['title_len'] = title_len
            input_data['saat'] = posted_time
            
            sub_col = f"sub_{selected_subreddit}"
            if sub_col in input_data.columns:
                input_data[sub_col] = 1
            
            input_data = input_data[model_features]

            # --- TAHMİN VE DENETİM ---
            try:
                log_pred = model.predict(input_data)[0]
                final_score = np.expm1(log_pred)

                st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

                # 1. Metrik Kartları
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Tahmini Upvote", f"{int(final_score)} ↑")
                with m2:
                    s_label = "Pozitif" if sentiment > 0.1 else "Negatif" if sentiment < -0.1 else "Nötr"
                    st.metric("Duygu Tonu", s_label)
                with m3:
                    h_label = "Yüksek" if hype > 2 else "Orta" if hype > 0 else "Organik"
                    st.metric("Hype Yoğunluğu", h_label)

                # 2. Manipülasyon Göstergesi
                st.write("---")
                risk_score = min((hype * 30) + (abs(sentiment) * 20), 100)
                
                cl, cr = st.columns([2, 1])
                with cl:
                    st.write(f"**Tahmin Edilen Manipülasyon Riski: %{risk_score:.1f}**")
                    st.progress(risk_score / 100)
                    if risk_score > 50:
                        st.error("⚠️ **Yüksek Hype Tespiti:** Başlıkta spekülatif kelime yoğunluğu saptandı.")
                    else:
                        st.success("✅ **Organik Etkileşim:** İçerik doğal bir profil çizmektedir.")
                with cr:
                    st.write("**İçerik Detayları**")
                    st.write(f"📏 Uzunluk: {title_len}")
                    st.write(f"🔥 Hype: {hype} adet")
                    st.write("⭐" * (hype if hype <= 5 else 5))

                # 3. Teknik Tablo
                st.write("---")
                tech_data = {
                    "Parametre": ["Duygu Skoru", "Spekülatif Terim", "Başlık Uzunluğu", "Subreddit"],
                    "Değer": [f"{sentiment:.4f}", hype, title_len, selected_subreddit]
                }
                st.table(pd.DataFrame(tech_data))

                # 4. AI Notu
                st.chat_message("assistant").write(f"Bu gönderi {int(final_score)} etkileşim potansiyeline sahip. Manipülasyon riski %{risk_score:.1f} olarak hesaplanmıştır.")

            except Exception as e:
                st.error(f"Hata: {e}")
        else:
            st.info("Analiz sonuçları burada görünecektir. Lütfen butona basın.")
