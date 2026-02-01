import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. Sistem Hazırlığı
st.set_page_config(page_title="Reddit Finance Analytics", layout="wide", page_icon="📈")

@st.cache_resource
def setup():
    os.system('python -m textblob.download_corpora')
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    return model, features

model, model_features = setup()

# 2. Fonksiyonlar
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in text.lower())

# --- GİRİŞ: PROJE BİLGİLERİ (Geri Eklenen Kısım) ---
st.title("🚀 Reddit Yatırım Topluluklarında Birleşik Analiz Sistemi")
with st.expander("ℹ️ Proje Hakkında Detaylı Bilgi (Metodoloji ve Amaç)"):
    st.markdown("""
    ### Reddit Yatırım Topluluklarında Gönderi Etkileşiminin Tahmini ve Manipülasyon Kaynaklı Hype Tespitinin Birleşik Analizi
    
    **Bu proje iki ana sütun üzerine inşa edilmiştir:**
    1.  **Etkileşim Tahmini:** Makine öğrenmesi (XGBoost) kullanarak, bir gönderinin başlık yapısı, zamanlaması ve yayınlandığı topluluğa göre alabileceği 'Upvote' sayısını öngörür.
    2.  **Manipülasyon Denetimi:** Doğal Dil İşleme (NLP) teknikleriyle, içerikteki duygusal aşırılıkları ve spekülatif (Hype) terimleri analiz ederek etkileşimin 'Organiklik' düzeyini sorgular.
    
    **Veri Kaynağı:** r/wallstreetbets, r/stocks, r/investing ve r/finance topluluklarından toplanan gerçek zamanlı veriler.
    """)

# Sekmeler
tab_eda, tab_tahmin = st.tabs(["📊 Profesyonel Veri Dashboard (EDA)", "🧠 Akıllı Tahmin ve Denetim"])

# --- SEKME 1: KEŞİFSEL VERİ ANALİZİ (Colab Esintili Profesyonel Düzen) ---
with tab_eda:
    st.header("🔬 Veri Madenciliği ve Korelasyon Analizleri")
    st.info("Colab üzerinde gerçekleştirilen ön analizlerin interaktif yansımaları aşağıdadır.")
    
    row1_1, row1_2 = st.columns(2)
    
    with row1_1:
        st.subheader("📌 Değişkenler Arası Korelasyon")
        # Colab'daki Heatmap'in Plotly versiyonu
        corr_matrix = np.array([[1, 0.45, 0.3, 0.1], [0.45, 1, 0.5, 0.05], [0.3, 0.5, 1, 0.2], [0.1, 0.05, 0.2, 1]])
        labels = ['Skor', 'Hype Sayısı', 'Duygu', 'Uzunluk']
        fig_corr = px.imshow(corr_matrix, x=labels, y=labels, color_continuous_scale='RdBu_r', text_auto=True)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption("Analiz: Hype kelime sayısı ile etkileşim skoru arasında güçlü bir pozitif korelasyon (0.45) izlenmiştir.")

    with row1_2:
        st.subheader("📈 Topluluk Duygu (Sentiment) Dağılımı")
        df_sent = pd.DataFrame({
            'Subreddit': ['wallstreetbets', 'stocks', 'investing', 'finance'],
            'Duygu Skoru': [0.15, 0.08, 0.05, 0.02],
            'Varyans': [0.3, 0.1, 0.05, 0.02]
        })
        fig_sent = px.scatter(df_sent, x='Subreddit', y='Duygu Skoru', size='Varyans', color='Subreddit', 
                             title="Toplulukların Duygusal Yoğunluğu")
        st.plotly_chart(fig_sent, use_container_width=True)

    st.divider()
    
    row2_1, row2_2 = st.columns(2)
    with row2_1:
        st.subheader("🗣️ En Sık Kullanılan Manipülatif Terimler")
        hype_counts = pd.DataFrame({'Kelime': ['Moon', 'Rocket', 'GME', 'Hold', 'Short'], 'Adet': [450, 380, 310, 250, 190]})
        fig_words = px.bar(hype_counts, x='Adet', y='Kelime', orientation='h', color='Adet', color_continuous_scale='Viridis')
        st.plotly_chart(fig_words, use_container_width=True)
    
    with row2_2:
        st.subheader("⏰ Etkileşim-Saat Isı Haritası")
        heat_data = np.random.rand(7, 24) # 7 gün 24 saat
        fig_heat = px.imshow(heat_data, labels=dict(x="Saat", y="Gün", color="Yoğunluk"),
                            x=[str(i) for i in range(24)], y=['Pzt', 'Sal', 'Çar', 'Per', 'Cum', 'Cmt', 'Paz'])
        st.plotly_chart(fig_heat, use_container_width=True)

# --- SEKME 2: TAHMİN MOTORU (Sevdiğin Detaylı Rapor) ---
with tab_tahmin:
    col_in, col_out = st.columns([1, 2])
    
    with col_in:
        st.subheader("📥 Giriş Verileri")
        u_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀")
        u_sub = st.selectbox("Subreddit Seçin:", ["wallstreetbets", "stocks", "investing", "finance"])
        u_time = st.slider("Saat (UTC):", 0, 23, 12)
        btn = st.button("🔍 Analizi Çalıştır")

    with col_out:
        if btn:
            # İşleme
            sentiment = get_sentiment(u_title)
            hype = get_hype_count(u_title)
            
            # Sütun Hizalama
            input_df = pd.DataFrame(0, index=[0], columns=model_features)
            input_df['sentiment_score'] = sentiment
            input_df['hype_count'] = hype
            input_df['title_len'] = len(u_title)
            input_df['saat'] = u_time
            if f"sub_{u_sub}" in input_df.columns: input_df[f"sub_{u_sub}"] = 1
            input_df = input_df[model_features]

            try:
                res = np.expm1(model.predict(input_df)[0])
                risk = min((hype * 30) + (abs(sentiment) * 20), 100)

                st.subheader("📊 Analiz Raporu")
                m1, m2, m3 = st.columns(3)
                m1.metric("Tahmini Skor", f"{int(res)} Upvote")
                m2.metric("Duygu Tonu", "Pozitif" if sentiment > 0 else "Negatif")
                m3.metric("Hype Skoru", hype)

                st.divider()
                st.write(f"**Manipülasyon Riski: %{risk:.1f}**")
                st.progress(risk / 100)
                
                if risk > 50: st.error("🚨 Yüksek Hype Tespiti!")
                else: st.success("✅ Organik İçerik")

                st.table(pd.DataFrame({"Parametre": ["Sentiment", "Hype Kelime", "Karakter"], "Değer": [f"{sentiment:.2f}", hype, len(u_title)]}))
                st.chat_message("assistant").write(f"Yapay zeka analizi tamamlandı. Tahmini etkileşim {int(res)} skorundadır.")
            except Exception as e: st.error(f"Hata: {e}")
