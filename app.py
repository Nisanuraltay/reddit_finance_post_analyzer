import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import os

# TextBlob için gerekli dil paketini indir (Streamlit Cloud için şart)
os.system('python -m textblob.download_corpora')

# 1. Modeli ve Özellik Listesini Yükle
# Dosya isimlerinin GitHub'dakilerle birebir aynı olduğundan emin olun.
model = joblib.load('final_reddit_model.pkl') 
model_features = joblib.load('final_features.pkl')

# 2. Yardımcı Fonksiyonlar
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in text.lower())

# 3. Arayüz Tasarımı (Geniş Yerleşim)
st.set_page_config(page_title="Reddit Finance Analyzer", page_icon="📈", layout="wide")

st.title("📈 Reddit Yatırım Topluluklarında Gönderi Analiz Sistemi")
st.markdown("""
**Proje Kapsamı:** Bu çalışma, finans paylaşımlarını analiz ederek **Etkileşim Tahmini** yapar ve 
içeriğin **Organik mi yoksa Hype/Manipülasyon kaynaklı mı** olduğunu birleşik bir yapıda denetler.
""")

# Yan Panel: Kullanıcı Girişleri
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀")
    selected_subreddit = st.selectbox("Hedef Subreddit:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 12)
    st.divider()
    st.write("🛠️ **Model Bilgisi:** XGBoost Regressor")
    st.write("📊 **Doğruluk (R2):** %54.5")

# Ana Ekran Analiz Bölümü
if st.button("🚀 Analizi Başlat ve Birleşik Raporu Oluştur"):
    # --- ÖZELLİK ÇIKARIMI ---
    sentiment = get_sentiment(user_title)
    hype = get_hype_count(user_title)
    title_len = len(user_title)
    
    # Model hazırlığı ve sütun hizalama (Hata almamak için kritik)
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    input_data['sentiment_score'] = sentiment
    input_data['hype_count'] = hype
    input_data['title_len'] = title_len
    input_data['saat'] = posted_time
    
    sub_col = f"sub_{selected_subreddit}"
    if sub_col in input_data.columns:
        input_data[sub_col] = 1
    
    # Sütunları modelin beklediği sıraya sok
    input_data = input_data[model_features]

    # --- TAHMİN VE DENETİM ---
    try:
        log_pred = model.predict(input_data)[0]
        final_score = np.expm1(log_pred)

        # Raporlama Alanı
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
        
        # Risk Skoru (Hype ve Sentiment üzerinden ağırlıklı hesap)
        risk_score = (hype * 30) + (abs(sentiment) * 20)
        risk_score = min(risk_score, 100)
        
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.write(f"**Tahmin Edilen Manipülasyon Riski: %{risk_score:.1f}**")
            st.progress(risk_score / 100)
            
            if risk_score > 50:
                st.error("⚠️ **Yüksek Hype Tespiti:** Başlıkta spekülatif kelime yoğunluğu ve aşırı duygusal tonlama saptandı. Etkileşimin yapay olma olasılığı yüksektir.")
            else:
                st.success("✅ **Organik Etkileşim:** İçerik, topluluk standartlarına uyumlu ve doğal bir bilgi paylaşımı profili çizmektedir.")

        with c_right:
            st.write("**İçerik Detayları**")
            st.write(f"📏 Başlık Uzunluğu: {title_len}")
            st.write(f"🔥 Spekülatif Terim: {hype} adet")
            st.write("⭐" * (hype if hype <= 5 else 5))

        # 3. Teknik Veri Özeti (Tablo)
        st.write("---")
        st.write("### 📋 Teknik Analiz Tablosu")
        tech_data = {
            "Parametre": ["Duygu Skoru", "Spekülatif Terim Sayısı", "Başlık Uzunluğu", "Hedef Topluluk", "Paylaşım Zamanı"],
            "Değer": [f"{sentiment:.4f}", hype, title_len, selected_subreddit, f"{posted_time}:00"]
        }
        st.table(pd.DataFrame(tech_data))

        # 4. Yapay Zeka Önerisi
        st.chat_message("assistant").write(
            f"**Özet Değerlendirme:** Girilen başlık, {selected_subreddit} topluluğunda yaklaşık {int(final_score)} upvote alma potansiyeline sahip. "
            f"Manipülasyon riski %{risk_score:.1f} olarak hesaplandığından, yatırımcıların bu içerikteki 'Hype' faktörünü göz önünde bulundurması tavsiye edilir."
        )

    except Exception as e:
        st.error(f"Sistem Hatası Oluştu: {e}")

else:
    st.info("Analizi başlatmak için sol paneldeki bilgileri doldurup 'Analizi Başlat' butonuna tıklayınız.")
