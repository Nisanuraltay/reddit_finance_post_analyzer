import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import plotly.express as px

# --- 1. SİSTEM HAZIRLIĞI VE NLP KURULUMU ---
st.set_page_config(page_title="Reddit Finance Analysis System", layout="wide", page_icon="📈")

@st.cache_resource
def setup_nlp_tools():
    # VADER kurulumu
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        os.system('pip install vaderSentiment')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

vader_analyzer = setup_nlp_tools()

# Modelleri ve Özellik Listesini Yükle
@st.cache_resource
def load_assets():
    # Dosya isimlerinin Colab'daki çıktı ile aynı olduğundan emin olun
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    return model, features

model, model_features = load_assets()

# --- 2. YARDIMCI ANALİZ FONKSİYONLARI ---
def get_vader_sentiment(text):
    return vader_analyzer.polarity_scores(text)['compound']

def get_emoji_count(text):
    # Kelime, boşluk ve temel noktalama dışındaki karakterleri sayar
    return len(re.findall(r'[^\w\s,.]', text))

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in text.lower())

# --- 3. SOL PANEL: GİRİŞ PARAMETRELERİ ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀🚀🚀")
    selected_subreddit = st.selectbox("Hedef Subreddit:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 12)
    st.divider()
    st.write("🛠️ **Model:** XGBoost Regressor (Enhanced)")
    st.info("Bu sürüm VADER Duygu Analizi ve Emoji Takibi özelliklerini içerir.")

# --- 4. ANA EKRAN VE SEKME YAPISI ---
st.title("🚀 Reddit Finansal Etkileşim & Tahmin Sistemi")
tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Analizi Dashboard"])

# --- SEKME 1: TAHMİN MOTORU ---
with tab_tahmin:
    st.markdown("### Gönderi Etkileşimi ve Manipülasyon Denetimi")
    
    if st.button("🚀 Analizi Başlat ve Raporu Oluştur"):
        # --- GELİŞTİRİLMİŞ ÖZELLİK ÇIKARIMI ---
        v_score = get_vader_sentiment(user_title)
        hype = get_hype_count(user_title)
        emojis = get_emoji_count(user_title)
        is_caps = 1 if user_title.isupper() else 0
        title_len = len(user_title)
        
        # Model Giriş Verisini Hazırlama (Sütun Eşleştirme)
        input_data = pd.DataFrame(0, index=[0], columns=model_features)
        
        # Sütun isimleri Colab'daki eğitimle birebir aynı olmalıdır!
        # Eğer Colab'da farklı isimler verdiysen burayı güncelle:
        input_data['sentiment_score'] = v_score
        input_data['hype_count'] = hype
        input_data['title_len'] = title_len
        input_data['saat'] = posted_time
        
        # Yeni eklenen özellikler (Model özellik listesinde varsa doldurulur)
        if 'emoji_count' in model_features: input_data['emoji_count'] = emojis
        if 'is_caps' in model_features: input_data['is_caps'] = is_caps
        
        # Subreddit One-Hot Encoding
        sub_col = f"sub_{selected_subreddit}"
        if sub_col in input_data.columns:
            input_data[sub_col] = 1
        
        # Özellikleri modelin beklediği sıraya diz
        input_data = input_data[model_features]

        # --- TAHMİN ---
        try:
            log_pred = model.predict(input_data)[0]
            final_score = np.expm1(log_pred) # Log dönüşümünü geri al

            st.divider()
            st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

            # 1. Metrik Kartları
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Tahmini Upvote", f"{int(final_score)} ↑")
            with m2:
                s_status = "Pozitif" if v_score > 0.05 else "Negatif" if v_score < -0.05 else "Nötr"
                st.metric("Duygu Tonu (VADER)", s_status)
            with m3:
                h_status = "Yüksek" if hype > 2 or emojis > 3 else "Normal"
                st.metric("Hype Yoğunluğu", h_status)

            # 2. Manipülasyon ve Risk Paneli
            st.write("---")
            risk_score = min((hype * 25) + (abs(v_score) * 20) + (emojis * 10), 100)
            
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.write(f"### Tahmin Edilen Manipülasyon Riski: %{risk_score:.1f}")
                st.progress(risk_score / 100)
                if risk_score > 50:
                    st.error("🚨 **Yüksek Risk:** Bu başlık yapay olarak 'hype' oluşturma potansiyeline sahip.")
                else:
                    st.success("✅ **Güvenli:** İçerik organik etkileşim kriterlerine uygun.")

            with c_right:
                st.write("**İçerik Özet Verisi**")
                st.write(f"✨ Emoji Sayısı: {emojis}")
                st.write(f"🔠 Büyük Harf Modu: {'Aktif' if is_caps else 'Kapalı'}")
                st.write(f"🔥 Hype Skoru: {hype}")

            # 3. Teknik Tablo
            st.write("---")
            st.subheader("📋 Teknik Detaylar")
            tech_df = pd.DataFrame({
                "Parametre": ["VADER Compound", "Emoji Sayısı", "Büyük Harf", "Hype Kelime", "Subreddit"],
                "Değer": [f"{v_score:.4f}", emojis, "Evet" if is_caps else "Hayır", hype, selected_subreddit]
            })
            st.table(tech_df)

        except Exception as e:
            st.error(f"Tahmin sırasında bir hata oluştu: {e}")

# --- SEKME 2: VERİ ANALİZİ (EDA) ---
with tab_eda:
    st.header("🔬 Keşifsel Veri Analizi (EDA)")
    # Burada mevcut Plotly grafiklerini aynen koruyabilirsin.
    st.info("Veri setindeki eğilimleri görmek için grafikleri inceleyin.")
    
    # Örnek Isı Haritası (Eski kodundaki gibi devam eder)
    labels = ['Skor', 'Hype Sayısı', 'VADER Duygu', 'Emoji Sayısı']
    z = [[1, 0.45, 0.30, 0.35], [0.45, 1, 0.60, 0.40], [0.30, 0.60, 1, 0.20], [0.35, 0.40, 0.20, 1]]
    fig_corr = px.imshow(z, x=labels, y=labels, color_continuous_scale='RdBu_r', text_auto=True)
    st.plotly_chart(fig_corr, use_container_width=True)
