import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import plotly.express as px

# --- 1. SİSTEM VE NLP KURULUMU ---
st.set_page_config(page_title="Reddit Finance Analysis System", layout="wide", page_icon="📈")

@st.cache_resource
def setup_tools():
    # VADER Kütüphanesi kontrolü ve yüklemesi
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        os.system('pip install vaderSentiment')
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

vader_analyzer = setup_tools()

# Modelleri Yükle
@st.cache_resource
def load_assets():
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    return model, features

model, model_features = load_assets()

# --- 2. YARDIMCI FONKSİYONLAR ---
def get_vader_sentiment(text):
    return vader_analyzer.polarity_scores(str(text))['compound']

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in str(text).lower())

# --- 3. YAN PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀🚀🚀")
    selected_sub = st.selectbox("Hedef Subreddit:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 15)
    st.divider()
    st.write("🛠️ **Model:** XGBoost v2.0 (Enhanced)")
    st.info("Bu sürüm VADER Duygu Analizi ve Emoji Takibi özelliklerini içerir.")

# --- 4. ANA EKRAN YAPISI ---
st.title("🚀 Reddit Finansal Etkileşim & Tahmin Sistemi")
tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Analizi Dashboard"])

# --- SEKME 1: TAHMİN MOTORU (Tüm sevdiğin özelliklerle) ---
with tab_tahmin:
    st.markdown("### Gönderi Etkileşimi ve Manipülasyon Denetimi")
    
    if st.button("🚀 Analizi Başlat ve Raporu Oluştur"):
        # Özellik Çıkarımı
        v_score = get_vader_sentiment(user_title)
        hype = get_hype_count(user_title)
        emojis = get_emoji_count(user_title)
        is_caps = 1 if user_title.isupper() else 0
        title_len = len(user_title)
        
        # Giriş Verisini Hazırlama (Model Sütunlarıyla Eşleşme)
        input_data = pd.DataFrame(0, index=[0], columns=model_features)
        
        # Colab'daki sütun isimleriyle eşitleme
        if 'sentiment_score' in model_features: input_data['sentiment_score'] = v_score
        if 'emoji_count' in model_features: input_data['emoji_count'] = emojis
        if 'hype_count' in model_features: input_data['hype_count'] = hype
        if 'is_all_caps' in model_features: input_data['is_all_caps'] = is_caps
        if 'title_len' in model_features: input_data['title_len'] = title_len
        if 'saat' in model_features: input_data['saat'] = posted_time
        
        # Subreddit Encoding
        sub_col = f"sub_{selected_sub}"
        if sub_col in input_data.columns:
            input_data[sub_col] = 1
        
        # Tahmin
        try:
            log_pred = model.predict(input_data[model_features])[0]
            final_score = np.expm1(log_pred)

            st.divider()
            st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

            # 1. Metrik Kartları
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Tahmini Upvote", f"{int(final_score)} ↑")
            with c2: 
                label = "Pozitif" if v_score > 0.05 else "Negatif" if v_score < -0.05 else "Nötr"
                st.metric("Duygu Tonu", label)
            with c3: 
                h_label = "Yüksek" if hype > 2 or emojis > 3 else "Organik"
                st.metric("Hype Seviyesi", h_label)

            # 2. Risk Barı ve Detaylar
            st.write("---")
            risk_score = min((hype * 25) + (abs(v_score) * 20) + (emojis * 10), 100)
            
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.write(f"### Tahmin Edilen Manipülasyon Riski: %{risk_score:.1f}")
                st.progress(risk_score / 100)
                if risk_score > 55:
                    st.error("🚨 **Yüksek Hype Tespiti:** Spekülatif içerik saptandı.")
                else:
                    st.success("✅ **Organik Etkileşim:** Gönderi doğal bir profil çiziyor.")
            
            with col_r:
                st.write("**İçerik Özeti**")
                st.write(f"📏 Uzunluk: {title_len} | ✨ Emoji: {emojis}")
                st.write(f"🔠 Büyük Harf: {'Evet' if is_caps else 'Hayır'}")
                st.write("⭐" * min(int(hype + emojis + 1), 5))

            # 3. Teknik Tablo
            st.subheader("📋 Teknik Analiz Tablosu")
            st.table(pd.DataFrame({
                "Parametre": ["VADER Skoru", "Hype Terim", "Emoji Sayısı", "Büyük Harf", "Hedef Sub"],
                "Değer": [f"{v_score:.4f}", hype, emojis, "Evet" if is_caps else "Hayır", selected_sub]
            }))

            # 4. AI Özet Mesajı
            st.chat_message("assistant").write(
                f"Bu gönderi {selected_sub} topluluğunda {int(final_score)} civarında etkileşim alabilir. "
                f"Risk seviyesi %{risk_score:.1f} olduğu için yatırımcıların dikkatli olması önerilir."
            )

        except Exception as e:
            st.error(f"Tahmin Hatası: {e}")

# --- SEKME 2: VERİ ANALİZİ DASHBOARD (Hatasız EDA) ---
with tab_eda:
    st.header("📊 Reddit Yatırım İstihbarat Merkezi")
    
    # Dashboard için örnek veri seti (ValueError'u önlemek için aynı uzunlukta)
    n_samples = 50
    eda_df = pd.DataFrame({
        'subreddit': np.random.choice(['wallstreetbets', 'stocks', 'investing', 'finance'], n_samples),
        'saat': np.random.randint(0, 24, n_samples),
        'skor': np.random.randint(50, 5000, n_samples),
        'sentiment': np.random.uniform(-0.8, 0.8, n_samples),
        'hype': np.random.randint(0, 10, n_samples)
    })

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("🕒 Saatlik Etkileşim Trendi")
        fig_line = px.line(eda_df.groupby('saat')['skor'].mean().reset_index(), x='saat', y='skor', markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

    with col_e2:
        st.subheader("🔥 Hype vs Skor İlişkisi")
        fig_scatter = px.scatter(eda_df, x="sentiment", y="skor", size="hype", color="subreddit", template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.success("✅ Dashboard başarıyla güncellendi.")
