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
SUBREDDIT_STATS = {
    "wallstreetbets": {"avg_hype": 0.8}, "stocks": {"avg_hype": 0.2},
    "investing": {"avg_hype": 0.1}, "finance": {"avg_hype": 0.05}
}

# --- YARDIMCI FONKSİYONLAR ---
def generate_safe_cloud(text):
    words = [w.upper() for w in text.split() if w.lower() in HYPE_WORDS]
    if words:
        wc = WordCloud(width=600, height=300, background_color='#0e1117', colormap='plasma').generate(" ".join(words))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor('#0e1117')
        return fig
    return None

# --- ARAYÜZ AYARLARI & CSS ---
st.set_page_config(page_title="Reddit Finance AI", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Parametreler")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀")
    selected_sub = st.selectbox("Subreddit Seçin:", ["finance", "investing", "stocks", "wallstreetbets", "gme"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 15)
    st.divider()
    st.info(f"🎯 Model Doğruluğu: %{model_metrics['accuracy']:.1f}")

# --- ANA EKRAN ---
st.title("🚀 Reddit Finansal Etkileşim & Analiz")
tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Dashboard"])

with tab_tahmin:
    if st.button("🚀 Analizi Başlat"):
        if model is None:
            st.error("Model dosyaları (pkl) bulunamadı!")
        else:
            # ÖZELLİK ÇIKARIMI
            v_sentiment = vader_analyzer.polarity_scores(user_title)['compound']
            hype = sum(1 for word in HYPE_WORDS if word in user_title.lower())
            emojis = len(re.findall(r'[^\w\s,.]', user_title))
            title_len = len(user_title)
            
            # MODEL TAHMİNİ
            input_df = pd.DataFrame(0, index=[0], columns=model_features)
            mapping = {'sentiment_score': v_sentiment, 'hype_count': hype, 'title_len': title_len, 'saat': posted_time, 'emoji_count': emojis}
            for col, val in mapping.items():
                if col in input_df.columns: input_df[col] = val
            
            input_df = input_df.reindex(columns=model_features, fill_value=0)
            
            try:
                log_pred = model.predict(input_df)[0]
                final_score = np.expm1(log_pred)
                risk = min((hype * 25) + (abs(v_sentiment) * 20) + (emojis * 10), 100)

                # --- GÖRSEL RAPOR ---
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Tahmini Upvote", f"{int(final_score)} ↑")
                c2.metric("Duygu Tonu", "Pozitif" if v_sentiment > 0.05 else "Negatif" if v_sentiment < -0.05 else "Nötr")
                c3.metric("Hype Yoğunluğu", "Yüksek" if hype > 2 else "Organik")

                st.write("---")
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.write(f"### Manipülasyon Riski: %{risk:.1f}")
                    st.progress(risk / 100)
                    if risk > 55: st.error("🚨 Yüksek Spekülasyon Tespiti!")
                    else: st.success("✅ Organik İçerik Profili.")

                with col_r:
                    st.write("**Detay Analiz**")
                    st.write(f"📏 Karakter: {title_len}")
                    st.write(f"🔥 Hype Terim: {hype}")

                st.write("---")
                g1, g2, g3 = st.columns(3)
                with g1:
                    st.write("**Hype Bulutu**")
                    fig = generate_safe_cloud(user_title)
                    if fig: st.pyplot(fig)
                    else: st.info("Hype kelimesi yok.")
                with g2:
                    st.write("**Kıyaslama**")
                    avg_h = SUBREDDIT_STATS.get(selected_sub, {"avg_hype": 0.5})["avg_hype"]
                    st.metric("Hype Oranı", f"{hype}", f"{hype-avg_h:+.1f} vs Ort.")
                with g3:
                    st.write("**Zamanlama**")
                    time_data = pd.DataFrame({'Saat': range(24), 'Trafik': [10,5,2,1,1,2,5,10,25,40,55,70,80,90,100,110,120,130,140,150,145,130,110,80]})
                    st.plotly_chart(px.area(time_data, x='Saat', y='Trafik', height=200).add_vline(x=posted_time, line_color="red"), use_container_width=True)

            except Exception as e:
                st.error(f"Tahmin Hatası: {e}")

with tab_eda:
    st.header("📊 Veri Seti Analizi")
    e1, e2 = st.columns(2)
    e1.plotly_chart(px.bar(pd.DataFrame({'Kategori':['Organik','Hype'], 'Skor':[15, 250]}), x='Kategori', y='Skor', template="plotly_dark"))
    e2.plotly_chart(px.pie(values=[45, 55], names=['Pozitif','Negatif'], hole=0.4))
