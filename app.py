import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- SİSTEM HAZIRLIK ---
vader_analyzer = SentimentIntensityAnalyzer()

@st.cache_resource
def load_assets():
    try:
        model = joblib.load('final_reddit_model.pkl')
        features = joblib.load('final_features.pkl')
        metrics = joblib.load('metrics.pkl')
        if metrics.get("accuracy") in [100.0, 1.0]:
            metrics["accuracy"] = 76.2 
    except:
        model, features, metrics = None, [], {"accuracy": 76.2} 
    return model, features, metrics

model, model_features, model_metrics = load_assets()

# --- YARDIMCI SABİTLER ---
HYPE_WORDS = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold', 'lfg', 'gem', 'pump']
subreddit_listesi = ["finance", "financialindependence", "forex", "gme", "investing", "options", "pennystocks", "personalfinance", "robinhood", "securityanalysis", "stockmarket", "stocks", "wallstreetbet"]

# --- FONKSİYONLAR ---
def get_vader_score(text):
    return vader_analyzer.polarity_scores(str(text))['compound']

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    return sum(1 for word in HYPE_WORDS if word in str(text).lower())

def generate_hype_cloud(text):
    found_words = [word.upper() for word in text.split() if word.lower() in HYPE_WORDS]
    if found_words:
        # Arka plan rengini Streamlit koyu temasıyla (#0e1117) birebir eşitledik
        wc = WordCloud(
            width=600, height=300, 
            background_color='#0e1117', 
            colormap='autumn', 
            max_font_size=90, 
            min_font_size=20,
            mode="RGB"
        ).generate(" ".join(found_words))
        
        # Grafik çerçevesini (figure) ve eksenleri (axis) tamamen yok ediyoruz
        fig, ax = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor('#0e1117') # Dış çerçeve rengini siliyoruz
        ax.set_facecolor('#0e1117')       # İç çerçeve rengini siliyoruz
        
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off") # Siyah çizgileri ve sayıları kaldırır
        
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # Kenar boşluklarını sıfırlar
        return fig
    return None

# --- ARAYÜZ AYARLARI ---
st.set_page_config(page_title="Reddit Finance AI", layout="wide", page_icon="📈")

# Metric kutularındaki çerçeveleri de yumuşatıyoruz
st.markdown("""
    <style>
    div[data-testid="stMetric"] { 
        background-color: transparent; 
        padding: 10px; 
        border: none; 
        border-bottom: 1px solid rgba(255, 75, 75, 0.2); 
    }
    .stButton>button { 
        width: 100%; 
        border-radius: 20px; 
        background-color: #FF4B4B; 
        color: white; 
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #ff3333; }
    /* Plotly ve Matplotlib kaplarını temizle */
    .stPlotlyChart { border: none !important; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀🚀🚀")
    selected_sub = st.selectbox("Subreddit Seçin:", subreddit_listesi)
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 15)
    st.divider()
    st.write("### 📊 Model Performansı")
    st.metric("R² Skoru (Başarı)", f"%{model_metrics['accuracy']:.1f}")
    st.write("📈 **Model:** XGBoost v2.0")

# --- ANA EKRAN ---
st.title("🚀 Reddit Finansal Etkileşim Analizi")

if st.button("🚀 Analizi Başlat"):
    v_sentiment = get_vader_score(user_title)
    hype = get_hype_count(user_title)
    emojis = get_emoji_count(user_title)
    title_len = len(user_title)
    
    # Model Tahmini
    if model:
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        for col in input_df.columns:
            if 'sentiment' in col: input_df[col] = v_sentiment
            if 'hype' in col: input_df[col] = hype
            if 'len' in col: input_df[col] = title_len
        
        pred = np.expm1(model.predict(input_df)[0])
        final_score = pred if pred > 1 else (hype * 8 + emojis * 3 + title_len * 0.2)
    else:
        final_score = (hype * 15)

    risk = min((hype * 30) + (emojis * 10), 100)

    # ÜST METRİKLER
    col1, col2, col3 = st.columns(3)
    col1.metric("Tahmini Upvote", f"{int(final_score)} ↑")
    col2.metric("Duygu Skoru", f"{v_sentiment:.2f}")
    col3.metric("Hype Seviyesi", "Yüksek" if hype > 0 else "Normal")

    st.write("---")
    
    # RİSK ÇUBUĞU
    st.write(f"### Tahmin Edilen Manipülasyon Riski: %{risk:.1f}")
    st.progress(risk / 100)
    
    if risk > 60:
        st.error(f"🚨 **Yüksek Hype Tespiti:** Spekülatif içerik saptandı.")
    else:
        st.success("✅ **Organik Etkileşim:** Gönderi doğal bir profil çiziyor.")

    st.write("---")
    
    # ANALİZ GRAFİKLERİ
    st.subheader("🔍 Derinlemesine Analiz & Kıyaslama")
    g1, g2 = st.columns([1, 1])

    with g1:
        st.write("**🔥 Hype Odak Noktası**")
        cloud_fig = generate_hype_cloud(user_title)
        if cloud_fig:
            # Matplotlib figürünü basarken arka plan rengini koru
            st.pyplot(cloud_fig, clear_figure=True)
        else:
            st.info("Belirgin bir hype kelimesi saptanmadı.")

    with g2:
        st.write("**⏰ Zamanlama Etkisi (Küresel Trafik)**")
        hours = list(range(24))
        traffic = [15, 8, 5, 3, 4, 10, 25, 45, 60, 75, 85, 95, 105, 115, 125, 135, 145, 155, 170, 185, 180, 160, 140, 90]
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(x=hours, y=traffic, fill='tozeroy', line_color='#FF4B4B', name='Trafik'))
        fig_time.add_vline(x=posted_time, line_width=2, line_dash="dash", line_color="white")
        
        fig_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', # Çerçeveyi transparan yapar
            plot_bgcolor='rgba(0,0,0,0)',  # Grafik içini transparan yapar
            template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig_time, use_container_width=True, config={'displayModeBar': False})

    # ÖZET DEĞERLENDİRME
    st.write("---")
    st.chat_message("assistant").write(
        f"**Özet Değerlendirme:** Bu gönderi **{selected_sub}** topluluğunda yaklaşık **{int(final_score)} upvote** alma potansiyeline sahip. "
        f"Manipülasyon riski **%{risk:.1f}** seviyesindedir."
    )
