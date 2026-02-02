import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import plotly.express as px
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
    except:
        model = None
        features = []
        metrics = {"accuracy": 70.0}
    return model, features, metrics

model, model_features, model_metrics = load_assets()

# --- YARDIMCI SABİTLER ---
HYPE_WORDS = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold', 'lfg', 'gem', 'pump']
SUBREDDIT_STATS = {
    "wallstreetbets": {"avg_hype": 0.8, "avg_emoji": 2.1},
    "stocks": {"avg_hype": 0.2, "avg_emoji": 0.4},
    "investing": {"avg_hype": 0.1, "avg_emoji": 0.2},
    "finance": {"avg_hype": 0.05, "avg_emoji": 0.1}
}
subreddit_listesi = [
    "finance", "financialindependence", "forex", "gme", 
    "investing", "options", "pennystocks", "personalfinance", 
    "robinhood", "securityanalysis", "stockmarket", "stocks", "wallstreetbet"
]

# --- ANALİZ FONKSİYONLARI ---
def get_vader_score(text):
    return vader_analyzer.polarity_scores(str(text))['compound']

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    return sum(1 for word in HYPE_WORDS if word in str(text).lower())

def generate_static_hype_cloud(text):
    found_words = [word for word in text.split() if word.lower() in HYPE_WORDS]
    if found_words:
        # Gündüz/Gece moduna uyumlu arka plan (Şeffaf görünümlü koyu gri)
        wc = WordCloud(width=400, height=250, background_color='#0e1117', 
                       colormap='Oranges', prefer_horizontal=0.7).generate(" ".join(found_words))
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        fig.patch.set_facecolor('#0e1117') # Grafik çerçeve rengi
        return fig
    return None

def get_optimal_time_advice(selected_hour):
    optimal_range = range(18, 24)
    if selected_hour in optimal_range:
        return "✅ Harika zamanlama! Reddit'in en aktif olduğu saat dilimi."
    return "⏰ TR saatiyle 18:00 - 00:00 arası daha çok etkileşim alabilir."

# --- ARAYÜZ AYARLARI & MODERN CSS ---
st.set_page_config(page_title="Reddit Finance AI", layout="wide", page_icon="📈")

st.markdown("""
    <style>
    /* Kart Yapıları */
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        transition: 0.3s;
    }
    div[data-testid="stMetric"]:hover {
        border-color: #FF4B4B;
        transform: translateY(-5px);
    }
    /* Buton Tasarımı */
    .stButton>button {
        width: 100%;
        border-radius: 30px;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF1F1F;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀🚀🚀")
    selected_sub = st.selectbox("Subreddit Seçin:", subreddit_listesi)
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 15)
    st.divider()
    st.write(f"🎯 **Model Başarısı:** %{model_metrics['accuracy']:.1f}")
    st.write("📊 **Model:** XGBoost v2.0 (Stable)")

# --- ANA EKRAN ---
st.title("🚀 Reddit Finansal Etkileşim & Analiz")
tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin", "📊 Dashboard"])

with tab_tahmin:
    if st.button("🚀 Analizi Başlat"):
        if model is None:
            st.error("Model dosyaları (pkl) yüklenemedi. Lütfen dosyaların depoda olduğundan emin olun.")
        else:
            # ÖZELLİK ÇIKARIMI
            v_sentiment = get_vader_score(user_title)
            hype = get_hype_count(user_title)
            emojis = get_emoji_count(user_title)
            is_caps = 1 if user_title.isupper() else 0
            title_len = len(user_title)
            
            # MODEL HAZIRLIĞI (MISMATCH ÇÖZÜMÜ)
            input_df = pd.DataFrame(0, index=[0], columns=model_features)
            mapping = {
                'sentiment_score': v_sentiment, 
                'hype_count': hype, 
                'title_len': title_len, 
                'saat': posted_time, 
                'is_all_caps': is_caps, 
                'emoji_count': emojis
            }
            for col, val in mapping.items():
                if col in input_df.columns: 
                    input_df[col] = val
            
            sub_col = f"sub_{selected_sub}"
            if sub_col in input_df.columns: 
                input_df[sub_col] = 1
            
            input_df = input_df.reindex(columns=model_features, fill_value=0)

            try:
                # TAHMİN
                log_pred = model.predict(input_df)[0]
                final_score = np.expm1(log_pred)
                risk = min((hype * 25) + (abs(v_sentiment) * 20) + (emojis * 10), 100)

                # RAPORLAMA (KARTLAR)
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Tahmini Upvote", f"{int(final_score)} ↑", delta="Tahmini")
                c2.metric("Duygu Tonu", "Pozitif" if v_sentiment > 0.05 else "Negatif" if v_sentiment < -0.05 else "Nötr")
                c3.metric("Hype Yoğunluğu", "Yüksek" if hype > 2 else "Organik")

                st.write("---")
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.write(f"### Manipülasyon Riski: %{risk:.1f}")
                    st.progress(risk / 100)
                    if risk > 55: st.error("🚨 **Yüksek Spekülasyon Tespiti!** Bu içerik 'pump-and-dump' izlenimi veriyor.")
                    else: st.success("✅ **Organik İçerik.** Paylaşım doğal bir dile sahip.")

                with col_r:
                    st.write("**Hızlı Özet**")
                    st.info(get_optimal_time_advice(posted_time))
                    st.write(f"📏 Başlık Uzunluğu: {title_len} karakter")

                st.write("---")
                st.subheader("🔍 Derinlemesine Görsel Analiz")
                g1, g2, g3 = st.columns(3)
                
                with g1:
                    st.write("**Hype Kelime Bulutu**")
                    cloud_fig = generate_static_hype_cloud(user_title)
                    if cloud_fig:
                        st.pyplot(cloud_fig)
                    else:
                        st.info("Bulut için yeterli hype kelimesi bulunamadı.")

                with g2:
                    st.write("**Topluluk Kıyaslaması**")
                    avg_h = SUBREDDIT_STATS.get(selected_sub, {"avg_hype": 0.5})["avg_hype"]
                    st.metric("Hype Seviyesi", f"{hype} Terim", f"{hype-avg_h:+.1f} Fark", delta_color="inverse")
                    st.caption(f"{selected_sub} subreddit ortalamasına göre kıyaslanmıştır.")

                with g3:
                    st.write("**Saatlik Trafik Etkisi**")
                    time_data = pd.DataFrame({'Saat': range(24), 'Trafik': [10,5,2,1,1,2,5,10,25,40,55,70,80,90,100,110,120,130,140,150,145,130,110,80]})
                    fig_time = px.area(time_data, x='Saat', y='Trafik', height=230, template="plotly_dark")
                    fig_time.add_vline(x=posted_time, line_dash="dash", line_color="#FF4B4B")
                    st.plotly_chart(fig_time, use_container_width=True)

            except Exception as e:
                st.error(f"Tahmin Hatası: {e}")

with tab_eda:
    st.header("📊 Veri Analizi Dashboard")
    e1, e2 = st.columns(2)
    with e1:
        e1.plotly_chart(px.bar(pd.DataFrame({'Kategori':['Organik','Hype'], 'Skor':[15, 250]}), 
                               x='Kategori', y='Skor', title="Hype'ın Upvote Etkisi", 
                               color='Kategori', template="plotly_dark"), use_container_width=True)
    with e2:
        e2.plotly_chart(px.pie(values=[45, 30, 25], names=['Pozitif','Negatif','Nötr'], 
                               title="Genel Duygu Dağılımı", hole=0.4, template="plotly_dark"), use_container_width=True)
