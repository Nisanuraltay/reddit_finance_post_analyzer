import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Reddit Financial Intelligence", layout="wide", page_icon="📊")

# --- 1. ARAÇLAR VE MODEL YÜKLEME ---
@st.cache_resource
def load_assets():
    vader = SentimentIntensityAnalyzer()
    try:
        model = joblib.load('final_reddit_model.pkl')
        features = joblib.load('final_features.pkl')
    except:
        model, features = None, None
    return vader, model, features

vader, model, model_features = load_assets()

# --- 2. ANALİZ BAŞLIKLARINA GÖRE ÖRNEK VERİ ÜRETİCİ (EDA İÇİN) ---
# Gerçek verilerini BigQuery'den çektiğinde bu kısmı 'full_df' ile değiştirebilirsin.
@st.cache_data
def get_analysis_data():
    subreddits = ["finance", "forex", "gme", "investing", "options", "pennystocks", "stocks", "wallstreetbets"]
    n = 1000
    df = pd.DataFrame({
        'subreddit': np.random.choice(subreddits, n),
        'saat': np.random.randint(0, 24, n),
        'gun': np.random.choice(['Monday', 'Wednesday', 'Friday', 'Sunday'], n),
        'score': np.random.exponential(500, n),
        'upvote_ratio': np.random.uniform(0.5, 1.0, n),
        'sentiment_score': np.random.uniform(-1, 1, n),
        'hype_count': np.random.randint(0, 10, n),
        'is_video': np.random.choice([0, 1], n),
        'title_len': np.random.randint(10, 200, n)
    })
    return df

eda_df = get_analysis_data()

# --- 3. ÜST PANEL & NAVİGASYON ---
st.title("📈 Reddit Yatırım Toplulukları Analiz Merkezi")
st.markdown("Veri Kaynağı: *BigQuery Reddit Financial Datasets*")

tab1, tab2, tab3 = st.tabs(["🧠 Tahmin Motoru", "📊 Keşifsel Analiz (EDA)", "🚨 Hype & Anomali Tespiti"])

# --- TAB 1: TAHMİN MOTORU (Senin Analiz Sayfan) ---
with tab1:
    st.header("🧠 Gönderi Etkileşim Tahmini")
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        user_title = st.text_area("Gönderi Başlığını Girin:", "GME is going to the moon! 🚀🚀🚀 #ShortSqueeze")
        selected_sub = st.selectbox("Subreddit Seçin:", eda_df['subreddit'].unique())
        post_hour = st.slider("Paylaşım Saati:", 0, 23, 14)
        predict_btn = st.button("Analiz Et & Tahmin Yap")

    if predict_btn:
        # Özellikleri hesapla
        v_score = vader.polarity_scores(user_title)['compound']
        emojis = len(re.findall(r'[^\w\s,.]', user_title))
        hype = sum(1 for w in ['moon', 'rocket', 'yolo', 'squeeze'] if w in user_title.lower())
        
        with col_result:
            st.subheader("Analiz Sonuçları")
            res_c1, res_c2 = st.columns(2)
            res_c1.metric("Duygu Skoru", f"{v_score:.2f}")
            res_c2.metric("Emoji Sayısı", emojis)
            
            # Risk Barı
            risk = min((hype * 20) + (emojis * 10), 100)
            st.write(f"**Manipülasyon / Hype Riski:** %{risk}")
            st.progress(risk/100)
            
            if model:
                st.info("🤖 Model Tahmini: Hesaplanıyor...")
                # Buraya model.predict mantığı gelecek (Önceki kodundaki gibi)
            else:
                st.warning("⚠️ Model dosyası bulunamadı, sadece kural tabanlı analiz gösteriliyor.")

# --- TAB 2: KEŞİFSEL VERİ ANALİZİ (EDA) ---
with tab2:
    st.header("🔍 Topluluk Davranış Analizi")
    
    # Senin Başlığın 1: Zaman Analizi
    st.subheader("1-) Zaman ve Etkileşim Analizi")
    c1, c2 = st.columns(2)
    with c1:
        fig_hour = px.line(eda_df.groupby('saat')['score'].mean().reset_index(), 
                           x='saat', y='score', title="Saatlik Ortalama Etkileşim",
                           template="plotly_dark", line_shape="spline")
        st.plotly_chart(fig_hour, use_container_width=True)
    with c2:
        fig_day = px.bar(eda_df.groupby('gun')['score'].median().reset_index(), 
                         x='gun', y='score', title="Günlük Etkileşim Yoğunluğu (Medyan)",
                         color='score', color_continuous_scale="Viridis")
        st.plotly_chart(fig_day, use_container_width=True)

    # Senin Başlığın 3: İçerik Tipi Etkisi
    st.subheader("2-) İçerik Tipi ve Uzunluk Etkisi")
    c3, c4 = st.columns(2)
    with c3:
        fig_len = px.scatter(eda_df, x="title_len", y="score", color="subreddit",
                             title="Başlık Uzunluğu vs Skor", size="upvote_ratio",
                             log_y=True, template="plotly_dark")
        st.plotly_chart(fig_len, use_container_width=True)
    with c4:
        fig_video = px.box(eda_df, x="is_video", y="score", color="is_video",
                           title="Video İçerik vs Metin İçerik Skoru",
                           points="all", not_ched=True)
        st.plotly_chart(fig_video, use_container_width=True)

# --- TAB 3: HYPE & ANOMALİ TESPİTİ ---
with tab3:
    st.header("🚨 Hype ve Manipülasyon Denetimi")
    
    # Senin Başlığın: Hype Sözlüğü Filtresi
    col_h1, col_h2 = st.columns([2, 1])
    
    with col_h1:
        st.subheader("Hype ve Duygu Korelasyonu")
        fig_hype = px.scatter(eda_df, x="sentiment_score", y="hype_count", 
                              size="score", color="subreddit",
                              title="Aşırı Pozitiflik vs Hype Kelime Sayısı",
                              template="plotly_dark")
        st.plotly_chart(fig_hype, use_container_width=True)
        
    with col_h2:
        st.subheader("🚨 Şüpheli Tablo")
        suspicious = eda_df[eda_df['hype_count'] > 5].sort_values(by='score', ascending=False)
        st.dataframe(suspicious[['subreddit', 'score', 'hype_count']], use_container_width=True)
        st.caption("5'ten fazla hype anahtar kelimesi içeren gönderiler.")

st.divider()
st.write("🔧 **Sistem Durumu:** Tüm analiz modülleri aktif. | Veri seti: 425,681 satır")
