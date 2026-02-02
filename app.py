import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- SİSTEM HAZIRLIK ---
vader_analyzer = SentimentIntensityAnalyzer()

# 2. MODEL VE ÖZELLİK LİSTESİNİ YÜKLE
@st.cache_resource
def load_assets():
    model = joblib.load('final_reddit_model.pkl')
    features = joblib.load('final_features.pkl')
    # Metrik dosyasını yükle (Dosya yoksa varsayılan 70 döner)
    try:
        metrics = joblib.load('metrics.pkl')
    except:
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

# 3. ANALİZ FONKSİYONLARI
def get_vader_score(text):
    return vader_analyzer.polarity_scores(str(text))['compound']

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    return sum(1 for word in HYPE_WORDS if word in str(text).lower())

def generate_hype_cloud(text):
    found_words = [word for word in text.split() if word.lower() in HYPE_WORDS]
    if found_words:
        # Arka plan rengini Streamlit koyu temasına (dark) uygun ayarladık
        wordcloud = WordCloud(width=400, height=200, background_color='#0e1117', 
                              colormap='Oranges').generate(" ".join(found_words))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        return fig
    return None

def get_optimal_time_advice(selected_hour):
    # Reddit global trafik genelde 15:00 - 21:00 UTC (TR: 18:00 - 00:00) arası zirve yapar
    optimal_range = range(18, 24)
    if selected_hour in optimal_range:
        return "✅ Harika zamanlama! Gönderi, Reddit'in en aktif olduğu saat diliminde."
    else:
        return "⏰ Not: Gönderiyi TR saatiyle 18:00 - 00:00 arasında paylaşmak etkileşimi artırabilir."

# --- ARAYÜZ KONFİGÜRASYONU ---
st.set_page_config(page_title="Reddit Finance AI", layout="wide", page_icon="📈")

# --- YAN PANEL (SIDEBAR) ----
subreddit_listesi = [
    "finance", "financialindependence", "forex", "gme", 
    "investing", "options", "pennystocks", "personalfinance", 
    "robinhood", "securityanalysis", "stockmarket", "stocks", "wallstreetbet"
]

# --- YAN PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀🚀🚀")
    
    # GÜNCELLENEN KISIM: Subreddit listesini değişkenden alıyoruz
    selected_sub = st.selectbox("Subreddit Seçin:", subreddit_listesi)
    
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 15)
    st.divider()
    
    # GÜNCELLENEN KISIM: Doğruluk oranı modelden gelen metrikle dinamikleşti
    st.write(f"🎯 **Hedef Doğruluk:** %{model_metrics['accuracy']:.1f}")
    st.write("📊 **Model:** XGBoost v2.0 (Enhanced)")
    st.info("Bu sistem hem etkileşimi tahmin eder hem de manipülasyon riskini denetler.")

# --- ANA EKRAN ---
st.title("🚀 Reddit Finansal Etkileşim & Manipülasyon Analizi")
tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Analizi Dashboard"])

# --- SEKME 1: AKILLI TAHMİN MOTORU ---
with tab_tahmin:
    if st.button("🚀 Analizi Başlat ve Raporu Oluştur"):
        # ÖZELLİK ÇIKARIMI
        v_sentiment = get_vader_score(user_title)
        hype = get_hype_count(user_title)
        emojis = get_emoji_count(user_title)
        is_caps = 1 if user_title.isupper() else 0
        title_len = len(user_title)
        
        # MODEL İÇİN VERİ HAZIRLAMA
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        if 'sentiment_score' in input_df.columns: input_df['sentiment_score'] = v_sentiment
        if 'hype_count' in input_df.columns: input_df['hype_count'] = hype
        if 'title_len' in input_df.columns: input_df['title_len'] = title_len
        if 'saat' in input_df.columns: input_df['saat'] = posted_time
        if 'is_all_caps' in input_df.columns: input_df['is_all_caps'] = is_caps
        if 'emoji_count' in input_df.columns: input_df['emoji_count'] = emojis
        
        sub_col = f"sub_{selected_sub}"
        if sub_col in input_df.columns:
            input_df[sub_col] = 1
        
        input_df = input_df[model_features]

        try:
            # TAHMİN
            log_pred = model.predict(input_df)[0]
            final_score = np.expm1(log_pred)
            
            # RİSK HESAPLAMA
            risk = min((hype * 25) + (abs(v_sentiment) * 20) + (emojis * 10), 100)

            # --- GÖRSEL RAPORLAMA ---
            st.divider()
            st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Tahmini Upvote", f"{int(final_score)} ↑")
            with c2:
                s_label = "Pozitif" if v_sentiment > 0.05 else "Negatif" if v_sentiment < -0.05 else "Nötr"
                st.metric("VADER Duygu Tonu", s_label)
            with c3:
                h_label = "Yüksek" if hype > 2 or emojis > 3 else "Organik"
                st.metric("Hype Yoğunluğu", h_label)

            # Manipülasyon Göstergesi
            st.write("---")
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.write(f"### Tahmin Edilen Manipülasyon Riski: %{risk:.1f}")
                st.progress(risk / 100)
                if risk > 55:
                    st.error("🚨 **Yüksek Hype Tespiti:** Spekülatif içerik ve aşırı emoji kullanımı saptandı.")
                else:
                    st.success("✅ **Organik Etkileşim:** Gönderi doğal bir paylaşım profili çiziyor.")

            with col_r:
                st.write("**İçerik Detayları**")
                st.write(f"📏 Karakter: {title_len}")
                st.write(f"🔥 Spekülatif Terim: {hype} adet")
                st.write(get_optimal_time_advice(posted_time))

            # --- DERİNLEMESİNE ANALİZ PANELİ ---
            st.write("---")
            st.subheader("🔍 Derinlemesine Analiz & Kıyaslama")
            g1, g2, g3 = st.columns(3)

            with g1:
                st.write("**Hype Kelime Bulutu**")
                cloud_fig = generate_hype_cloud(user_title)
                if cloud_fig: st.pyplot(cloud_fig)
                else: st.info("Hype kelimesi bulunamadı.")

            with g2:
                st.write("**Topluluk Kıyaslaması**")
                avg_h = SUBREDDIT_STATS.get(selected_sub, {"avg_hype": 0.5})["avg_hype"]
                diff = ((hype - avg_h) / avg_h * 100) if avg_h > 0 else (hype * 100)
                st.write(f"Bu gönderi, **{selected_sub}** ortalamasından:")
                st.metric("Hype Oranı", f"{hype} Terim", f"%{diff:.1f} {'Fazla' if diff >=0 else 'Az'}", delta_color="inverse")

            with g3:
                st.write("**Zamanlama Etkisi**")
                time_data = pd.DataFrame({
                    'Saat': list(range(24)), 
                    'Trafik': [10,5,2,1,1,2,5,10,25,40,55,70,80,90,100,110,120,130,140,150,145,130,110,80]
                })
                fig_time = px.area(time_data, x='Saat', y='Trafik', title="Global Reddit Etkileşim Grafiği")
                fig_time.add_vline(x=posted_time, line_dash="dash", line_color="red", annotation_text="Sizin Saatiniz")
                st.plotly_chart(fig_time, use_container_width=True)

            # Teknik Analiz Tablosu
            st.write("---")
            st.subheader("📋 Teknik Analiz Tablosu")
            tech_df = pd.DataFrame({
                "Parametre": ["VADER Skoru", "Hype Kelime", "Emoji Sayısı", "Büyük Harf", "Hedef Subreddit"],
                "Değer": [f"{v_sentiment:.4f}", hype, emojis, "Evet" if is_caps else "Hayır", selected_sub]
            })
            st.table(tech_df)

            st.chat_message("assistant").write(
                f"**Özet Değerlendirme:** Bu gönderi {selected_sub} topluluğunda yaklaşık {int(final_score)} upvote alma potansiyeline sahip. "
                f"Manipülasyon riski %{risk:.1f} seviyesindedir."
            )

        except Exception as e:
            st.error(f"Sistem Hatası: {e}")

# --- SEKME 2: VERİ ANALİZİ DASHBOARD ---
with tab_eda:
    st.header("🔬 Colab Veri Analiz Çıktıları (EDA)")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        eda_data = pd.DataFrame({'Kategori': ['Organik', 'Orta Hype', 'Yüksek Hype'], 'Ortalama Skor': [15, 65, 280]})
        fig = px.bar(eda_data, x='Kategori', y='Ortalama Skor', color='Ortalama Skor', title="Hype Seviyesine Göre Etkileşim Artışı", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    with e_col2:
        fig2 = px.pie(values=[45, 25, 30], names=['Pozitif', 'Negatif', 'Nötr'], title="Veri Seti Genel Duygu Dağılımı", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

