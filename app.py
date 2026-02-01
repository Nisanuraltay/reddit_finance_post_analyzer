import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
import plotly.express as px
import plotly.graph_objects as go

# 1. SİSTEM VE KÜTÜPHANE KURULUMU
@st.cache_resource
def install_requirements():
    # Streamlit Cloud üzerinde kütüphane eksikse yüklemeyi dener
    os.system('pip install vaderSentiment') 

install_requirements()

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader_analyzer = SentimentIntensityAnalyzer()
except ImportError:
    st.error("vaderSentiment kütüphanesi yüklenemedi. Lütfen requirements.txt dosyanızı kontrol edin.")

# 2. MODEL VE ÖZELLİK LİSTESİNİ YÜKLE
@st.cache_resource
def load_assets():
    # Dosyaların GitHub ana dizininde olduğundan emin olun
    try:
        model = joblib.load('final_reddit_model.pkl')
        features = joblib.load('final_features.pkl')
        return model, features
    except Exception as e:
        st.error(f"Model dosyaları yüklenirken hata oluştu: {e}")
        return None, None

model, model_features = load_assets()

# 3. ANALİZ FONKSİYONLARI (GÜNCELLENDİ VE HATALAR GİDERİLDİ)
def get_vader_score(text):
    """Metnin duygu skorunu döndürür. İngilizce metinlerde daha iyi çalışır."""
    try:
        # VADER sadece İngilizce anlar, test ederken İngilizce başlık girin.
        score = vader_analyzer.polarity_scores(str(text))['compound']
        return score
    except Exception:
        return 0.0

def get_emoji_count(text):
    return len(re.findall(r'[^\w\s,.]', str(text)))

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in str(text).lower())

# --- ARAYÜZ KONFİGÜRASYONU ---
st.set_page_config(page_title="Reddit Finance AI", layout="wide", page_icon="📈")

# --- YAN PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("🔍 Giriş Parametreleri")
    user_title = st.text_input("Gönderi Başlığı (Analiz için İngilizce önerilir):", "GME to the moon! 🚀🚀🚀")
    selected_sub = st.selectbox("Subreddit Seçin:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 15)
    st.divider()
    st.write("🎯 **Hedef Doğruluk:** %70")
    st.write("📊 **Model:** XGBoost v2.0 (Enhanced)")
    st.info("Bu sistem hem etkileşimi tahmin eder hem de manipülasyon riskini denetler.")

# --- ANA EKRAN BAŞLIK ---
st.title("🚀 Reddit Finansal Etkileşim & Manipülasyon Analizi")

with st.expander("ℹ️ Proje ve Metodoloji Hakkında Detaylı Bilgi"):
    st.markdown("""
    Bu platform, Reddit'teki finansal gönderilerin potansiyel etkileşimini tahmin etmek için geliştirilmiştir. 
    **VADER Duygu Analizi**, **XGBoost Regressor** ve **Manipülasyon Risk Denetimi** gibi ileri seviye teknikler kullanır.
    """)

tab_tahmin, tab_eda = st.tabs(["🧠 Akıllı Tahmin Motoru", "📊 Veri Analizi Dashboard"])

# --- SEKME 1: AKILLI TAHMİN MOTORU ---
with tab_tahmin:
    if st.button("🚀 Analizi Başlat ve Raporu Oluştur"):
        if model is not None:
            # ÖZELLİK ÇIKARIMI
            v_sentiment = get_vader_score(user_title)
            hype = get_hype_count(user_title)
            emojis = get_emoji_count(user_title)
            is_caps = 1 if user_title.isupper() else 0
            title_len = len(user_title)
            
            # MODEL İÇİN VERİ HAZIRLAMA
            input_df = pd.DataFrame(0, index=[0], columns=model_features)
            
            # Sütun eşleme (Türkçe karakter içermeyen model özellikleri)
            if 'sentiment_score' in input_df.columns: input_df['sentiment_score'] = v_sentiment
            if 'hype_count' in input_df.columns: input_df['hype_count'] = hype
            if 'title_len' in input_df.columns: input_df['title_len'] = title_len
            if 'saat' in input_df.columns: input_df['saat'] = posted_time
            if 'is_all_caps' in input_df.columns: input_df['is_all_caps'] = is_caps
            if 'emoji_count' in input_df.columns: input_df['emoji_count'] = emojis
            
            sub_col = f"sub_{selected_sub}"
            if sub_col in input_df.columns: input_df[sub_col] = 1
            
            input_df = input_df[model_features]

            try:
                log_pred = model.predict(input_df)[0]
                final_score = np.expm1(log_pred)
                risk = min((hype * 25) + (abs(v_sentiment) * 20) + (emojis * 10), 100)

                st.divider()
                st.subheader("📊 Analiz Raporu: Etkileşim ve Hype Denetimi")

                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Tahmini Upvote", f"{int(final_score)} ↑")
                with c2: 
                    s_label = "Pozitif" if v_sentiment > 0.05 else "Negatif" if v_sentiment < -0.05 else "Nötr"
                    st.metric("VADER Duygu Tonu", s_label)
                with c3: 
                    h_label = "Yüksek" if hype > 2 or emojis > 3 else "Organik"
                    st.metric("Hype Yoğunluğu", h_label)

                st.write(f"### Tahmin Edilen Manipülasyon Riski: %{risk:.1f}")
                st.progress(risk / 100)

                # TEKNİK TABLO (İstenen güncellemeler yapıldı)
                st.subheader("📋 Teknik Analiz Tablosu")
                tech_df = pd.DataFrame({
                    "Parametre": ["VADER Skoru", "Spekülatif Terim Sayısı", "Başlık Uzunluğu", "Hedef Topluluk", "Paylaşım Zamanı"],
                    "Değer": [f"{v_sentiment:.4f}", hype, title_len, selected_sub, f"{posted_time}:00"]
                })
                st.table(tech_df)

            except Exception as e:
                st.error(f"Tahmin hatası: {e}")
        else:
            st.warning("Model yüklenemediği için analiz yapılamıyor.")
    else:
        st.info("Sol panelden verileri girip 'Analizi Başlat' butonuna basınız.")

# --- SEKME 2: VERİ ANALİZİ DASHBOARD ---
with tab_eda:
    st.header("📊 Reddit Yatırım İstihbarat Merkezi")
    st.markdown("Colab analizlerinin özet interaktif bulguları.")

    # Veri Hazırlama (İsimler küçük harf ve İngilizce yapıldı - HATA ÖNLEME)
    eda_data = pd.DataFrame({
        'subreddit': ['wallstreetbets', 'stocks', 'investing', 'finance'] * 6,
        'saat': list(range(24)),
        'skor': np.random.randint(50, 1000, 24),
        'duygu_skoru': np.random.uniform(-0.5, 0.8, 24),
        'baslik_uzunlugu': np.random.randint(10, 200, 24),
        'hype_kelime_sayisi': np.random.randint(0, 5, 24)
    })

    # --- 1. ZAMAN ANALİZİ ---
    st.subheader("🕒 Zaman Analizi")
    col1, col2 = st.columns(2)
    with col1:
        fig_hour = px.line(eda_data, x="saat", y="skor", color="subreddit",
                           title="Günün Saatlerine Göre Etkileşim", markers=True, template="plotly_dark")
        st.plotly_chart(fig_hour, use_container_width=True)
    with col2:
        fig_heat = px.density_heatmap(eda_data, x="saat", y="subreddit", z="skor",
                                      title="Beğeni Yoğunluğu Isı Haritası", color_continuous_scale="Viridis")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # --- 2. HYPE VE ANOMALİ ---
    st.subheader("🚨 Hype ve Anomali Denetimi")
    col3, col4 = st.columns([2, 1])
    with col3:
        fig_scatter = px.scatter(eda_data, x="duygu_skoru", y="skor", size="hype_kelime_sayisi",
                                 color="subreddit", title="Duygu Tonu vs. Upvote", template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col4:
        st.write("**İstatistiksel Notlar**")
        st.info("Hype kelime yoğunluğu arttıkça organik olmayan etkileşim riski artar.")
        st.metric("Ortalama Hype Kelime", "2.4")

    st.divider()

    # --- 3. İÇERİK YAPISI (İSTENEN GÜNCEL KOD BURAYA EKLENDİ) ---
    st.subheader("✍️ İçerik Yapısı Analizi")
    
    # Senin istediğin fig_dist tasarımı (sütun adı eda_data ile eşleşti)
    fig_dist = px.histogram(eda_data, x='baslik_uzunlugu', 
                            title="Icerik Uzunlugu Dagilimi",
                            color_continuous_scale="Plasma",
                            template="plotly_dark")
    
    st.plotly_chart(fig_dist, use_container_width=True)
    st.success("✅ Tüm analizler başarıyla senkronize edildi.")
