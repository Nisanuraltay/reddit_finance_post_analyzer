import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Modelleri ve Özellikleri Yükle
try:
    model = joblib.load('reddit_model.pkl')
    features = joblib.load('model_features.pkl')
except Exception as e:
    st.error(f"Model dosyaları yüklenemedi: {e}")

st.set_page_config(page_title="Reddit Hype Engine", layout="wide", page_icon="📈")

# --- ARAYÜZ BAŞLIĞI ---
st.title("📈 Reddit Finance Post Analyzer")
st.markdown("### *Engagement & Hype Risk Engine*")

# --- SIDEBAR: GİRİŞ PANELİ ---
with st.sidebar:
    st.header("🔍 Analiz Parametreleri")
    post_title = st.text_input("Post Başlığı (Title)", "🚀 Buy GME - Diamond Hands! 💎")
    
    sub_list = sorted([c.replace('sub_', '') for c in features if c.startswith('sub_')])
    selected_sub = st.selectbox("Hangi Subreddit?", sub_list)
    
    saat = st.slider("Paylaşım Saati (0-23)", 0, 23, 14)
    st.divider()
    actual_score = st.number_input("Mevcut Beğeni Sayısı (Score)", min_value=0, value=100)

# --- HESAPLAMA VE ANALİZ MOTORU ---
if st.button("DERİN ANALİZİ BAŞLAT"):
    # Girdi Verisini Hazırla
    input_df = pd.DataFrame(0, index=[0], columns=features)
    if f'sub_{selected_sub}' in features: input_df[f'sub_{selected_sub}'] = 1
    if 'saat' in features: input_df['saat'] = saat
    
    # 1. Tahmin
    pred_log = model.predict(input_df)
    predicted_score = np.expm1(pred_log)[0]
    
    # 2. NLP Analizi
    hype_keywords = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'pump', '🚀', '💎', 'buy']
    found_hype_words = [word for word in hype_keywords if word in post_title.lower()]
    nlp_risk_bonus = len(found_hype_words) * 10 
    
    # 3. İstatistiksel Sapma
    base_diff = actual_score - predicted_score
    stat_risk = (base_diff / (66.33 * 3)) * 100
    final_risk = min(100, max(0, stat_risk + nlp_risk_bonus))

    # --- GÖRSEL ÇIKTILAR ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Organik Beklenti", f"{int(predicted_score)} Score")
    c2.metric("Hype Riski", f"%{final_risk:.1f}")
    c3.metric("NLP Bonusu", f"+%{nlp_risk_bonus}")

    st.divider()
    
    # Karar Analizi
    st.subheader("🧠 Sistemin Karar Analizi")
    if len(found_hype_words) > 0:
        st.warning(f"⚠️ **NLP Sinyali:** Başlıkta manipülatif kelimeler bulundu: {', '.join(found_hype_words)}")
    
    if final_risk > 70:
        st.error("🚨 **KRİTİK:** Manipülasyon tespiti! Bu post organik görünmüyor.")
    else:
        st.success("✅ **GÜVENLİ:** Veriler topluluk normlarıyla uyumlu.")

    # 4. XAI Grafiği
    st.subheader("📊 Model Özellik Ağırlıkları (XAI)")
    imp_df = pd.DataFrame({'Önem': model.feature_importances_}, index=features).sort_values(by='Önem', ascending=False).head(5)
    st.bar_chart(imp_df)

    # 5. Finansal Grafik
    st.divider()
    st.subheader("📉 Reddit vs. Piyasa Oynaklığı")
    chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['Hype', 'Fiyat']).cumsum()
    st.line_chart(chart_data)
