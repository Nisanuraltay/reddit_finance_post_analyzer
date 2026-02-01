import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import os

# TextBlob için gerekli dil paketini indir (Streamlit Cloud için şart)
os.system('python -m textblob.download_corpora')

# 1. Modeli ve Özellik Listesini Yükle
model = joblib.load('final_reddit_model.pkl') 
model_features = joblib.load('final_features.pkl')

# 2. Yardımcı Fonksiyonlar
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

def get_hype_count(text):
    hype_words = ['moon', 'rocket', 'yolo', 'squeeze', 'diamond', 'hands', 'ape', 'short', 'buy', 'hold']
    return sum(1 for word in hype_words if word in text.lower())

# 3. Arayüz Tasarımı
st.set_page_config(page_title="Reddit Finance Analyzer", page_icon="📈", layout="wide")
st.title("📈 Reddit Yatırım Topluluklarında Gönderi Analiz Sistemi")
st.markdown("""
**Proje Özeti:** Bu sistem, finans paylaşımlarını analiz ederek etkileşim tahmini yapar ve 
içeriğin organik mi yoksa manipülasyon kaynaklı mı olduğunu tespit eder.
""")

# Yan Panel: Kullanıcı Girişleri
with st.sidebar:
    st.header("🔍 Analiz Parametreleri")
    user_title = st.text_input("Gönderi Başlığı:", "GME to the moon! 🚀")
    selected_subreddit = st.selectbox("Hedef Subreddit:", ["wallstreetbets", "stocks", "investing", "finance"])
    posted_time = st.slider("Paylaşım Saati (0-23):", 0, 23, 12)
    st.info("Bilgi: Model, başlık içeriği ve topluluk dinamiklerini birleşik olarak analiz eder.")

# Ana Ekran
if st.button("🚀 Analizi Başlat ve Birleşik Raporu Oluştur"):
    # --- ÖZELLİK ÇIKARIMI ---
    sentiment = get_sentiment(user_title)
    hype = get_hype_count(user_title)
    title_len = len(user_title)
    
    # Model hazırlığı ve sütun hizalama
    input_data = pd.DataFrame(0, index=[0], columns=model_features)
    input_data['sentiment_score'] = sentiment
    input_data['hype_count'] = hype
    input_data['title_len'] = title_len
    input_data['saat'] = posted_time
    
    sub_col = f"sub_{selected_subreddit}"
    if sub_col in input_data.columns:
        input_data[sub_col] = 1
    
    input_data = input_data[model_features]

    # --- TAHMİN VE ANALİZ ---
    try:
        log_pred = model.predict(input_data)
