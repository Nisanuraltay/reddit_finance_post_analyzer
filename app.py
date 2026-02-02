import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- 1. SAYFA VE TEMA AYARLARI ---
st.set_page_config(page_title="Reddit Finans Analiz Robotu", layout="wide")

# Görseldeki profesyonel renk paleti (image_0336ff.png)
COLORS = ['#4E79A7', '#F28E2B', '#E15759'] # Mavi, Turuncu, Kırmızı

st.markdown(f"""
    <style>
    /* Sidebar Tasarımı (Koyu/Gri Kontrastı) */
    [data-testid="stSidebar"] {{
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }}
    /* Metric Kartları (Görsel 1 tarzı temiz kutular) */
    .metric-card {{
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 5px solid {COLORS[0]};
    }}
    .metric-value {{
        font-size: 32px;
        font-weight: bold;
        color: {COLORS[2]};
    }}
    /* Sekme Tasarımı */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding: 10px 20px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- 2. VERİ YÜKLEME VE ANALİZ MOTORU ---
analyzer = SentimentIntensityAnalyzer()

@st.cache_data
def load_and_process_data():
    all_files = glob.glob("*.csv")
    if not all_files:
        return pd.DataFrame()

    data_list = []
    for filename in all_files:
        df_temp = pd.read_csv(filename)
        # Topluluk ismini dosya adından al (Örn: wallstreetbet_clean.csv -> wallstreetbet)
        df_temp['community_source'] = os.path.basename(filename).split('_')[0]
        data_list.append(df_temp)
    
    df = pd.concat(data_list, axis=0, ignore_index=True)
    
    # Tarih Dönüşümü
    for col in ['created', 'created_utc']:
        if col in df.columns:
            df['created'] = pd.to_datetime(df[col], unit='s', errors='coerce')
            break
            
    # NLP & Hype Analizi (Başlıklar üzerinden)
    hype_words = ['moon', 'rocket', 'diamond hands', 'squeeze', 'short', 'pump', 'yolo', 'lamboo']
    
    if 'title' in df.columns:
        df['title'] = df['title'].fillna("").astype(str)
        # Duygu Analizi (VADER)
        df['sentiment_score'] = df['title'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
        df['sentiment_type'] = df['sentiment_score'].apply(
            lambda x: 'Pozitif (Bullish)' if x > 0.05 else ('Negatif (Bearish)' if x < -0.05 else 'Nötr')
        )
        # Hype Kelime Sayımı
        df['hype_count'] = df['title'].apply(lambda x: sum(1 for w in hype_words if w in x.lower()))
        
    return df

df = load_and_process_data()

# --- 3. SIDEBAR (FİLTRELEME PANELİ) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("MyModel Settings")
    
    if not df.empty:
        selected_subs = st.multiselect(
            "Origins (Topluluklar)", 
            options=df['community_source'].unique(),
            default=df['community_source'].unique()
        )
        
        score_filter = st.slider("Min Score Etkisi", 0, int(df['score'].max() if 'score' in df.columns else 1000), 100)
        
        # Filtreleme Uygula
        mask = (df['community_source'].isin(selected_subs)) & (df['score'] >= score_filter)
        filtered_df = df.loc[mask]
    else:
        st.error("Klasörde CSV bulunamadı! Lütfen dosyalarınızı app.py yanına koyun.")
        st.stop()

# --- 4. ANA DASHBOARD ARAYÜZÜ ---
st.title("🚀 Reddit Tahmin ve Analiz Robotu")

# Üst Metrik Kutuları (Görsel 1 Tarzı)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><p>Toplam Gönderi</p><div class="metric-value">{len(filtered_df)}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><p>Ort. Duygu (Sentiment)</p><div class="metric-value">{filtered_df["sentiment_score"].mean():.2f}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><p>Hype Yoğunluğu</p><div class="metric-value">%{ (len(filtered_df[filtered_df["hype_count"] > 0]) / len(filtered_df) * 100):.1f}</div></div>', unsafe_allow_html=True)
with m4:
    anomalies_count = len(filtered_df[(filtered_df['num_comments'] > filtered_df['num_comments'].mean()*2) & (filtered_df['upvote_ratio'] < 0.8)])
    st.markdown(f'<div class="metric-card"><p>Şüpheli İşlem</p><div class="metric-value">{anomalies_count}</div></div>', unsafe_allow_html=True)

st.write("---")

# SAYFALANDIRMA (Tabs)
tab1, tab2, tab3 = st.tabs(["📈 Etkileşim Tahmini", "🚨 Hype & Manipülasyon", "🧠 Duygu Analizi"])

# --- TAB 1: ETKİLEŞİM TAHMİNİ ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("İçerik Tipi Dağılımı")
        fig_bar = px.bar(filtered_df['community_source'].value_counts().reset_index(), 
                         x='index', y='community_source', labels={'index':'Topluluk', 'community_source':'Adet'},
                         color_discrete_sequence=[COLORS[0]], template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("Zaman Analizi (Daily Trends)")
        trend_data = filtered_df.groupby(filtered_df['created'].dt.date).size().reset_index(name='count')
        fig_line = px.line(trend_data, x='created', y='count', 
                          color_discrete_sequence=[COLORS[1]], template="plotly_white")
        st.plotly_chart(fig_line, use_container_width=True)

# --- TAB 2: HYPE & MANİPÜLASYON TESPİTİ ---
with tab2:
    st.subheader("Anomali Radarı (Score vs Comments)")
    fig_scatter = px.scatter(filtered_df, x="score", y="num_comments", 
                            size="hype_count", color="upvote_ratio",
                            color_continuous_scale='RdYlGn', template="plotly_white",
                            hover_data=['title', 'author'])
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.write("**Ödül Enflasyonu Analizi (Top 5 Şüpheli)**")
    if 'total_awards_received' in filtered_df.columns:
        filtered_df['award_ratio'] = filtered_df['total_awards_received'] / (filtered_df['score'] + 1)
        st.dataframe(filtered_df.sort_values(by='award_ratio', ascending=False)[['title', 'author', 'award_ratio', 'community_source']].head(5), use_container_width=True)

# --- TAB 3: DUYGU VE İÇERİK ANALİZİ ---
with tab3:
    col_nlp1, col_nlp2 = st.columns([1, 2])
    with col_nlp1:
        st.subheader("Sentiment Dağılımı")
        fig_pie = px.pie(filtered_df, names='sentiment_type', 
                        color_discrete_map={'Pozitif (Bullish)': COLORS[0], 'Negatif (Bearish)': COLORS[2], 'Nötr': '#BDC3C7'})
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_nlp2:
        st.subheader("Yazar Güvenilirliği (Hype Skoru)")
        author_stats = filtered_df.groupby('author')['hype_count'].sum().sort_values(ascending=False).head(10).reset_index()
        fig_author = px.bar(author_stats, x='hype_count', y='author', orientation='h',
                           color_discrete_sequence=[COLORS[2]], template="plotly_white")
        st.plotly_chart(fig_author, use_container_width=True)

# --- 5. AI INSIGHTS (AKILLI YORUM MOTORU) ---
st.divider()
st.header("🤖 MyModel AI Smart Insights")

def generate_insights(data):
    avg_s = data['sentiment_score'].mean()
    h_ratio = len(data[data['hype_count'] > 0]) / len(data)
    
    res = f"### 📊 Analiz Özeti\n"
    res += f"- **Duygu:** Topluluk şu an {'🟢 Bullish' if avg_s > 0.1 else '🔴 Bearish' if avg_s < -0.1 else '🟡 Nötr'} bir eğilimde.\n"
    res += f"- **Manipülasyon Riski:** İçeriklerin %{h_ratio*100:.1f}'i spekülatif dil (hype) içeriyor.\n"
    
    if anomalies_count > 0:
        res += f"- **Uyarı:** {anomalies_count} gönderide anormal etkileşim saptandı. Bot saldırısı riski yüksek!"
    return res

st.info(generate_insights(filtered_df))
