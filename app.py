import streamlit as st

st.set_page_config(
    page_title="Reddit Hype Analyzer",
    layout="wide"
)

st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Post Analyzer",
        "Insights",
        "Model Explainability",
        "About"
    ]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("📈 Reddit Finance Post Analyzer")
    st.markdown("""
    This tool helps analyze Reddit finance posts by:
    - Predicting engagement potential
    - Detecting hype and manipulation risk
    """)

# ---------------- POST ANALYZER ----------------
elif page == "Post Analyzer":
    st.title("🔍 Post Analyzer")
    st.info("Analyze a Reddit post to estimate engagement and hype risk.")

# ---------------- INSIGHTS ----------------
elif page == "Insights":
    st.title("📊 Insights Dashboard")
    st.info("Summary insights from exploratory data analysis.")

# ---------------- EXPLAINABILITY ----------------
elif page == "Model Explainability":
    st.title("🧠 Model Explainability")
    st.info("Feature importance and model behavior.")

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    **Reddit Finance Post Analyzer**

    Dataset: Kaggle Reddit Finance Data  
    Goal: Engagement prediction & hype risk detection
    """)
