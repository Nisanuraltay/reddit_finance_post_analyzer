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

    with st.form("post_form"):
        title = st.text_input("Post Title")
        selftext = st.text_area("Post Content")

        col1, col2, col3 = st.columns(3)

        with col1:
            hour = st.slider("Post Hour", 0, 23, 12)
            day = st.selectbox(
                "Day of Week",
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )

        with col2:
            is_video = st.checkbox("Is Video Post?")
            num_crossposts = st.number_input("Number of Crossposts", 0, 100, 0)

        with col3:
            upvote_ratio = st.slider("Upvote Ratio", 0.0, 1.0, 0.85)
            total_awards = st.number_input("Total Awards", 0, 100, 0)

        submitted = st.form_submit_button("Analyze Post")

    if submitted:
        st.subheader("📊 Analysis Result")

        # MOCK OUTPUT (şimdilik)
        st.metric("📈 Engagement Forecast", "High")
        st.metric("🚨 Hype Risk Score", "78 / 100")

        st.warning("⚠️ Risk Reasons")
        st.markdown("""
        - High comment-to-score anomaly  
        - Presence of hype-related keywords  
        - Low upvote ratio relative to engagement  
        """)


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

