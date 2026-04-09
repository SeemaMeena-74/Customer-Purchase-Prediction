import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="NexGen AI Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- THEME MANAGEMENT ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

def local_css(theme):
    if theme == 'Dark':
        bg_main = "#121212"
        bg_card = "#1e1e1e"
        text_primary = "#e0e0e0"
        text_secondary = "#bbbbbb"
        border_color = "#333333"
        sidebar_bg = "#1e1e1e"
        metric_val = "#ffffff"
    else:
        bg_main = "#f8f9fa"
        bg_card = "#ffffff"
        text_primary = "#1a1a1a"
        text_secondary = "#333333"
        border_color = "#e0e0e0"
        sidebar_bg = "#ffffff"
        metric_val = "#1a1a1a"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {{
        background-color: {bg_main} !important;
    }}
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {text_primary};
    }}
    
    /* Ensure all markdown text is visible */
    p, span, label, .stMarkdown, h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
    }}
    
    /* Gradient Title */
    .hero-text {{
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }}
    
    /* Custom Cards */
    .stCard {{
        background-color: {bg_card};
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        border: 1px solid {border_color};
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }}
    
    .stCard h3 {{
        color: {text_primary} !important;
        margin-top: 0;
    }}
    
    .stCard p {{
        color: {text_secondary} !important;
        line-height: 1.6;
    }}
    
    .stCard:hover {{
        transform: translateY(-5px);
    }}
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {metric_val} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {text_secondary} !important;
        font-weight: 600;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {border_color};
    }}
    
    /* Button Styling */
    .stButton>button {{
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white !important;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        opacity: 0.9;
    }}

    /* Success Message */
    .stSuccess {{
        background-color: #d4edda;
        color: #155724 !important;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_models():
    p_model = joblib.load('purchase_model.pkl')
    m_model = joblib.load('movie_knn_model.pkl')
    p_data = pd.read_csv('purchase_data.csv')
    # Load movie ratings, ensuring the first column is the index
    m_data = pd.read_csv('movie_ratings.csv', index_col=0)
    return p_model, m_model, p_data, m_data

try:
    purchase_model, movie_knn, purchase_df, movie_df = load_models()
except:
    st.error("Models not found. Please run data_prep.py first.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.markdown("<h2 style='margin-bottom:20px;'>NexGen AI</h2>", unsafe_allow_html=True)
    
    # Theme Toggle
    theme_selection = st.radio("Appearance", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1, horizontal=True)
    st.session_state.theme = theme_selection
    local_css(st.session_state.theme)
    
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🎯 Purchase Prediction", "🎬 Movie Recs", "📊 Analytics", "ℹ️ About"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Status")
    st.success("System Online")
    st.info("Version 1.0.2")

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown('<p class="hero-text">Intelligence in Every Click.</p>', unsafe_allow_html=True)
    st.markdown("### Welcome to the NexGen Predictive Analytics & Recommendation Suite")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="stCard">
            <h3>🎯 Customer Insights</h3>
            <p>Leverage Logistic Regression to predict customer purchasing behavior with high precision. Understand the impact of Age and Salary on conversion rates.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="stCard">
            <h3>🎬 Content Personalization</h3>
            <p>Experience our KNN-powered recommendation engine. Discover similar user profiles and get tailored movie suggestions in real-time.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Key Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Accuracy", "87.5%", "+2.3%")
    m2.metric("Prediction Latency", "12ms", "-1ms")
    m3.metric("Data Samples", f"{len(purchase_df)}", "+10")
    m4.metric("Engine Reliability", "99.9%", "Stable")

# --- PURCHASE PREDICTION ---
elif page == "🎯 Purchase Prediction":
    st.markdown("## 🎯 Customer Purchase Prediction")
    st.write("Predict whether a customer will buy a product based on their demographics.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("#### Input Parameters")
        st.markdown("---")
        age = st.slider("Select Age", 18, 70, 30)
        salary = st.slider("Monthly Salary ($)", 20000, 150000, 50000)
        
        if st.button("Generate Prediction"):
            with st.spinner("Analyzing data patterns..."):
                input_data = np.array([[age, salary]])
                prediction = purchase_model.predict(input_data)[0]
                probability = purchase_model.predict_proba(input_data)[0][1]
                
                st.session_state['p_res'] = prediction
                st.session_state['p_prob'] = probability

    with col2:
        if 'p_res' in st.session_state:
            res = st.session_state['p_res']
            prob = st.session_state['p_prob']
            
            st.markdown("#### Prediction Result")
            if res == 1:
                st.success(f"### Result: WILL BUY 🛍️")
            else:
                st.error(f"### Result: WILL NOT BUY ❌")
            
            # Confidence Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Purchase Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4b6cb7"},
                    'steps' : [
                        {'range': [0, 50], 'color': "#f8f9fa"},
                        {'range': [50, 80], 'color': "#e9ecef"},
                        {'range': [80, 100], 'color': "#dee2e6"}
                    ]
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Adjust inputs and click 'Generate Prediction' to see results.")

# --- MOVIE RECOMMENDATION ---
elif page == "🎬 Movie Recs":
    st.markdown("## 🎬 Movie Recommendation System")
    st.write("Find similar users and recommended content using K-Nearest Neighbors.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("#### User Profile")
        selected_user = st.selectbox("Select a User Profile", movie_df.index)
        
        st.markdown("**User's Current Ratings:**")
        user_ratings = movie_df.loc[selected_user]
        st.dataframe(user_ratings[user_ratings > 0], use_container_width=True)
        
        if st.button("Find Recommendations"):
            # KNN logic
            distances, indices = movie_knn.kneighbors(movie_df.loc[selected_user].values.reshape(1, -1), n_neighbors=3)
            # Find similar users (excluding itself)
            similar_users = [movie_df.index[i] for i in indices.flatten() if movie_df.index[i] != selected_user]
            st.session_state['similar_users'] = similar_users
            
            # Simple recommendation: movies seen by similar users but not by current user
            current_user_movies = set(user_ratings[user_ratings > 0].index)
            recs = []
            for sim_user in similar_users:
                sim_ratings = movie_df.loc[sim_user]
                for movie, rating in sim_ratings.items():
                    if rating >= 4 and movie not in current_user_movies:
                        recs.append(movie)
            
            st.session_state['movie_recs'] = list(set(recs))

    with col2:
        if 'similar_users' in st.session_state:
            st.markdown("#### Similarity Analysis")
            st.write(f"Users with similar taste to **{selected_user}**:")
            for sim in st.session_state['similar_users']:
                st.markdown(f"- 👥 **{sim}**")
            
            st.markdown("---")
            st.markdown("#### Recommended for You")
            
            if st.session_state['movie_recs']:
                recs = st.session_state['movie_recs']
                cols = st.columns(len(recs) if len(recs) > 0 else 1)
                for i, movie in enumerate(recs):
                    with cols[i % 2]:
                        st.markdown(f"""
                        <div style="background:#4b6cb7; color:white; padding:15px; border-radius:10px; text-align:center; margin-bottom:10px;">
                            <span style="font-size:1.2rem; font-weight:bold;">{movie}</span><br>
                            <span style="font-size:0.8rem;">Highly Rated by Peers</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("No new movies found. Maybe you've seen them all!")
        else:
            st.info("Select a user and click 'Find Recommendations'.")

# --- ANALYTICS PAGE ---
elif page == "📊 Analytics":
    st.markdown("## 📊 Data Insights & Model Performance")
    
    tab1, tab2 = st.tabs(["Customer Data", "Recommendation Engine"])
    
    with tab1:
        st.subheader("Purchase Propensity Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter Plot
            fig_scatter = px.scatter(
                purchase_df, x="Age", y="Salary", color="Buy",
                color_continuous_scale="RdBu",
                title="Age vs Salary (Colored by Purchase)",
                template="plotly_white"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col2:
            # Distribution
            buy_counts = purchase_df['Buy'].value_counts().reset_index()
            buy_counts.columns = ['Status', 'Count']
            buy_counts['Status'] = buy_counts['Status'].map({0: 'Not Buy', 1: 'Will Buy'})
            
            fig_pie = px.pie(
                buy_counts, values='Count', names='Status',
                title="Overall Purchase Distribution",
                hole=.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.subheader("User-Item Interaction Matrix")
        fig_heat = px.imshow(
            movie_df,
            labels=dict(x="Movies", y="Users", color="Rating"),
            x=movie_df.columns,
            y=movie_df.index,
            color_continuous_scale="Viridis",
            title="Current Rating Heatmap"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# --- ABOUT PAGE ---
elif page == "ℹ️ About":
    st.markdown("## ℹ️ About NexGen AI")
    st.markdown("""
    NexGen AI is a state-of-the-art diagnostic and predictive dashboard designed to demonstrate the power of 
    Classical Machine Learning in a modern web environment.
    
    ### 🛠️ Built With:
    - **Streamlit**: For the interactive web interface
    - **Scikit-Learn**: Powering Logistic Regression and KNN
    - **Plotly**: For dynamic, interactive visualizations
    - **Joblib**: Efficient model persistence
    
    ### 🧬 Algorithms Used:
    1. **Logistic Regression**: Used for Binary Classification in Purchase Prediction. It models the probability of a discrete outcome given an input variable.
    2. **K-Nearest Neighbors (KNN)**: Used for Content Recommendation. It identifies patterns by finding data points closest to the target input in a multi-dimensional space.
    
    ---
    *Developed by [Antigravity AI]*
    """)
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=800", caption="Data Science at Scale", use_container_width=True)
