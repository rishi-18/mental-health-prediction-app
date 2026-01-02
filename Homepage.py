# import streamlit as st 
# from utils import load_model
# from pathlib import Path

# st.set_page_config("Mental Health Predictor",layout="wide")

# st.title("Mental Heath Predictor App")

# st.markdown("""Welcome to the **Mental Health Predictor App ** dashboard.  
# Use the sidebar to explore the following:
# - 📊 **EDA Dashboard**: Insights into the dataset and Exploratory Data Analysis
# - 🧠 **Treatment Predictor**: MCQ-style quiz to predict if you may need mental health treatment
# """)

# model_path = Path("/Users/nilaysingh/Desktop/Mental-Health-Predictor-App/models/Mental_Health_Prediction_model.cbm")
# model = load_model(model_path)

# st.markdown("""---  
# Made with ❤️ using Streamlit | [GitHub](https://github.com/yourusername/mental_health_app)
# """)


import streamlit as st

from pathlib import Path

# Page config
st.set_page_config(page_title="Mental Health Predictor", layout="wide", page_icon="🧠",initial_sidebar_state="expanded")

with st.sidebar:
    st.markdown("## 🧭 Navigation")
    st.markdown("Welcome to the Mental Health App!")


# Title section
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🧠 Mental Health Predictor App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 20px;'>A Streamlit-powered dashboard for predicting mental health treatment needs using survey data.</p>", unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 18px; line-height: 1.6;'>
Mental health challenges are increasingly prevalent in the tech industry, where high-pressure environments, long hours, and remote work can often contribute to stress and burnout. Despite growing awareness, many individuals may still feel hesitant to acknowledge or address these concerns.
</p>

<p style='text-align: center; font-size: 18px; line-height: 1.6;'>
This project aims to explore patterns in mental well-being through data, uncover trends across various demographics and professional settings, and offer a simple predictive tool to assess whether someone might benefit from mental health support.
</p>

<p style='text-align: center; font-size: 18px; font-style: italic; line-height: 1.6;'>
Please note: This tool is not a substitute for professional diagnosis or treatment. It is designed to promote awareness and encourage reflection.
</p>

<p style='text-align: center; font-size: 18px; line-height: 1.6;'>
We hope you find this application insightful and meaningful — and that it sparks important conversations around mental well-being. 💙
</p>
""", unsafe_allow_html=True)





st.markdown("---")

# Info cards
with st.container():
    col1, col2 ,col3= st.columns(3)

    with col1:
        st.subheader("📊 EDA Dashboard")
        st.write(
            """
            Dive into the dataset and uncover key trends and insights through visualizations.
            Explore relationships between occupation, region, mental health history, and more.
            """
        )
        
    
    with col2:
        st.subheader("🧠 Treatment Predictor")
        st.write(
            """
            Take a quick, multiple-choice quiz to assess whether you may need mental health support.
            Your inputs are processed through a machine learning model trained on real survey data.
            """
        )

    with col3:
        st.subheader("Model Analysis")
        st.write(
            """
            Dive into the trained model to understand how it analyzes mental health patterns.
            Explore the features, performance metrics, and ethical considerations behind its predictions. 
            """
        )

# Call to action
st.markdown("---")
st.markdown("<h3 style='text-align: center;'>🚀 Ready to begin?</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Use the sidebar to navigate through the app.</p>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("<h3 style='text-align: center;'>🤝 Let's Connect Through Socials</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        "<div style='text-align: center;'>"
        "<a href='https://github.com/rishi-18' target='_blank'>"
        "<img src='https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white'>"
        "</a></div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        "<div style='text-align: center;'>"
        "<a href='https://www.linkedin.com/in/rishi-gupta-8113531b3/' target='_blank'>"
        "<img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white'>"
        "</a></div>",
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <p style='text-align: center; font-size: 14px;'>Made with ❤️ using Streamlit |
    <a href='https://github.com/n1lays1ngh/Mental-Health-Predictor-App' target='_blank'>GitHub Repo</a></p>
    """,
    unsafe_allow_html=True
)

