import streamlit as st
import pandas as pd
import joblib
from utils import preprocess_input ,load_model

model_path = "models/Mental_Health_Prediction_model2.cbm"

model = load_model(model_path)


st.set_page_config(
    page_title="Mental Health Prediction App",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Mental Health Prediction App")

st.markdown("""
<h5>This quiz is designed to assess your mental health state based on a predictive model trained on the 
<a href="https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey" target="_blank">Mental Health in Tech Survey</a> dataset from Kaggle.</h5>
""", unsafe_allow_html=True)

st.markdown("""
<h4> Disclaimer</h4>
<p style='font-size:16px;'>
This application provides a predictive insight based on the Mental Health in Tech Survey dataset and a machine learning model trained on it.
<br><br>
<b>It is not intended to provide a medical diagnosis or replace professional mental health services.</b> The predictions offered here are statistical in nature and should be interpreted with caution.
If you are experiencing mental health issues, we strongly encourage you to seek guidance from a qualified mental health professional.
</p>
""", unsafe_allow_html=True)

feature_order = [
    'Gender',
    'self_employed',
    'family_history',
    'Mental_Health_History',
    'mental_health_interview',
    'care_options',
    'Continent',
    'Occupation_Category'
]

gender_options = ["Male", "Female", "Other"]
yes_no = ["Yes", "No"]
continent_options = ["Asia", "Europe", "North America", "South America", "Africa", "Oceania", "Antarctica"]
occupation_options = ["Professional","Non-professional"]


with st.form("mental_health_quiz"):
    gender = st.selectbox("What is your gender?", gender_options)
    self_employed = st.selectbox("Are you self-employed?", yes_no)
    family_history = st.selectbox("Any family history of mental illness?", yes_no)
    personal_history = st.selectbox("Do you have a history of mental health issues?", yes_no)
    comfortable_interview = st.selectbox("Are you comfortable discussing mental health in interviews or talking openly about it ?", yes_no)
    care_options = st.selectbox("Are you aware of mental health care options available at your workplace? ", yes_no)
    continent = st.selectbox("Which continent are you located in?", continent_options)
    occupation = st.selectbox("Which category best describes your occupation?", occupation_options)

    submitted = st.form_submit_button("Predict")

    if submitted:
        user_answers = {
            'Gender': gender,
            'self_employed': self_employed,
            'family_history': family_history,
            'Mental_Health_History': personal_history,
            'mental_health_interview': comfortable_interview,
            'care_options': care_options,
            'Continent': continent,
            'Occupation_Category': occupation
        }

        # Preprocess input using utility function
        input_df = preprocess_input(user_answers, feature_order)
        
        # # Make prediction
        # prediction = model.predict(input_df)[0]

        # # Display result
        # if prediction == 1:
        #     st.success(" Based on your responses, you may benefit from mental health support. Consider seeking professional help.")
        # else:
        #     st.info(" You're not currently flagged for needing assistance, but staying mindful and proactive is always helpful.")

        prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of class 1
        prediction = int(prediction_proba >= 0.5)
        confidence_percent = prediction_proba * 100

        # --- Display Results ---
        st.markdown("## 🧠 Prediction Result")

        if prediction == 1:
            st.success("🩺 **Prediction:** You may benefit from **mental health treatment**.")
        else:
            st.info("✅ **Prediction:** No immediate need for mental health treatment detected.")

        st.markdown("### 🔍 Model Confidence")
        st.metric(label="Likelihood of Needing Treatment", value=f"{confidence_percent:.2f}%")
        st.progress(prediction_proba)

