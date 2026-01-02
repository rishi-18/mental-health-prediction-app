import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from utils import load_model,load_data
import altair as alt
st.set_page_config(
    page_title="Model Analysis Dashboard",
    layout="wide"
)

st.title("Model Analysis")
st.write("This page provides a detailed explanation of the predictive model developed and trained to power the Mental Health Predictor application.")

st.subheader(" Model Choice and Rationale",divider="red")
st.markdown("<h5>Predictive Model Used: CatBoost Classifier </h5>",unsafe_allow_html=True)
st.markdown("""
We chose **CatBoost Classifier** for its ability to:

- Handle **categorical features natively** (no encoding needed)
- Perform well on **imbalanced datasets**
- Support **early stopping** to avoid overfitting
- Deliver **fast and interpretable** results

This made it ideal for our dataset, which consists entirely of categorical variables.
""")


st.subheader(" Feature Importance Analysis",divider="red")

model_path = "models/Mental_Health_Prediction_model2.cbm"

model = load_model(model_path)

X_train = load_data("Data/X_train2.csv")

feature_names = model.feature_names_
importances = model.get_feature_importance()

# Create DataFrame of feature importances
feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

# Take top 20 for visualization
feat_df_top20 = feat_df.head(20)
# chart_height = 20 * len(feat_df_top20)

# Altair horizontal bar chart
chart = alt.Chart(feat_df_top20).mark_bar(color='steelblue').encode(
    x=alt.X('Importance:Q'),
    y=alt.Y('Feature:N', sort='-x'),
    tooltip=['Feature:N', 'Importance:Q']
).properties(
    width=700,
    height=600,
    title=' Top 8 Feature Importances (CatBoost)'
)

# Display in Streamlit
st.altair_chart(chart, use_container_width=True)

st.markdown("""
            ###  Top Predictive Features

The most influential features identified by the model include:

- **Continent**  
- **Care options**  
- **Self-employed status**  
- **Comfort with mental health interviews**  
- **Family history**  
- **Gender**



###  Dropped Features

The following features were excluded due to low predictive importance and weak association with the target variable (as indicated by low Cramér's V scores):

`Changes_Habits`, `Social_Weakness`, `Work_Interest`, `Mood_Swings`, `Growing_Stress`, `Coping_Struggles`, `Days_Indoors`
""")

st.subheader("Performance Metrics Observed During Model Training",divider="red")
col1 , col2 = st.columns(2)

df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'] * 2,
    'Score': [0.79, 0.65, 0.72, 0.71, 0.83, 0.77],
    'Class': ['No'] * 3 + ['Yes'] * 3
})

df_avg = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score'] * 2,
    'Score': [0.75, 0.74, 0.74, 0.75, 0.74, 0.74],
    'Type': ['Macro Avg'] * 3 + ['Weighted Avg'] * 3
})


with col1:
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Metric:N', title=None),
        y=alt.Y('Score:Q', title='Score'),
        color='Class:N',
        column=alt.Column('Class:N', title=''),
        tooltip=['Metric', 'Score', 'Class']
    ).properties(width=150, height=300)

    st.altair_chart(chart)

with col2:
    avg_chart = alt.Chart(df_avg).mark_bar().encode(
        x=alt.X('Metric:N', title=None),
        y=alt.Y('Score:Q', title='Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('Type:N', scale=alt.Scale(scheme='set2')),
        column=alt.Column('Type:N', title=''),
        tooltip=['Metric', 'Score', 'Type']
    ).properties(width=150, height=300)

    st.altair_chart(avg_chart)


st.markdown("""###  Model Evaluation Summary

The model demonstrates balanced performance across both classes with high scores in key metrics:

- **Class 1 (Needs Treatment)** shows a higher **recall (0.83)**, indicating the model is effective at correctly identifying individuals likely to need mental health support—crucial in a healthcare context where missing such cases can have serious consequences.
- **Class 0 (No Treatment Needed)** has a higher **precision (0.79)**, meaning fewer false positives for those predicted as not needing treatment.

The **Macro and Weighted Average metrics** (Precision ~0.75, Recall ~0.74, F1 ~0.74) are closely aligned, suggesting:

- The model performs **consistently well across both classes**.
- The dataset is **fairly balanced**, and the model is **not biased toward any particular class**.

Overall, this indicates a **well-calibrated model** that prioritizes correctly identifying those in need of mental health treatment, aligning with the real-world goal of minimizing false negatives in a sensitive domain.
""")


st.subheader("Conclusion",divider='red')
st.markdown("""
            - **Strong Predictive Power**  
  The model excels at identifying individuals likely to require mental health treatment, achieving high recall for the “Yes” class — minimizing false negatives in critical scenarios.

- **Balanced Evaluation**  
  Metrics like precision, recall, and F1-score remain consistent across both classes. Macro and weighted averages show the model performs fairly even with class imbalance.

- **Relevant Features Identified**  
  Key features such as continent, care options, employment status, family history, and gender significantly contribute to accurate predictions.

- **Noise Reduction**  
  Features with low predictive value or weak statistical association were dropped, helping reduce overfitting and improve clarity.

- **Ethically Aligned**  
  Favoring high recall for the positive class supports early intervention and aligns with best practices in mental health screening.
""")

st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<h4>Ready to test it out ?</h4>",unsafe_allow_html=True)


st.page_link("pages/Mental_Health_Prediction_App.py", label="🔗 Test the Mental Health Prediction App")


#Sidebar

st.sidebar.title("Model Analysi")
st.sidebar.subheader("Sections")

with st.sidebar.expander("Model Choice and Rationale"):
    st.write("- Predictive Model Used")


with st.sidebar.expander("Feature Importance Analysis"):
    st.markdown("- Visualization")
    st.markdown("- Top Predictive Features")
    st.markdown("- Dropped Features")

with st.sidebar.expander("Performance Metrics Observed During Model Training"):
    st.markdown("- Visualization")
    st.markdown("- Model Evaluation Summary")


with st.sidebar.expander("Conclusion"):
    st.markdown("- Conclusion")
    st.markdown("- Links")