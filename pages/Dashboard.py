import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
from utils import load_data,display_feature_insight,display_treatment_feature
import altair as alt

st.set_page_config(page_title=" Mental Health Predictor Dashboard", layout="wide")
# st.header(" Mental Health Predictor App - EDA Dashboard",divider='blue')
st.markdown("<h1> Mental Health Predictor App - EDA Dashboard </h1>",unsafe_allow_html=True)
# st.divider()

original_data_path = Path("Data/Mental Health Dataset.csv")
original_data = load_data(original_data_path)

cleaned_data_path = Path("Data/cleaned_dataset.csv")
cleaned_data = load_data(cleaned_data_path)

st.write("This Dashboard provides insights to the Exploratory Data Analysis on the Mental Heath Dataset on Kaggle ")
# st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Dataset Preview" , divider="red")
st.markdown("<br>",unsafe_allow_html=True)
st.write("Original Dataset")

with st.expander("View Original Data"):
    st.dataframe(original_data)

st.markdown("<br>",unsafe_allow_html=True)

st.write("Dataset after EDA")

with st.expander("View Data"):
    st.dataframe(cleaned_data)
 
st.markdown("<br>",unsafe_allow_html=True)


### DATA CLEANING 

st.subheader("Data Cleaning",divider="red")

st.markdown("<h4> Handling Missing Values</h4>", unsafe_allow_html=True)

with st.expander("View Missing Values"):
    st.dataframe(original_data.isnull().sum())

st.write("Since 'Self-employed' is an important predictor, we imputed its missing values by introducing a new category labeled 'Missing'. This allows the model to learn from the absence of data as a separate class.  ")

st.code("df['self_employed'] = df['self_employed'].fillna('Missing')" , language="python")

st.write("The 'Timestamp' feature was removed as it does not contribute meaningful information to the prediction task. ")

st.code("df = df.drop('Timestamp',axis = 1)")

st.markdown("<h4>Duplicate Entries</h4>", unsafe_allow_html=True)

st.markdown("""
Although duplicates are usually removed during cleaning, in **Mental health analysis**, repeated responses can capture shifts in an individual's condition over time.

**We kept duplicate entries** to retain potentially valuable patterns and variations.
""")
st.markdown("<br>",unsafe_allow_html=True)

### FEATURE ENGINEERING

st.subheader("Feature Engineering",divider="red")
st.write("As part of feature engineering, we modified two existing features by creating meaningful new categories, enhancing their utility for downstream modeling")
#Country to Continent
st.markdown("<h4> Country to Continent Conversion </h4>",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    country_counts = original_data['Country'].value_counts()
    st.bar_chart(country_counts,y_label="Frequency")
    st.markdown("<h6 style='text-align: center;'>Country Distribution</h6>", unsafe_allow_html=True)

with col2:
    continent_counts = cleaned_data['Continent'].value_counts()
    st.bar_chart(continent_counts,y_label="Frequency")
    st.markdown("<h6 style='text-align: center;'>Continent Distribution</h6>", unsafe_allow_html=True)

st.write("""
To reduce the high cardinality of the **Country** feature and simplify geographical grouping, we converted individual countries into broader **Continent** categories.
This makes the data more interpretable and reduces noise in the model.
""")

st.markdown("<br>",unsafe_allow_html=True)
#Occupation to Occupation_categories

st.markdown("<h4> Occupation to Occupation Category</h4>",unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    country_counts = original_data['Occupation'].value_counts().sort_index()
    st.bar_chart(country_counts,y_label="Frequency")
    st.markdown("<h6 style='text-align: center;'>Occupation Distribution</h6>", unsafe_allow_html=True)

with col2:
    continent_counts = cleaned_data['Occupation_Category'].value_counts()
    st.bar_chart(continent_counts,y_label="Frequency")
    st.markdown("<h6 style='text-align: center;'>Category-wise Distribution</h6>", unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

st.write(""" The `Occupation` field contained many distinct values, which can introduce sparsity and overfitting. We grouped them into broader **Professional** and **Non-Professional** categories for better model performance.
""")



### FEATURE ANALYSIS 

st.subheader("Univariate Analysis",divider="red")
st.markdown("<h5> Visualization </h5>",unsafe_allow_html=True)
features = [
    "Gender","self_employed", "family_history",
    "Days_Indoors", "Growing_Stress", "Changes_Habits", "Mental_Health_History",
    "Mood_Swings", "Coping_Struggles", "Work_Interest", "Social_Weakness",
    "mental_health_interview", "care_options", "Continent","Occupation_Category"
]

selected_feature = st.selectbox("Choose a feature to examine its univariate distribution:", features)
feature_counts = cleaned_data[selected_feature].value_counts().sort_index()

feature_df = feature_counts.reset_index()
feature_df.columns = [selected_feature, "Frequency"]
feature_df.set_index(selected_feature, inplace=True)

st.bar_chart(feature_df)

display_feature_insight(selected_feature)

st.subheader("Bivariate Analysis ",divider="red")


st.markdown("<h5> Visualization </h5>",unsafe_allow_html=True)





features = [
    "Gender","self_employed", "family_history",
    "Days_Indoors", "Growing_Stress", "Changes_Habits", "Mental_Health_History",
    "Mood_Swings", "Coping_Struggles", "Work_Interest", "Social_Weakness",
    "mental_health_interview", "care_options", "Continent","Occupation_Category"
]

feature_selected= st.selectbox("Compare 'treatment' with:", features)

grouped = cleaned_data.groupby([feature_selected, 'treatment']).size().unstack(fill_value=0)
grouped.columns = ['No (Treatment)', 'Yes (Treatment)']
grouped_percent = grouped.div(grouped.sum(axis=1), axis=0) * 100



view_mode = st.radio("View mode", ["Counts", "Percentages"], horizontal=True)


if view_mode == "Counts":
    st.bar_chart(grouped)
else:
    st.bar_chart(grouped_percent)

display_treatment_feature(feature_selected)

st.divider()

st.markdown("""
<h4> NOTE:</h4>
<p style='font-size: 16px;'>
In this exploratory data analysis (EDA), we interpret the <strong>'treatment'</strong> feature as a proxy for whether a respondent may need mental health support. 
Specifically, a "Yes" response to the treatment question is considered indicative of an individual having recognized and acted upon their mental health needs by seeking professional help. 
<br><br>
While this may not fully capture all cases of mental distress or undiagnosed conditions, it serves as a consistent and practical baseline for analyzing patterns across demographic, behavioral, and psychological features.
<br><br>
Please keep in mind that this assumption simplifies a complex topic and should be interpreted with appropriate caution.
</p>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<h4> Statistical Association Analysis</h4>
<p style='font-size:16px;'>
To deepen our understanding of the patterns observed in the dataset, we employ two key statistical techniques:
<br><br>
<strong>1. Chi-Square Test of Independence:</strong> This helps us determine whether there is a statistically significant association between the treatment feature and each categorical variable.
<br><br>
<strong>2. Cramér's V:</strong> This metric quantifies the strength of that association on a scale from 0 (no association) to 1 (perfect association).
<br><br>
By combining both the Chi-Square test and Cramér’s V, we can identify which features have the most meaningful influence on treatment-seeking behavior — offering crucial insight for both modeling and intervention design.
</p>
""", unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
st.markdown("<h4>Table Summarizing the Chi-sq and Cramer's V Values</h4>",unsafe_allow_html=True)
chi2_cramers_df = pd.read_csv("Data/chi2_cramersv_summary.csv", index_col=0)


st.dataframe(chi2_cramers_df.style.format({
    "Chi2_Statistic": "{:.2f}",
    "P_Value": "{:.6f}",
    "Cramers_V": "{:.4f}"
}).background_gradient(cmap='YlOrRd', subset=["Cramers_V"]))

st.markdown("<br>",unsafe_allow_html=True)

st.markdown("<h4>Visualizing the Chi-sq and Cramer's V Values</h4>",unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    chi2_cramers_df = chi2_cramers_df.reset_index()
    chi2_cramers_df.rename(columns={'index': 'Feature'}, inplace=True) 
    df_sorted = chi2_cramers_df.sort_values("Chi2_Statistic", ascending=False)

    
    bar_chart = alt.Chart(df_sorted).mark_bar().encode(
        x=alt.X("Chi2_Statistic:Q", title="Chi-Square Value"),
        y=alt.Y("Feature:N", sort='-x', title="Feature"),
        tooltip=["Feature", "Chi2_Statistic", "Degrees_of_Freedom", "P_Value", "Cramers_V"]
    ).properties(width=700, height=500,title = "Chi-Square Test Results")

    st.altair_chart(bar_chart, use_container_width=True)

with col2:
    
    df_sorted2= chi2_cramers_df.sort_values("Cramers_V", ascending=False)

    cramers_chart = alt.Chart(df_sorted2).mark_bar().encode(
        x=alt.X("Cramers_V:Q", title="Cramér's V Value"),
        y=alt.Y("Feature:N", sort='-x', title="Feature"),
        tooltip=[
            alt.Tooltip("Feature:N"),
            alt.Tooltip("Chi2_Statistic:Q", format=".2f"),
            alt.Tooltip("Degrees_of_Freedom:Q"),
            alt.Tooltip("P_Value:Q", format=".4f"),
            alt.Tooltip("Cramers_V:Q", format=".4f")
        ]
    ).properties(width=700, height=500, title="Cramér's V Association Strength")

    st.altair_chart(cramers_chart, use_container_width=True)

st.divider()
st.markdown("<h4>Key Findings from the Bivariate Analysis</h4>",unsafe_allow_html=True)

st.markdown("""
<h5><u>Strong Associations</u></h5>
<ul>
  <li><b>Family History (Cramér's V = 0.3668):</b> Individuals with a family history of mental illness are significantly more likely to seek treatment.</li>
  <li><b>Care Options (V = 0.2937):</b> Access to mental health care is a strong factor influencing treatment-seeking behavior.</li>
</ul>

<h5><u>Moderate Associations</u></h5>
<ul>
  <li><b>Gender (V = 0.1772):</b> Gender shows a moderate relationship, suggesting potential gender-based differences in mental health perceptions or access.</li>
  <li><b>Continent (V = 0.1544):</b> Regional differences moderately impact mental health treatment access or attitudes.</li>
</ul>

<h5><u>Weak but Significant Associations</u></h5>
<ul>
  <li><b>Mental Health Interview, Self-employed, Growing Stress, Coping Struggles</b> show statistically significant relationships but with lower strength (V &lt; 0.1). These may still be meaningful in a multivariate setting.</li>
</ul>

<h5><u>Minimal to No Associations</u></h5>
<ul>
  <li><b>Mood Swings, Days Indoors, Social Weakness, Changes in Habits, Work Interest</b> show very weak or no meaningful association with treatment-seeking behavior.</li>
</ul>
""", unsafe_allow_html=True)

st.divider()

##CONCLUSION 

st.markdown("""
<h3>Conclusion</h3>
<p>
This interactive dashboard presents key insights from an exploratory data analysis (EDA) on the Mental Health in Tech Survey dataset.
It includes:
</p>
<ul>
  <li>Cleaning and preprocessing steps</li>
  <li>Univariate and bivariate visualizations</li>
  <li>Statistical associations between features and mental health treatment-seeking behavior</li>
</ul>

<p>Use this analysis to better understand the dataset before diving into predictive modeling.</p>
""", unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

st.write("For further exploration of the project, please refer to the links below: ")
st.page_link("pages/Model_Analysis.py", label=" Model Analysis ", icon = "📊")

st.page_link("pages/Mental_Health_Prediction_App.py",label = "Mental Health Prediction App",icon = "🔮")



### SIDEBAR ###

st.sidebar.title("EDA Dashboard ")
st.sidebar.subheader("Sections")

with st.sidebar.expander("Dataset Preview"):
    st.write("- Original Dataset")
    st.write("- Cleaned Dataset")

with st.sidebar.expander("Dataset Cleaning"):
    st.markdown("- Handling Missing Values")
    st.markdown("- Duplicate Entries")

with st.sidebar.expander("Feature Engineering"):
    st.markdown("- Country to Continent Conversion")
    st.markdown("- Occupation to Occupation Category")

with st.sidebar.expander("Univariate Analysis"):
    st.markdown("- Visualization")
    st.markdown("- Interpretation")

with st.sidebar.expander("Bivariate Analysis"):
    st.markdown("- Comparision with Treatment")
    st.markdown("- Statistical Association Analysis")
    st.markdown("- Key Findings from the Bivariate Analysis")

with st.sidebar.expander("Conclusion"):
    st.markdown("- Links ")
    
