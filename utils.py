import pandas as pd
from catboost import CatBoostClassifier
from pathlib import Path
import streamlit as st 

def load_model(model_path:Path) -> CatBoostClassifier:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model

def load_data(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def preprocess_input(user_answers: dict, feature_order: list) -> pd.DataFrame:
    """
    user_answers: dict of feature_name -> selected_option (e.g., {'Gender': 'Male', ...})
    feature_order: list of model's expected input features (ordered)
    Returns: DataFrame with one row ready for prediction
    """
    df_input = pd.DataFrame([user_answers])
    df_input = df_input.reindex(columns=feature_order)
    return df_input


feature_insights = {
    "Gender": [
        "The dataset is heavily skewed towards Male respondents.",
        "Female representation is much lower.",
        "No indication of missing or ‘Other’ categories in the chart."
    ],
    "self_employed": [
        "A vast majority of respondents reported not being self-employed.",
        "Very few individuals are self-employed, indicating a largely employed or dependent population base."
    ],
    "family_history": [
        "More respondents do not have a family history of mental illness than those who do.",
        "However, the proportion of Yes responses is notable and not negligible — suggesting family history might still be a significant factor."
    ],
    "Days_Indoors": [
        "Distribution is fairly balanced.",
        "Slightly higher counts for people spending 2-4 days and 5-7 days indoors per week.",
        "Lower frequencies for 1-2 days and 0 days, indicating most people spend multiple days indoors, which may correlate with remote working or social withdrawal factors."
    ],
    "Growing_Stress": [
        "Responses are relatively evenly split among Yes, No, and Maybe.",
        "Indicates a diverse experience of stress among respondents, with no overwhelming majority in any category."
    ],
    "Changes_Habits": [
        "Similar balanced distribution as 'Growing Stress'.",
        "Consistent spread of Yes, No, and Maybe, indicating habit changes are neither uncommon nor universal."
    ],
    "Mental_Health_History": [
        "Slightly balanced — a good proportion of respondents reported having a personal mental health history.",
        "Maybe and No responses are also present in significant numbers.",
        "This suggests varied past experiences with mental health among the population."
    ],
    "Mood_Swings": [
        "Nearly even distribution.",
        "Mood swings seem to be a fairly common experience across the dataset."
    ],
    "Coping_Struggles": [
        "Predominantly Maybe responses.",
        "This suggests a large number of individuals are uncertain or occasionally experience difficulty coping — a valuable insight for mental health support programs."
    ],
    "Work_Interest": [
        "Balanced distribution with a slight tilt towards Maybe.",
        "Indicates varying degrees of work interest levels, potentially affected by stress, job satisfaction, or mental well-being."
    ],
    "Social_Weakness": [
        "Responses are well distributed across Yes, No, and Maybe.",
        "Suggests social confidence and connection levels differ notably among individuals."
    ],
    "mental_health_interview": [
        "The majority of respondents have not attended a mental health interview.",
        "Very low numbers for Yes and Maybe — pointing towards potential underutilization of professional mental health services."
    ],
    "care_options": [
        "Fairly balanced spread across all three categories.",
        "Indicates a mixed awareness or availability of care options among respondents."
    ],
    "Continent": [
        "The dataset is highly concentrated in Africa.",
        "Smaller representations from Europe, Oceania, Asia, and America.",
        "Important to note potential geographic bias in mental health trends analysis."
    ],
    "Occupation_Category": [
        "Reasonably balanced between Professional and Non-Professional categories.",
        "Useful for analyzing occupation-based differences in mental health outcomes."
    ],
    "treatment": [
        "Near-equal numbers of individuals reported Yes and No to seeking mental health treatment.",
        "This suggests half the dataset has pursued professional help, while half have not — offering a balanced ground for predictive modeling and intervention analysis."
    ]
}


def display_feature_insight(feature_name):
    insight = feature_insights.get(feature_name, "No insight available for this feature.")
    st.markdown("<h5> Interpretation: </h5>",unsafe_allow_html=True)
    for point in insight:
        st.write(f"- {point}")

treatment_feature_insights = {
    "Gender": [
        "Male respondents dominate both the ‘Yes’ and ‘No’ treatment categories.",
        "Proportionally, a slightly higher number of females seek treatment compared to males when adjusted for sample size.",
        "However, overall male counts remain much higher."
    ],
    "Occupation_Category": [
        "Professional and Non-Professional groups have nearly equal numbers for both treatment outcomes.",
        "No significant difference in treatment-seeking behavior between occupation categories."
    ],
    "self_employed": [
        "Majority of both self-employed and non-self-employed respondents did not seek treatment.",
        "Fewer self-employed individuals seek treatment, but this aligns with their lower overall count."
    ],
    "family_history": [
        "Respondents with a family history of mental illness are far more likely to have sought treatment.",
        "Clear positive association between family history and treatment-seeking behavior."
    ],
    "Days_Indoors": [
        "Very little variation in treatment rates across different indoor time categories.",
        "Treatment-seeking appears fairly consistent regardless of how many days people stay indoors."
    ],
    "Growing_Stress": [
        "People reporting ‘Yes’ to growing stress are slightly more likely to seek treatment.",
        "Those answering ‘Maybe’ have noticeably lower treatment-seeking rates."
    ],
    "Changes_Habits": [
        "Fairly balanced distribution between ‘Yes’ and ‘No’ treatment across all habit change responses.",
        "Indicates that changes in habits alone might not be a strong driver for treatment-seeking."
    ],
    "Mental_Health_History": [
        "Respondents with a mental health history are significantly more likely to seek treatment.",
        "Strong positive relationship between past mental health experiences and treatment behavior."
    ],
    "Mood_Swings": [
        "Consistent pattern across ‘Yes’, ‘No’, and ‘Maybe’ mood swing levels.",
        "Slight increase in treatment rates among those reporting mood swings."
    ],
    "Coping_Struggles": [
        "People struggling to cope (Yes) are slightly more inclined to seek treatment.",
        "Not a very strong effect, but it is noticeable."
    ],
    "Work_Interest": [
        "Fairly balanced treatment distribution across work interest levels.",
        "People answering ‘Maybe’ or ‘No’ seem to seek treatment slightly less, but the difference is small."
    ],
    "Social_Weakness": [
        "Very balanced treatment rates across all levels of social weakness.",
        "No evident trend suggesting social weakness directly affects treatment-seeking behavior."
    ],
    "mental_health_interview": [
        "Strong association: respondents who had a mental health interview are much more likely to have sought treatment.",
        "Most people who did not attend an interview also did not seek treatment."
    ],
    "care_options": [
        "Respondents aware of or having care options (Yes) have higher treatment-seeking rates.",
        "Those answering ‘No’ or ‘Maybe’ show lower treatment rates — suggesting awareness and access to care matters."
    ],
    "Continent": [
        "African respondents dominate the dataset, with substantial representation in both Yes and No categories.",
        "Other continents have fewer respondents, and treatment rates are relatively balanced.",
        "Too little data from Europe, Oceania, Asia, and America to draw strong conclusions."
    ]
}

def display_treatment_feature(feature_name):
    insights = treatment_feature_insights.get(feature_name, ["No treatment analysis available for this feature."])
    st.markdown(f"<h4>Analysis of Treatment vs {feature_name}</h4>",unsafe_allow_html=True)
    for point in insights:
        st.write(f"- {point}")
