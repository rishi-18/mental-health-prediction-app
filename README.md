# 🧠 Mental Health Predictor App

An interactive Streamlit dashboard that explores mental health trends in the tech industry and uses a machine learning model to predict whether an individual might benefit from mental health support.

---

## 📌 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Screenshots](#screenshots)
5. [Setup Instructions](#setup-instructions)
6. [Model Information](#model-information)
7. [Disclaimer](#disclaimer)
8. [Contributing](#contributing)
9. [License](#license)
10. [Connect](#connect)

---

## 🧩 Overview

Mental health issues are increasingly common in the tech industry, but they often go unnoticed or unspoken. This app aims to:
- Visualize insights from survey data
- Help individuals self-reflect through a quiz-based predictor
- Raise awareness around mental well-being in professional environments

---
## 🚀 Explore the Mental Health Predictor App

You can interact with the full application, including the predictive quiz and exploratory dashboards, by visiting the link below:

[🔗 Mental Health Predictor App](https://mental-health-predictor-app-n1lays1ngh.streamlit.app/)

This app allows you to:
- Review detailed exploratory data analysis (EDA) on the Mental Health in Tech Survey dataset
- Understand key patterns and factors influencing treatment-seeking behavior
- Try out a predictive model trained using CatBoost to assess mental health tendencies based on user input

---
## 🚀 Features
- 📊 EDA Dashboard with demographic and mental health visualizations
- 🧠 Quiz-based Treatment Predictor using CatBoost
- 🔍 Feature Importance and Model Transparency
- 🌐 Responsive UI with Streamlit and Tailwind-style theming

---

## 🛠️ Tech Stack

| Layer         | Tool/Library                     |
|--------------|----------------------------------|
| Frontend     | Streamlit                        |
| Backend      | Python (CatBoost, Pandas)        |
| ML Model     | CatBoost Classifier (.cbm file)  |
| Visualization| Plotly, Seaborn, Matplotlib      |

---

## 📸 Screenshots
### 1. Homepage 
![Homepage ](<assets/Homepage Screenshot.png>)

### 2. Dashboard 
![Dashboard](<assets/Dashboard Screenshot.png>)

### 3. Model Analysis 
![Model Analysis ](<assets/Model Analysis Screenshot.png>)

### 4. Predictor App
![Predictor App ](<assets/Predictor App Screenshot.png>)

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/mental-health-predictor-app.git
cd mental-health-predictor-app

# 2. Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📊 Model Information

- **Model Used**: CatBoost Classifier
- **Why CatBoost?** Handles categorical data, offers interpretability, and works well with class imbalance.
- **Top Features**: Continent, Care Options, Family History, Gender, etc.
- **Performance**: F1-Score ~0.74, Recall (Needs Treatment) = 0.83

---

## ⚠️ Disclaimer

> This tool is **not** a replacement for medical or psychological diagnosis. It is only meant for awareness and educational purposes.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 🪪 License

MIT License. See [`LICENSE`](LICENSE) file.

---

## 🌐 Connect

[![GitHub](https://img.shields.io/badge/GitHub-000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername)
&nbsp;
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yourusername)
