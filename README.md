# Student Success Suite

An academic analytics portfolio project that uses **data analysis, predictive modelling and explainable machine learning** to explore student-performance patterns and support data-informed student-success decisions.

## What the project does

The repository now contains a working end-to-end demo that:

- loads and validates student data;
- explores behavioural and academic indicators;
- trains a classification model to estimate pass/risk outcomes;
- evaluates model performance with classification metrics;
- ranks model drivers using interpretable logistic-regression coefficients;
- provides an interactive Streamlit interface for exploring hypothetical student scenarios;
- includes a responsible-use note for education and high-stakes decision contexts.

## Project structure

```text
Student-Success-Suite/
├── app/
│   └── streamlit_app.py
├── src/
│   └── model.py
├── sample_students.csv
├── requirements.txt
├── student_success_suite_optionA.zip
└── README.md
```

## Technical stack

**Python · Pandas · NumPy · Scikit-learn · Plotly · Streamlit · Machine Learning · Data Analytics · Explainable AI · Data Visualisation**

## Run locally

```bash
git clone https://github.com/uma-27/Student-Success-Suite.git
cd Student-Success-Suite
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Modelling approach

The demo uses a preprocessing pipeline that standardises numeric variables and one-hot encodes categorical variables before fitting a logistic-regression classifier. The current feature set includes study hours, absences, parental education, internet access, gender, participation, past failures and previous GPA.

The application reports test accuracy, ROC AUC when available, a classification report and an interpretable feature-influence view based on model coefficients.

## Responsible use

The included CSV is a **small demonstration dataset**, so the model should not be interpreted as a validated student-risk system. A real deployment would require substantially larger and representative data, fairness and bias testing, privacy controls, governance, model monitoring and human review.

## Why this project matters

Student data can provide more value than static grades alone. This project demonstrates how analytics and transparent machine learning can be combined to identify patterns, communicate model behaviour and explore earlier support signals without treating AI predictions as unquestionable decisions.

## Current status

**Working portfolio prototype.** The core analytics and interactive application are implemented. Future extensions could include larger datasets, cross-validation, fairness metrics, SHAP-based local explanations, authentication and deployment.

## About me

I am **Asritha Adari**, completing postgraduate Computer Science studies at the University of Sydney, with interests in Data Analytics, AI, Machine Learning and practical technology solutions.

- GitHub: https://github.com/uma-27
- LinkedIn: https://www.linkedin.com/in/uma-sree-asritha-adari-0b2210231/
