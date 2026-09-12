from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model import DEFAULT_FEATURES, load_data, predict_student, train_and_evaluate

st.set_page_config(page_title="Student Success Suite", page_icon="🎓", layout="wide")

st.title("🎓 Student Success Suite")
st.caption("Explainable academic analytics for student-performance and risk insights")

DATA_PATH = ROOT / "sample_students.csv"

@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_resource
def get_model(df: pd.DataFrame):
    return train_and_evaluate(df, DEFAULT_FEATURES)


df = get_data()
results = get_model(df)

with st.sidebar:
    st.header("Project")
    st.write(
        "This portfolio application demonstrates an end-to-end academic analytics workflow: "
        "data exploration, predictive modelling and interpretable model outputs."
    )
    st.info(
        "The included dataset is a small demonstration dataset. Predictions are illustrative "
        "and should not be used for real academic or high-stakes decisions."
    )

st.subheader("Dataset overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", len(df))
c2.metric("Pass rate", f"{df['passed'].mean() * 100:.0f}%")
c3.metric("Average grade", f"{df['final_grade'].mean():.1f}")
c4.metric("Average prior GPA", f"{df['gpa_prev'].mean():.2f}")

with st.expander("View sample data"):
    st.dataframe(df, use_container_width=True)

st.subheader("Exploratory insights")
col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(
        df,
        x="study_hours",
        y="final_grade",
        color=df["passed"].map({1: "Passed", 0: "At risk"}),
        hover_data=["absences", "participation", "gpa_prev"],
        labels={"color": "Outcome"},
        title="Study hours vs final grade",
    )
    st.plotly_chart(fig, use_container_width=True)
with col2:
    summary = (
        df.groupby("passed", as_index=False)[["study_hours", "absences", "participation", "gpa_prev"]]
        .mean()
        .replace({"passed": {0: "At risk", 1: "Passed"}})
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Model evaluation")
m1, m2 = st.columns(2)
m1.metric("Test accuracy", f"{results.metrics['accuracy']:.2f}")
auc = results.metrics["roc_auc"]
m2.metric("ROC AUC", "N/A" if auc is None else f"{auc:.2f}")

st.write("Classification report")
st.dataframe(results.report.round(3), use_container_width=True)

st.subheader("Explainability")
st.write(
    "The chart below ranks transformed features by the absolute magnitude of the logistic-regression "
    "coefficient. Larger values indicate greater influence on the model's predictions in this demo."
)
importance = results.feature_importance.head(12).copy()
importance["direction"] = importance["coefficient"].apply(lambda x: "Higher pass likelihood" if x > 0 else "Lower pass likelihood")
fig_imp = px.bar(
    importance.sort_values("importance"),
    x="importance",
    y="feature",
    orientation="h",
    color="direction",
    hover_data=["coefficient"],
    title="Top model drivers",
)
st.plotly_chart(fig_imp, use_container_width=True)

st.subheader("Interactive student scenario")
st.write("Adjust the inputs to explore how the model responds to a hypothetical student profile.")

left, right = st.columns(2)
with left:
    study_hours = st.slider("Study hours", 0.0, 10.0, 3.0, 0.5)
    absences = st.slider("Absences", 0, 20, 2)
    participation = st.slider("Participation score", 0, 10, 7)
    past_failures = st.slider("Past failures", 0, 5, 0)
    gpa_prev = st.slider("Previous GPA", 0.0, 4.0, 3.0, 0.1)
with right:
    parent_edu = st.selectbox("Parent education", ["none", "highschool", "bachelor", "master"], index=2)
    internet = st.selectbox("Internet access", ["yes", "no"])
    gender = st.selectbox("Gender", ["F", "M"])

student = {
    "study_hours": study_hours,
    "absences": absences,
    "parent_edu": parent_edu,
    "internet": internet,
    "gender": gender,
    "participation": participation,
    "past_failures": past_failures,
    "gpa_prev": gpa_prev,
}

pred, prob = predict_student(results.pipeline, student)

if pred == 1:
    st.success(f"Illustrative model outcome: likely to pass — estimated probability {prob:.1%}")
else:
    st.warning(f"Illustrative model outcome: higher academic-risk signal — estimated pass probability {prob:.1%}")

st.caption(
    "Responsible-use note: this demo is educational. Real student-support systems require larger representative datasets, "
    "fairness testing, governance, human review and privacy safeguards."
)
