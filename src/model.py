from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "passed"
DEFAULT_FEATURES = [
    "study_hours",
    "absences",
    "parent_edu",
    "internet",
    "gender",
    "participation",
    "past_failures",
    "gpa_prev",
]


@dataclass
class ModelResults:
    pipeline: Pipeline
    metrics: dict[str, float | None]
    report: pd.DataFrame
    confusion: np.ndarray
    feature_importance: pd.DataFrame


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(DEFAULT_FEATURES + [TARGET])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df.copy()


def _build_pipeline(df: pd.DataFrame, features: Iterable[str]) -> Pipeline:
    features = list(features)
    numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in features if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    features: Iterable[str] = DEFAULT_FEATURES,
    test_size: float = 0.30,
    random_state: int = 42,
) -> ModelResults:
    features = list(features)
    clean = df.dropna(subset=features + [TARGET]).copy()
    X = clean[features]
    y = clean[TARGET].astype(int)

    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    pipeline = _build_pipeline(clean, features)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    auc = None
    if y_test.nunique() == 2:
        auc = float(roc_auc_score(y_test, probs))

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": auc,
        "test_rows": float(len(y_test)),
    }

    report = pd.DataFrame(classification_report(y_test, preds, output_dict=True, zero_division=0)).T
    confusion = confusion_matrix(y_test, preds)

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    feature_importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "coefficient": coefficients,
                "importance": np.abs(coefficients),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return ModelResults(
        pipeline=pipeline,
        metrics=metrics,
        report=report,
        confusion=confusion,
        feature_importance=feature_importance,
    )


def predict_student(pipeline: Pipeline, student: dict) -> tuple[int, float]:
    row = pd.DataFrame([student])
    pred = int(pipeline.predict(row)[0])
    prob = float(pipeline.predict_proba(row)[0, 1])
    return pred, prob
