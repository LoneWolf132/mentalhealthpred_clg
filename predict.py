import os

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_COLUMNS = [
    "gender",
    "age",
    "academic_pressure",
    "work_pressure",
    "study_satisfaction",
    "job_satisfaction",
    "work_study_hours",
    "financial_stress",
    "cgpa",
    "sleep_duration",
    "dietary_habits",
    "family_history",
    "suicidal_thoughts"
]

def depression_level(prob):
    """
    Statistical severity ONLY.
    This is NOT a clinical or safety decision.
    """
    if prob < 0.25:
        return "None"
    elif prob < 0.45:
        return "Mild"
    elif prob < 0.65:
        return "Moderate"
    else:
        return "Severe"


def predict_depression(input_df: pd.DataFrame) -> dict:
    """
    ML-only prediction.
    No overrides, no ethics, no text interpretation.
    """
    pipeline = joblib.load(os.path.join(BASE_DIR, "depression_pipeline.pkl"))

    # Enforce schema order and presence
    input_df = input_df.reindex(columns=FEATURE_COLUMNS)

    prob = pipeline.predict_proba(input_df)[0][1]
    ml_severity = depression_level(prob)

    return {
        "probability": prob,
        "ml_severity": ml_severity  # renamed for clarity
    }
