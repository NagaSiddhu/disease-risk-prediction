"""Prediction helpers (no Streamlit code here, so they are easy to test)."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).parent


def load_bundle(path=None):
    return joblib.load(path or HERE / "model.joblib")


def predict_risk(bundle, df: pd.DataFrame) -> np.ndarray:
    """Probability of HIGH risk for every row of df."""
    return bundle["model"].predict_proba(df[bundle["features"]])[:, 1]


def member_probabilities(bundle, row: dict) -> dict:
    """Probability of high risk given by each model inside the ensemble (for one patient)."""
    X = pd.DataFrame([row])[bundle["features"]]
    return {name: float(est.predict_proba(X)[0, 1])
            for name, est in bundle["model"].named_estimators_.items()}


def check_columns(bundle, df: pd.DataFrame):
    """Return the list of required columns that are missing from an uploaded CSV."""
    return [c for c in bundle["features"] if c not in df.columns]


def label_for(prob: float, threshold: float = 0.5) -> str:
    return "HIGH RISK" if prob >= threshold else "LOW RISK"
