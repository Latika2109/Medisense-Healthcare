"""Module 2 - WHAS500 one-year mortality classifier (Exp 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sksurv.column import encode_categorical
from sksurv.datasets import load_whas500

from .config import SEED

# `los` (hospital length of stay) is dropped: patients who die in hospital have short
# stays, so it leaks the outcome and would not be known at admission.
DROP = ["los"]
HORIZON_DAYS = 365

FEATURE_HELP = {
    "age": "Age (years)", "gender": "Sex (1 = female)", "hr": "Heart rate (bpm)",
    "sysbp": "Systolic BP (mmHg)", "diasbp": "Diastolic BP (mmHg)", "bmi": "BMI (kg/m²)",
    "afb": "Atrial fibrillation (1 = yes)", "av3": "Complete heart block (1 = yes)",
    "chf": "Congestive heart failure (1 = yes)", "cvd": "History of cardiovascular disease (1 = yes)",
    "miord": "MI order (1 = recurrent)", "mitype": "MI type (1 = Q-wave)", "sho": "Cardiogenic shock (1 = yes)",
}


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Features (numeric-encoded, ``los`` removed) and label: death within 365 days."""
    X, y = load_whas500()
    X = encode_categorical(X).drop(columns=DROP)
    X.columns = [c.split("=")[0] for c in X.columns]  # 'afb=1' -> 'afb'
    label = ((y["fstat"]) & (y["lenfol"] <= HORIZON_DAYS)).astype(int)
    return X.astype(float), np.asarray(label)


def candidate_models() -> dict[str, Pipeline]:
    """Class-weighted classifiers (imbalance handled via ``class_weight='balanced'``)."""
    return {
        "Logistic Regression": Pipeline([
            ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("scale", StandardScaler().set_output(transform="pandas")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))]),
        "Random Forest": Pipeline([
            ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                           random_state=SEED, n_jobs=1))]),
    }
