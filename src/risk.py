"""Module 1 - Diabetes risk (Exp 6): data, pipelines, CV, calibration, fairness."""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, PIMA_URL, SEED
from .metrics import binary_metrics

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
           "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
FEATURES = COLUMNS[:-1]
CV_SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
              "f1": "f1", "roc_auc": "roc_auc"}


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace physiologically impossible zeros with NaN in selected columns."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ZeroToNaN":
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.columns if self.columns is not None else ZERO_AS_MISSING
        for c in cols:
            X[c] = X[c].replace(0, np.nan)
        return X


def load_pima() -> pd.DataFrame:
    """Load Pima dataset (cached in data/, downloaded on first use)."""
    path = DATA_DIR / "pima-indians-diabetes.csv"
    if path.exists():
        df = pd.read_csv(path, header=None, names=COLUMNS)
    else:
        df = pd.read_csv(PIMA_URL, header=None, names=COLUMNS)
        DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(path, header=False, index=False)
    return df


def build_pipeline(estimator: Any, scale: bool) -> Pipeline:
    """zero->NaN, median impute (fit only on training folds), optional scaling, model."""
    steps: list[tuple[str, Any]] = [
        ("zeros", ZeroToNaN()),
        ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
    ]
    if scale:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def candidate_models() -> dict[str, Pipeline]:
    """Logistic Regression, Random Forest and (if importable) XGBoost."""
    models = {
        "Logistic Regression": build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED), True),
        "Random Forest": build_pipeline(
            RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=1), False),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost (extra)"] = build_pipeline(
            XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.9,
                          eval_metric="logloss", random_state=SEED, n_jobs=1), False)
    except Exception as exc:  # missing libomp etc.
        warnings.warn(f"XGBoost unavailable, skipped: {exc!s:.120}")
    return models


def cv_scores(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5) -> dict[str, dict[str, float]]:
    """Stratified k-fold CV -> mean/std per metric."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    res = cross_validate(pipe, X, y, cv=skf, scoring=CV_SCORING, n_jobs=1)
    return {m: {"mean": float(res[f"test_{m}"].mean()), "std": float(res[f"test_{m}"].std())} for m in CV_SCORING}


def calibrate(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray) -> CalibratedClassifierCV:
    """Sigmoid-calibrated version of ``pipe`` (internal 5-fold, training data only)."""
    cal = CalibratedClassifierCV(pipe, method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return cal.fit(X, y)


def age_group(age: pd.Series) -> pd.Series:
    """Bucket ages into 21-29 / 30-44 / 45+."""
    return pd.cut(age, bins=[0, 29, 44, 200], labels=["21-29", "30-44", "45+"])


def fairness_table(X: pd.DataFrame, y: np.ndarray, prob: np.ndarray) -> list[dict[str, Any]]:
    """Per-age-group metrics, base rate and positive-prediction rate."""
    groups = age_group(X["Age"])
    rows = []
    for g in groups.cat.categories:
        m = (groups == g).to_numpy()
        if m.sum() == 0:
            continue
        met = binary_metrics(y[m], prob[m])
        rows.append({"group": str(g), "n": int(m.sum()), "base_rate": float(y[m].mean()),
                     "pred_pos_rate": float((prob[m] >= 0.5).mean()), **met})
    return rows
