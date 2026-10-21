"""Explainability helpers shared by tabular modules (Exp 8)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from .config import SEED


def native_importance(pipe: Pipeline, feature_names: list[str]) -> pd.Series:
    """Model-native importance (|coef| for linear, impurity for trees)."""
    clf = pipe[-1]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).ravel()
    else:
        raise ValueError("Model has no native importance")
    return pd.Series(vals, index=feature_names).sort_values(ascending=False)


def perm_importance(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 20) -> pd.DataFrame:
    """Permutation importance (ROC-AUC drop) on held-out data."""
    r = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=SEED)
    return (pd.DataFrame({"importance": r.importances_mean, "std": r.importances_std}, index=X.columns)
            .sort_values("importance", ascending=False))


def _transform(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every pipeline step except the final estimator."""
    Xt = pipe[:-1].transform(X)
    return pd.DataFrame(np.asarray(Xt), columns=X.columns, index=X.index)


def make_explainer(pipe: Pipeline, X_background: pd.DataFrame) -> shap.Explainer:
    """SHAP explainer for the pipeline's final estimator on transformed features."""
    bg = _transform(pipe, X_background)
    clf = pipe[-1]
    if hasattr(clf, "feature_importances_"):
        return shap.TreeExplainer(clf, bg)
    return shap.LinearExplainer(clf, bg)


def shap_values(pipe: Pipeline, explainer: shap.Explainer, X: pd.DataFrame) -> shap.Explanation:
    """SHAP Explanation for the positive class (log-odds / margin scale)."""
    Xt = _transform(pipe, X)
    exp = explainer(Xt)
    if exp.values.ndim == 3:  # (n, features, classes) -> positive class
        exp = exp[..., 1]
    return exp


def save_shap_global(exp: shap.Explanation, path: Path, title: str = "SHAP summary") -> None:
    """Save beeswarm summary plot."""
    plt.figure()
    shap.plots.beeswarm(exp, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")


def local_shap_figure(exp_row: shap.Explanation) -> Any:
    """Waterfall figure for a single-row Explanation (caller closes figure)."""
    plt.figure()
    shap.plots.waterfall(exp_row, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def save_bar(series: pd.Series, path: Path, title: str, xlabel: str) -> None:
    """Horizontal bar chart of a Series."""
    s = series.sort_values()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(s.index, s.values, color="#3b6ea5")
    ax.set(title=title, xlabel=xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
