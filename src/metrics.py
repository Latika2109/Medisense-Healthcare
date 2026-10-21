"""Shared binary-classification metrics and plotting helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC-AUC for probabilities ``y_prob``."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    out["roc_auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {k: float(v) for k, v in out.items()}


def save_confusion_matrix(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str], path: Path, title: str) -> list[list[int]]:
    """Plot and save a confusion matrix; return it as nested lists."""
    cm = confusion_matrix(y_true, (np.asarray(y_prob) >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
           yticklabels=labels, xlabel="Predicted", ylabel="Actual", title=title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cm.tolist()


def write_json(obj: Any, path: Path) -> None:
    """Write ``obj`` as pretty JSON."""
    path.write_text(json.dumps(obj, indent=2))
