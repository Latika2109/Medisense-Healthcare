"""Train/evaluate drug-review sentiment models (3-class) with CV and a held-out test split."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src import sentiment
from src.config import MODELS_DIR, SEED, ensure_dirs
from src.metrics import write_json

SCORING = {"accuracy": "accuracy", "precision_macro": "precision_macro", "recall_macro": "recall_macro", "f1_macro": "f1_macro"}


def main() -> None:
    """Run Exp 7."""
    ensure_dirs()
    np.random.seed(SEED)
    df = sentiment.load_reviews()
    print("class balance:", df["label"].value_counts(normalize=True).sort_index().round(3).to_dict())
    X_tr, X_te, y_tr, y_te = train_test_split(df["review"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out: dict = {"cv": {}, "heldout": {}, "labels": sentiment.LABELS}
    for name, p in sentiment.candidate_models().items():
        r = cross_validate(p, X_tr, y_tr, cv=skf, scoring=SCORING)
        out["cv"][name] = {m: {"mean": float(r[f"test_{m}"].mean()), "std": float(r[f"test_{m}"].std())} for m in SCORING}
        fitted = clone(p).fit(X_tr, y_tr)
        pred = fitted.predict(X_te)
        out["heldout"][name] = {"accuracy": accuracy_score(y_te, pred),
                                "precision_macro": precision_score(y_te, pred, average="macro"),
                                "recall_macro": recall_score(y_te, pred, average="macro"),
                                "f1_macro": f1_score(y_te, pred, average="macro"),
                                "confusion_matrix": confusion_matrix(y_te, pred).tolist()}
        print(name, "CV", {m: round(v["mean"], 3) for m, v in out["cv"][name].items()},
              "| test", {k: round(v, 3) for k, v in out["heldout"][name].items() if k != "confusion_matrix"})
        joblib.dump(fitted, MODELS_DIR / ("sentiment_lr.joblib" if "Logistic" in name else "sentiment_nb.joblib"))
    out.update(n_train=len(X_tr), n_test=len(X_te), seed=SEED)
    write_json(out, MODELS_DIR / "sentiment_metrics.json")


if __name__ == "__main__":
    main()
