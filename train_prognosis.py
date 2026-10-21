"""Train WHAS500 one-year mortality classifier; save model, metrics and figures."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from src import prognosis, xai
from src.config import FIG_DIR, MODELS_DIR, SEED, ensure_dirs
from src.metrics import binary_metrics, save_confusion_matrix, write_json
from src.risk import CV_SCORING


def main() -> None:
    """Run Exp 4 pipeline."""
    ensure_dirs()
    np.random.seed(SEED)
    X, y = prognosis.load_data()
    print(f"n={len(y)} positives={y.sum()} ({y.mean():.1%}) features={list(X.columns)}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    models = prognosis.candidate_models()
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    cv = {}
    for name, p in models.items():
        r = cross_validate(p, X_tr, y_tr, cv=skf, scoring=CV_SCORING)
        cv[name] = {m: {"mean": float(r[f"test_{m}"].mean()), "std": float(r[f"test_{m}"].std())} for m in CV_SCORING}
        print(f"CV {name:20s} " + " ".join(f"{m}={v['mean']:.3f}±{v['std']:.3f}" for m, v in cv[name].items()))
    best = max(cv, key=lambda n: cv[n]["roc_auc"]["mean"])
    pipe = clone(models[best]).fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_te)[:, 1]
    test = binary_metrics(y_te, prob)
    print("Best:", best, "| held-out:", {k: round(v, 3) for k, v in test.items()})
    cm = save_confusion_matrix(y_te, prob, ["Survived 1y", "Died <=1y"], FIG_DIR / "prognosis_confusion.png", f"Prognosis: {best}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:", cm)

    nat = xai.native_importance(pipe, list(X.columns))
    perm = xai.perm_importance(pipe, X_te, y_te)
    xai.save_bar(perm["importance"], FIG_DIR / "prognosis_permutation_importance.png", "Permutation importance (ROC-AUC drop)", "drop")
    expl = xai.make_explainer(pipe, X_tr)
    xai.save_shap_global(xai.shap_values(pipe, expl, X_te), FIG_DIR / "prognosis_shap_summary.png", f"SHAP summary - {best}")

    joblib.dump(pipe, MODELS_DIR / "prognosis.joblib")
    X_tr.sample(100, random_state=SEED).to_csv(MODELS_DIR / "prognosis_background.csv", index=False)
    write_json({"best_model": best, "cv_5fold_train": cv, "heldout": test, "confusion_matrix": cm,
                "positive_rate": float(y.mean()), "n_train": len(X_tr), "n_test": len(X_te),
                "native_importance": nat.to_dict(), "permutation_importance": perm["importance"].to_dict(),
                "feature_medians": X_tr.median().to_dict(), "feature_min": X.min().to_dict(), "feature_max": X.max().to_dict(),
                "seed": SEED}, MODELS_DIR / "prognosis_metrics.json")


if __name__ == "__main__":
    main()
