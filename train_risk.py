"""Train the diabetes risk module: CV comparison, held-out test, calibration, XAI, fairness."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src import risk, xai
from src.config import FIG_DIR, MODELS_DIR, SEED, ensure_dirs
from src.metrics import binary_metrics, save_confusion_matrix, write_json


def main() -> None:
    """Run the full Exp 6 pipeline and save artifacts."""
    ensure_dirs()
    np.random.seed(SEED)
    df = risk.load_pima()
    X, y = df[risk.FEATURES], df["Outcome"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = risk.candidate_models()
    cv = {name: risk.cv_scores(p, X_tr, y_tr) for name, p in models.items()}
    for name, s in cv.items():
        print(f"CV {name:22s} " + " ".join(f"{m}={v['mean']:.3f}±{v['std']:.3f}" for m, v in s.items()))
    best = max(cv, key=lambda n: cv[n]["roc_auc"]["mean"])  # selection on CV only, never on test
    print("Best by CV ROC-AUC:", best)

    test = {}
    for name, p in models.items():
        fitted = clone(p).fit(X_tr, y_tr)
        test[name] = binary_metrics(y_te, fitted.predict_proba(X_te)[:, 1])
    base = clone(models[best]).fit(X_tr, y_tr)
    cal = risk.calibrate(models[best], X_tr, y_tr)
    p_raw, p_cal = base.predict_proba(X_te)[:, 1], cal.predict_proba(X_te)[:, 1]
    final = binary_metrics(y_te, p_cal)
    brier_raw = float(np.mean((p_raw - y_te) ** 2))
    brier_cal = float(np.mean((p_cal - y_te) ** 2))
    print("Held-out (uncalibrated, all models):", {k: {m: round(v, 3) for m, v in d.items()} for k, d in test.items()})
    print("Held-out calibrated best:", {k: round(v, 3) for k, v in final.items()}, f"brier {brier_raw:.4f}->{brier_cal:.4f}")
    cm = save_confusion_matrix(y_te, p_cal, ["No diabetes", "Diabetes"], FIG_DIR / "risk_confusion.png", f"Risk: {best}")

    # XAI on the uncalibrated base pipeline (calibration is monotone, so ranking is unchanged)
    names = risk.FEATURES
    nat = xai.native_importance(base, names)
    perm = xai.perm_importance(base, X_te, y_te)
    xai.save_bar(nat, FIG_DIR / "risk_native_importance.png", "Native importance", "importance")
    xai.save_bar(perm["importance"], FIG_DIR / "risk_permutation_importance.png", "Permutation importance (ROC-AUC drop)", "drop")
    expl = xai.make_explainer(base, X_tr)
    exp = xai.shap_values(base, expl, X_te)
    xai.save_shap_global(exp, FIG_DIR / "risk_shap_summary.png", f"SHAP summary - {best}")

    # Fairness: held-out test AND out-of-fold predictions on all data (larger n per group)
    oof = cross_val_predict(models[best], X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    fair = {"test_set": risk.fairness_table(X_te, y_te, p_cal), "out_of_fold_all_data": risk.fairness_table(X, y, oof)}
    for r in fair["out_of_fold_all_data"]:
        print("Fairness OOF", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

    joblib.dump(cal, MODELS_DIR / "risk_calibrated.joblib")
    joblib.dump(base, MODELS_DIR / "risk_base.joblib")
    X_tr.sample(200, random_state=SEED).to_csv(MODELS_DIR / "risk_background.csv", index=False)
    write_json({
        "best_model": best, "cv_5fold_train": cv, "heldout_uncalibrated": test,
        "heldout_best_calibrated": final, "confusion_matrix": cm,
        "brier_uncalibrated": brier_raw, "brier_calibrated": brier_cal,
        "native_importance": nat.to_dict(), "permutation_importance": perm["importance"].to_dict(),
        "fairness": fair, "n_train": len(X_tr), "n_test": len(X_te), "seed": SEED,
        "feature_medians": X_tr.replace(0, np.nan).median().to_dict(),
    }, MODELS_DIR / "risk_metrics.json")


if __name__ == "__main__":
    main()
