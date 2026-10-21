# 🩺 MediSense

Unified healthcare AI/ML mini project (Python, scikit-learn, PyTorch, spaCy, SHAP, Streamlit).

> ⚠️ **Academic prototype - NOT a medical device and NOT medical advice.** Do not use for diagnosis or treatment.
> Use only synthetic or de-identified text.

## Experiment → module map

| Exp | Module | Code | Train / eval script | Data |
|-----|--------|------|---------------------|------|
| 6 | Diabetes risk | `src/risk.py` | `train_risk.py` | Pima Indians Diabetes (downloaded from jbrownlee/Datasets) |
| 4 | Prognosis (1-yr mortality) | `src/prognosis.py` | `train_prognosis.py` | WHAS500 via `sksurv.datasets.load_whas500` |
| 5 | Clinical NER | `src/ner.py`, `src/lexicon.py` | `train_ner.py` | hand-labelled `data/ner_test.jsonl` (synthetic) |
| 3 | Imaging CNN + Grad-CAM | `src/imaging.py` | `train_cnn.py` | PneumoniaMNIST via `medmnist` |
| 7 | Sentiment (optional) | `src/sentiment.py` | `train_sentiment.py` | UCI Drug Review (Drugs.com) |
| 8 | XAI | `src/xai.py` + Grad-CAM + NER highlights | called from the scripts/app | - |
| - | Fairness (by age group) | `src/risk.py` | `train_risk.py` | Pima |

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # tested on Python 3.12, macOS arm64
pip install -r requirements.txt
# macOS only, to enable the optional XGBoost model:  brew install libomp
```

## Run

```bash
python train_risk.py        # models/risk_*.{joblib,json,csv}, reports/figures/risk_*.png
python train_prognosis.py
python train_ner.py
python train_cnn.py --epochs 8   # ~10 min on CPU; downloads PneumoniaMNIST (~4 MB)
python train_sentiment.py        # downloads ~43 MB zip once
streamlit run app.py             # loads saved artifacts only
python -m pytest -q
```

Every script uses seed 42. Raw data lands in `data/` (large files are git-ignored).

## Results (real runs, seed 42)

Held-out test split (20% stratified; model selected on 5-fold CV of the training split only).

| Module | Model | Acc | Prec | Recall | F1 | ROC-AUC |
|--------|-------|-----|------|--------|----|---------|
| Risk (n_test=154) | Logistic Regression, calibrated | 0.701 | 0.587 | 0.500 | 0.540 | 0.812 |
| Prognosis (n_test=100) | Random Forest, class-weighted | 0.780 | 0.625 | 0.536 | 0.577 | 0.810 |
| Imaging (official test, n=624) | Small CNN | 0.894 | 0.943 | 0.885 | 0.913 | 0.964 |

* Risk 5-fold CV ROC-AUC: LR 0.843±0.019, RF 0.827±0.026. (XGBoost is skipped if `libomp` is missing; it was skipped in the reported run.)
* Calibration (sigmoid): Brier 0.1754 → 0.1735 on the test split - a small gain.
* Prognosis confusion matrix `[[TN, FP],[FN, TP]]` = `[[63, 9], [13, 15]]`. Imaging: `[[213, 21], [45, 345]]`.
* NER (12 synthetic notes, exact-match): P 1.00 / R 0.95 / F1 0.97 - optimistic; the lexicon and test set were written by the same person, and the test set is tiny.
* Sentiment (3-class, 20k reviews, macro): LR F1 0.52, Multinomial NB F1 0.44 (NB never predicts the rare "neutral" class).
* Metrics are noisy on test sets of 100-150 rows; compare with the CV numbers in `models/*_metrics.json`.
* PneumoniaMNIST validation AUC (~0.99) is much higher than test AUC (0.964): a known distribution shift.

## Design notes

* **No leakage:** Pima zeros → NaN → median imputation (and scaling) live *inside* the sklearn `Pipeline`, so they are fit per CV fold.
* **WHAS500:** label = `fstat==1 and lenfol<=365`. `los` (hospital stay) is excluded since it leaks the outcome. Imbalance (27.6% positive) handled with class weights.
* **NER:** blank spaCy English + sentencizer + `EntityRuler` (lexicon) + regex component for dosage (`500 mg twice daily`). scispaCy is *not* included; to try it, install `scispacy` + `en_core_sci_sm` and add its entities as an extra component.
* **XAI:** native + permutation importance and SHAP (global figure, local waterfall in app) for tabular models; Grad-CAM for the CNN; highlighted spans for NER.
* **Fairness:** metrics by age group (21-29, 30-44, 45+) on the test set and on out-of-fold predictions for all 768 rows (larger groups). Recall rises with age (0.45 → 0.73) while ROC-AUC falls (0.85 → 0.74).
* **Bigger imaging dataset:** `medmnist` supports `size=64/128/224` (e.g. `PneumoniaMNIST(size=224)`), or use the Kaggle *Chest X-Ray Images (Pneumonia)* dataset (Kermany et al.) with a pretrained ResNet - a GPU is advisable.
* **Sentiment data:** `load_reviews(n=...)` can use the full 161k-review train file.

## Layout

```
app.py  train_*.py  src/  models/  data/  reports/figures/  tests/  requirements.txt
```

## Limitations & ethics

Small, historical, single-population datasets; no external validation; explanations (SHAP/Grad-CAM) approximate model behaviour only.
See the *Ethics & Limits* tab in the app.
