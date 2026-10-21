"""MediSense Streamlit dashboard. Loads saved artifacts only (no training at start-up)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ner, prognosis, risk, xai
from src.config import DISCLAIMER, FIG_DIR, MODELS_DIR

st.set_page_config(page_title="MediSense", page_icon="🩺", layout="wide")

st.markdown(
    f"""<div style="position:sticky;top:0;z-index:999;background:#b00020;color:white;padding:8px 14px;
    border-radius:6px;font-size:0.9rem;margin-bottom:8px">
    ⚠️ <b>NOT MEDICAL ADVICE.</b> {DISCLAIMER}</div>""",
    unsafe_allow_html=True,
)
st.title("🩺 MediSense")
st.caption("Unified healthcare AI/ML mini project · academic prototype · models trained offline, loaded from `models/`")


# ---------- artifact helpers ----------
def metrics(name: str) -> dict[str, Any] | None:
    """Read models/<name>_metrics.json if present."""
    p = MODELS_DIR / f"{name}_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def load_joblib(name: str) -> Any:
    """Load a joblib artifact once."""
    return joblib.load(MODELS_DIR / name)


@st.cache_resource
def load_explainer(model_file: str, background_csv: str) -> Any:
    """SHAP explainer for a saved pipeline (built once)."""
    pipe = joblib.load(MODELS_DIR / model_file)
    bg = pd.read_csv(MODELS_DIR / background_csv)
    return pipe, xai.make_explainer(pipe, bg)


@st.cache_resource
def load_nlp() -> Any:
    """Build the rule-based spaCy pipeline."""
    return ner.build_nlp()


@st.cache_resource
def load_cnn() -> Any:
    """Load the CNN weights."""
    import torch
    from src.imaging import SmallCNN

    m = SmallCNN()
    m.load_state_dict(torch.load(MODELS_DIR / "cnn.pt", map_location="cpu"))
    return m.eval()


def need(*files: str) -> bool:
    """Show a hint and return False if any artifact is missing."""
    missing = [f for f in files if not (MODELS_DIR / f).exists()]
    if missing:
        st.warning(f"Missing artifacts {missing}. Run the matching `train_*.py` script first (see README).")
    return not missing


def gauge(p: float, title: str) -> go.Figure:
    """Probability gauge (0-100%)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%"}, title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#333"},
               "steps": [{"range": [0, 33], "color": "#a8d5a2"}, {"range": [33, 66], "color": "#f9d38b"},
                         {"range": [66, 100], "color": "#f4a6a6"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def show_metric_row(m: dict[str, float]) -> None:
    """Render metric values as st.metric columns."""
    cols = st.columns(len(m))
    for c, (k, v) in zip(cols, m.items()):
        c.metric(k.replace("_", " ").upper(), f"{v:.3f}")


def local_explain(pipe: Any, explainer: Any, row: pd.DataFrame, top: int = 5) -> None:
    """Top contributing features table + SHAP waterfall for one row."""
    exp = xai.shap_values(pipe, explainer, row)
    contrib = pd.Series(exp.values[0], index=row.columns)
    top_s = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top)
    st.markdown("**Top contributing features** (SHAP, log-odds; + raises risk, − lowers it)")
    st.dataframe(pd.DataFrame({"feature": top_s.index, "value": row.iloc[0][top_s.index].to_numpy(),
                               "SHAP": top_s.round(3).to_numpy()}), hide_index=True, width="stretch")
    fig = xai.local_shap_figure(exp[0])
    st.pyplot(fig)
    plt.close("all")


def fig_if_exists(name: str, caption: str) -> None:
    """Display a saved figure if it exists."""
    p = FIG_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


tabs = st.tabs(["🩸 Risk", "❤️ Prognosis", "📝 NER", "🫁 Imaging", "💬 Sentiment", "🔍 XAI & Fairness", "⚖️ Ethics & Limits"])

# ---------- Risk ----------
with tabs[0]:
    st.subheader("Diabetes risk (Exp 6) - Pima Indians Diabetes")
    mt = metrics("risk")
    if need("risk_calibrated.joblib", "risk_base.joblib", "risk_background.csv") and mt:
        st.caption(f"Model: **{mt['best_model']}** (chosen by 5-fold CV ROC-AUC), sigmoid-calibrated. "
                   "Enter 0 for an unknown measurement - it is treated as missing and imputed.")
        c1, c2, c3, c4 = st.columns(4)
        vals = {
            "Pregnancies": c1.number_input("Pregnancies", 0, 20, 2),
            "Glucose": c2.number_input("Glucose (mg/dL)", 0, 300, 120),
            "BloodPressure": c3.number_input("Diastolic BP (mmHg)", 0, 200, 70),
            "SkinThickness": c4.number_input("Skin thickness (mm)", 0, 100, 25),
            "Insulin": c1.number_input("Insulin (µU/mL)", 0, 900, 80),
            "BMI": c2.number_input("BMI", 0.0, 70.0, 28.0, step=0.1),
            "DiabetesPedigreeFunction": c3.number_input("Pedigree function", 0.0, 3.0, 0.4, step=0.01),
            "Age": c4.number_input("Age", 18, 100, 35),
        }
        row = pd.DataFrame([vals])[risk.FEATURES].astype(float)
        p = float(load_joblib("risk_calibrated.joblib").predict_proba(row)[0, 1])
        left, right = st.columns([1, 2])
        left.plotly_chart(gauge(p, "Estimated probability"), width="stretch")
        left.caption("Probability from a model trained on a small, historical, single-population dataset.")
        with right:
            base, expl = load_explainer("risk_base.joblib", "risk_background.csv")
            local_explain(base, expl, row)
        st.markdown("**Held-out test metrics (calibrated best model)**")
        show_metric_row(mt["heldout_best_calibrated"])
        comp = pd.DataFrame({n: {k: f"{v['mean']:.3f} ± {v['std']:.3f}" for k, v in s.items()} for n, s in mt["cv_5fold_train"].items()}).T
        st.markdown("**Stratified 5-fold CV (training split)**")
        st.dataframe(comp, width="stretch")
        fig_if_exists("risk_confusion.png", "Held-out confusion matrix")

# ---------- Prognosis ----------
with tabs[1]:
    st.subheader("One-year mortality after MI (Exp 4) - WHAS500")
    mt = metrics("prognosis")
    if need("prognosis.joblib", "prognosis_background.csv") and mt:
        st.caption(f"Label: death within 365 days. Model: **{mt['best_model']}** with class weighting "
                   f"(positive rate {mt['positive_rate']:.1%}). Hospital length of stay is excluded (outcome leakage).")
        med, lo, hi = mt["feature_medians"], mt["feature_min"], mt["feature_max"]
        cols = st.columns(4)
        pv: dict[str, float] = {}
        for i, f in enumerate(sorted(med)):
            label = prognosis.FEATURE_HELP.get(f, f)
            with cols[i % 4]:
                if set([lo[f], hi[f]]) <= {0.0, 1.0}:
                    pv[f] = float(st.selectbox(label, [0, 1], index=int(med[f]), key=f"pg_{f}"))
                else:
                    pv[f] = st.number_input(label, float(lo[f]), float(hi[f]) * 1.2, float(med[f]), key=f"pg_{f}")
        row = pd.DataFrame([pv])[list(med)].astype(float)
        pipe, expl = load_explainer("prognosis.joblib", "prognosis_background.csv")
        p = float(pipe.predict_proba(row[pipe.feature_names_in_])[0, 1])
        left, right = st.columns([1, 2])
        left.plotly_chart(gauge(p, "P(death ≤ 1 year)"), width="stretch")
        with right:
            local_explain(pipe, expl, row[pipe.feature_names_in_])
        st.markdown("**Held-out test metrics**")
        show_metric_row(mt["heldout"])
        cm = np.array(mt["confusion_matrix"])
        st.dataframe(pd.DataFrame(cm, index=["Actual: survived", "Actual: died"], columns=["Pred: survived", "Pred: died"]))
        fig_if_exists("prognosis_confusion.png", "Confusion matrix")

# ---------- NER ----------
with tabs[2]:
    st.subheader("Clinical NER (Exp 5) - spaCy EntityRuler + regex dosage")
    st.caption("Use only synthetic / de-identified text. Never paste real patient data.")
    sample = ("Patient presents with chest pain and shortness of breath. History of hypertension and type 2 diabetes. "
              "Started metformin 500 mg twice daily. Chest x-ray shows opacity in the right lower lobe.")
    text = st.text_area("Clinical note (synthetic)", sample, height=130)
    if text.strip():
        res = ner.extract(load_nlp(), text)
        st.markdown(ner.highlight_html(text, res["entities"]), unsafe_allow_html=True)
        st.markdown("**Legend:** " + " ".join(f'<span style="background:{c};padding:2px 6px;border-radius:4px;color:#111">{l}</span>'
                                              for l, c in ner.COLORS.items()), unsafe_allow_html=True)
        for i, s in enumerate(res["sentences"], 1):
            st.markdown(f"**Sentence {i}:** {s['text']}")
            if s["entities"]:
                st.dataframe(pd.DataFrame(s["entities"])[["text", "label"]], hide_index=True)
    mt = metrics("ner")
    if mt:
        o = mt["overall"]
        st.markdown(f"**Hand-labelled test set ({mt['n_notes']} notes, exact match):** "
                    f"P={o['precision']:.3f} · R={o['recall']:.3f} · F1={o['f1']:.3f}")
        st.dataframe(pd.DataFrame(mt["per_label"]).T.round(3))
        st.caption("Tiny, self-written test set and lexicon-based system: scores are optimistic and not a clinical benchmark. "
                   "Extend `data/ner_test.jsonl` and re-run `python train_ner.py`.")

# ---------- Imaging ----------
with tabs[3]:
    st.subheader("Chest X-ray pneumonia (Exp 3) - PneumoniaMNIST 28×28 CNN + Grad-CAM")
    mt = metrics("cnn")
    if need("cnn.pt", "cnn_samples.npz") and mt:
        from src import imaging

        data = np.load(MODELS_DIR / "cnn_samples.npz")
        imgs, labels = data["images"], data["labels"]
        src_choice = st.radio("Image source", ["Demo test sample", "Upload image (PNG/JPG, resized to 28×28 grayscale)"], horizontal=True)
        img, truth = None, None
        if src_choice.startswith("Demo"):
            i = st.slider("Sample", 0, len(imgs) - 1, 0)
            img, truth = imgs[i], int(labels[i])
        else:
            up = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
            if up:
                from PIL import Image
                img = np.asarray(Image.open(up).convert("L").resize((28, 28)), dtype=np.uint8)
        if img is not None:
            model = load_cnn()
            p = float(imaging.predict_proba(model, img[None])[0])
            cam = imaging.grad_cam(model, img)
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.image(np.kron(img, np.ones((8, 8), dtype=np.uint8)), caption="Input" + (f" (true: {'pneumonia' if truth else 'normal'})" if truth is not None else ""))
            fig, ax = plt.subplots(figsize=(3, 3)); ax.imshow(img, cmap="gray"); ax.imshow(cam, cmap="jet", alpha=0.45); ax.axis("off")
            c2.pyplot(fig); plt.close(fig)
            c3.plotly_chart(gauge(p, "P(pneumonia)"), width="stretch")
            st.caption("Grad-CAM highlights regions that increased the pneumonia score; it is a rough explanation, not proof of what the model 'sees'.")
        st.markdown("**Official test-split metrics**")
        show_metric_row(mt["test"])
        fig_if_exists("cnn_confusion.png", "Confusion matrix (test)")
        fig_if_exists("cnn_gradcam.png", "Grad-CAM gallery")

# ---------- Sentiment ----------
with tabs[4]:
    st.subheader("Drug review sentiment (Exp 7, optional) - TF-IDF + LR / Multinomial NB")
    mt = metrics("sentiment")
    if need("sentiment_lr.joblib", "sentiment_nb.joblib") and mt:
        txt = st.text_area("Review (synthetic)", "This medication worked well for my headaches but made me very drowsy.")
        if txt.strip():
            c = st.columns(2)
            for col, (nm, f) in zip(c, [("Logistic Regression", "sentiment_lr.joblib"), ("Multinomial NB", "sentiment_nb.joblib")]):
                pr = load_joblib(f).predict_proba([txt])[0]
                col.markdown(f"**{nm}** → `{mt['labels'][int(pr.argmax())]}`")
                col.bar_chart(pd.Series(pr, index=mt["labels"]))
        rows = {n: {**{k: f"{v['mean']:.3f}" for k, v in s.items()}} for n, s in mt["cv"].items()}
        st.markdown("**5-fold CV (macro-averaged)**")
        st.dataframe(pd.DataFrame(rows).T, width="stretch")
        st.markdown("**Held-out test**")
        st.dataframe(pd.DataFrame({n: {k: round(v, 3) for k, v in d.items() if k != "confusion_matrix"} for n, d in mt["heldout"].items()}).T)
        st.caption("Labels from star ratings (1-4 negative, 5-6 neutral, 7-10 positive); neutral is rare, so macro scores are low.")

# ---------- XAI & Fairness ----------
with tabs[5]:
    st.subheader("Explainability (Exp 8) and fairness")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Diabetes model")
        fig_if_exists("risk_native_importance.png", "Native feature importance")
        fig_if_exists("risk_permutation_importance.png", "Permutation importance (held-out)")
        fig_if_exists("risk_shap_summary.png", "SHAP global summary")
    with c2:
        st.markdown("##### Prognosis model")
        fig_if_exists("prognosis_permutation_importance.png", "Permutation importance (held-out)")
        fig_if_exists("prognosis_shap_summary.png", "SHAP global summary")
    st.markdown("Local SHAP (per patient) appears in the Risk/Prognosis tabs, Grad-CAM in Imaging, highlighted entities in NER.")
    mt = metrics("risk")
    if mt:
        st.markdown("##### Fairness: diabetes model by age group")
        for key, title in [("out_of_fold_all_data", "Out-of-fold predictions on all 768 patients (larger n per group)"),
                           ("test_set", "Held-out test set only (small n - noisy)")]:
            st.markdown(f"*{title}*")
            st.dataframe(pd.DataFrame(mt["fairness"][key]).set_index("group").round(3), width="stretch")
        st.info("Groups differ in base rate (diabetes prevalence) and in recall, so a single threshold does not give equal "
                "error rates. Dataset: Pima women aged 21+, so results do not describe men, other ethnicities or children.")

# ---------- Ethics ----------
with tabs[6]:
    st.subheader("Limitations, bias and ethics")
    st.markdown("""
- **Not a medical device.** Outputs are research demonstrations; never use them for diagnosis or treatment.
- **Small, old, narrow data.** Pima (768 women, one population), WHAS500 (500 MI patients from Worcester, MA, 1990s-2000s),
  PneumoniaMNIST (28×28 down-sampled paediatric chest X-rays), a lexicon built by hand. None generalise to other settings.
- **Missing values.** Zeros in Pima are treated as missing and median-imputed inside the CV pipeline; imputation can hide bias.
- **Bias & fairness.** Metrics differ across age groups; no race/ethnicity/sex-balanced evaluation is possible with these datasets.
- **Calibration ≠ validity.** Calibrated probabilities are only as trustworthy as the training population.
- **Explanations are approximations.** SHAP and Grad-CAM show model behaviour, not causal medical reasoning.
- **Privacy.** Use only synthetic or de-identified text. The app stores nothing and calls no external service.
- **Imaging shift.** PneumoniaMNIST validation scores are noticeably higher than test scores (known distribution shift).
- **Human oversight.** Any real deployment would need clinical validation, regulatory approval and clinician-in-the-loop use.
""")
