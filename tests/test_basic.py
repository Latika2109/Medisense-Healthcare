"""Basic smoke tests (fast, offline, no downloads)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import imaging, ner, risk


def _toy() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(100, 20, (200, 8)), columns=risk.FEATURES).abs()
    X.loc[:20, "Glucose"] = 0
    y = (X["Glucose"] + rng.normal(0, 5, 200) > 100).astype(int).to_numpy()
    return X, y


def test_zero_to_nan_only_selected_columns() -> None:
    df = pd.DataFrame({c: [0.0, 1.0] for c in risk.FEATURES})
    out = risk.ZeroToNaN().fit_transform(df)
    assert out["Glucose"].isna().iloc[0] and not out["Pregnancies"].isna().any() and not out["Age"].isna().any()


def test_risk_pipeline_fits_with_missing_values() -> None:
    X, y = _toy()
    pipe = risk.candidate_models()["Logistic Regression"].fit(X, y)
    p = pipe.predict_proba(X)[:, 1]
    assert p.shape == (200,) and ((p >= 0) & (p <= 1)).all()


def test_ner_dosage_and_entities() -> None:
    res = ner.extract(ner.build_nlp(), "Started metformin 500 mg twice daily for type 2 diabetes.")
    got = {(e["text"], e["label"]) for e in res["entities"]}
    assert ("500 mg twice daily", "DOSAGE") in got and ("metformin", "MEDICATION") in got
    assert ("type 2 diabetes", "DISEASE") in got


def test_ner_testset_loads_and_scores() -> None:
    r = ner.evaluate(ner.build_nlp(), ner.load_test_set())
    assert r["overall"]["f1"] > 0.5


def test_cnn_shapes_and_gradcam() -> None:
    img = (np.random.default_rng(0).random((28, 28)) * 255).astype(np.uint8)
    m = imaging.SmallCNN()
    assert imaging.predict_proba(m, img[None]).shape == (1,)
    cam = imaging.grad_cam(m, img)
    assert cam.shape == (28, 28) and 0 <= cam.min() and cam.max() <= 1
