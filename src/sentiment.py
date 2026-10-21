"""Module 5 (optional) - drug review sentiment with TF-IDF + LR / Multinomial NB (Exp 7)."""
from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import DATA_DIR, DRUG_URL, SEED

LABELS = ["negative", "neutral", "positive"]


def rating_to_label(r: float) -> int:
    """1-4 -> 0 (negative), 5-6 -> 1 (neutral), 7-10 -> 2 (positive)."""
    return 0 if r <= 4 else (1 if r <= 6 else 2)


def load_reviews(n: int = 20000) -> pd.DataFrame:
    """UCI Drug Review (Drugs.com) train split, cached in data/; stratified-ish random sample of ``n``."""
    cache = DATA_DIR / "drugsComTrain_raw.tsv"
    if not cache.exists():
        raw = urllib.request.urlopen(DRUG_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            cache.write_bytes(z.read("drugsComTrain_raw.tsv"))
    df = pd.read_csv(cache, sep="\t", usecols=["review", "rating"]).dropna()
    df["review"] = df["review"].str.replace("&#039;", "'", regex=False).str.replace(r"\s+", " ", regex=True)
    df["label"] = df["rating"].map(rating_to_label)
    return df.sample(min(n, len(df)), random_state=SEED).reset_index(drop=True)


def candidate_models() -> dict[str, Pipeline]:
    """TF-IDF + Logistic Regression and TF-IDF + Multinomial NB."""
    tfidf = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True, stop_words="english")
    return {
        "TF-IDF + Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=SEED))]),
        "TF-IDF + Multinomial NB": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.3))]),
    }
