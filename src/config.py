"""Project-wide paths and constants."""
from __future__ import annotations

from pathlib import Path

SEED: int = 42
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT / "data"
MODELS_DIR: Path = ROOT / "models"
FIG_DIR: Path = ROOT / "reports" / "figures"

PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
DRUG_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
)

DISCLAIMER = (
    "MediSense is an academic prototype for a college mini project. It is NOT a "
    "medical device and gives NO medical advice, diagnosis or treatment. "
    "Consult a qualified clinician for any health decision."
)


def ensure_dirs() -> None:
    """Create output directories if missing."""
    for d in (DATA_DIR, MODELS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
