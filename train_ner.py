"""Evaluate the rule-based NER pipeline on data/ner_test.jsonl and save metrics."""
from __future__ import annotations

from src import ner
from src.config import MODELS_DIR, ensure_dirs
from src.metrics import write_json


def main() -> None:
    """Evaluate and write models/ner_metrics.json."""
    ensure_dirs()
    res = ner.evaluate(ner.build_nlp(), ner.load_test_set())
    o = res["overall"]
    print(f"NER overall (exact match) P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f} on {res['n_notes']} notes")
    for lab, m in res["per_label"].items():
        print(f"  {lab:14s} P={m['precision']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f} n={m['support']}")
    for e in res["errors"]:
        print("  error:", e)
    write_json(res, MODELS_DIR / "ner_metrics.json")


if __name__ == "__main__":
    main()
