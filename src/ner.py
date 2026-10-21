"""Module 3 - clinical NER with spaCy EntityRuler + regex dosage (Exp 5)."""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

from .config import DATA_DIR
from .lexicon import LEXICON

LABELS = ["DISEASE", "SYMPTOM", "MEDICATION", "INVESTIGATION", "DOSAGE", "ANATOMY"]
COLORS = {"DISEASE": "#f4a6a6", "SYMPTOM": "#f9d38b", "MEDICATION": "#a8d5a2",
          "INVESTIGATION": "#a6c8f4", "DOSAGE": "#d3b5ec", "ANATOMY": "#b8e0e0"}

_UNIT = r"(?:mg|mcg|µg|g|ml|units?|iu|puffs?|tablets?|tabs?|capsules?)"
_FREQ = (r"(?:once|twice|three times|four times|[1-4]\s*times)\s+(?:daily|a day|per day|weekly|a week)"
         r"|every\s+\d+\s*(?:hours?|hrs?|h)\b|(?:od|bd|bid|tid|qid|qd|prn|qhs)\b|at (?:night|bedtime)")
# dose with optional frequency ("500 mg twice daily"), or a frequency phrase on its own.
DOSAGE_RE = re.compile(rf"\b\d+(?:\.\d+)?\s?{_UNIT}\b(?:\s+(?:{_FREQ}|daily))?|\b(?:{_FREQ})", re.IGNORECASE)


@Language.component("dosage_regex")
def dosage_regex(doc: Doc) -> Doc:
    """Add DOSAGE entities from a regex; dosage wins over overlapping lexicon matches."""
    spans = []
    for m in DOSAGE_RE.finditer(doc.text):
        s = doc.char_span(m.start(), m.end(), label="DOSAGE", alignment_mode="expand")
        if s is not None:
            spans.append(s)
    doc.ents = _merge(doc, spans)
    return doc


def _merge(doc: Doc, dosage: list[Span]) -> tuple[Span, ...]:
    """Keep dosage spans; drop other entities that overlap them."""
    keep = [e for e in doc.ents if not any(e.start < d.end and d.start < e.end for d in dosage)]
    return tuple(filter_spans(keep + dosage))


def build_nlp() -> Language:
    """Blank English pipeline: sentencizer -> EntityRuler(lexicon) -> dosage regex.

    scispaCy is not wired in (needs a separate model download); see README.
    """
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True})
    ruler.add_patterns([{"label": lab, "pattern": term} for lab, terms in LEXICON.items() for term in terms])
    nlp.add_pipe("dosage_regex")
    return nlp


def extract(nlp: Language, text: str) -> dict[str, Any]:
    """Sentence-chunked entity extraction."""
    doc = nlp(text)
    sents = [{"text": s.text, "entities": [{"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                                           for e in s.ents]} for s in doc.sents]
    return {"sentences": sents, "entities": [e for s in sents for e in s["entities"]]}


def highlight_html(text: str, entities: list[dict[str, Any]]) -> str:
    """HTML with coloured entity highlights (XAI for NER: which spans drove the labels)."""
    out, pos = [], 0
    for e in sorted(entities, key=lambda e: e["start"]):
        out.append(html.escape(text[pos:e["start"]]))
        out.append(f'<mark style="background:{COLORS.get(e["label"], "#ddd")};padding:2px 4px;border-radius:4px;color:#111">'
                   f'{html.escape(text[e["start"]:e["end"]])} <sub><b>{e["label"]}</b></sub></mark>')
        pos = e["end"]
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def load_test_set(path: Path = DATA_DIR / "ner_test.jsonl") -> list[dict[str, Any]]:
    """Read hand-labelled notes: {"text": ..., "entities": [[surface, LABEL], ...]} in text order."""
    items = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pos, gold = 0, set()
        for surface, label in rec["entities"]:
            i = rec["text"].find(surface, pos)
            if i < 0:
                raise ValueError(f"Entity {surface!r} not found (in order) in: {rec['text'][:60]}")
            gold.add((i, i + len(surface), label))
            pos = i + len(surface)
        items.append({"text": rec["text"], "gold": gold})
    return items


def evaluate(nlp: Language, items: list[dict[str, Any]]) -> dict[str, Any]:
    """Exact-match entity-level precision/recall/F1 overall and per label."""
    tp: dict[str, int] = {}; fp: dict[str, int] = {}; fn: dict[str, int] = {}
    errors = []
    for it in items:
        pred = {(e.start_char, e.end_char, e.label_) for e in nlp(it["text"]).ents}
        for s, name in ((pred & it["gold"], tp), (pred - it["gold"], fp), (it["gold"] - pred, fn)):
            for _, _, lab in s:
                name[lab] = name.get(lab, 0) + 1
        if pred != it["gold"]:
            t = it["text"]
            errors.append({"missed": [t[a:b] + f" [{l}]" for a, b, l in sorted(it["gold"] - pred)],
                           "spurious": [t[a:b] + f" [{l}]" for a, b, l in sorted(pred - it["gold"])]})

    def prf(t: int, p: int, n: int) -> dict[str, float]:
        pr = t / (t + p) if t + p else 0.0
        rc = t / (t + n) if t + n else 0.0
        return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "support": t + n}

    per = {lab: prf(tp.get(lab, 0), fp.get(lab, 0), fn.get(lab, 0)) for lab in LABELS}
    overall = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return {"overall": overall, "per_label": per, "n_notes": len(items), "errors": errors}
