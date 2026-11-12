"""Category and severity classifiers.

Task type: supervised multi-class text classification (NOT regression/forecasting).
 - category : safety | environment | esg_compliance      (nominal classes)
 - severity : low | medium | high                        (ordinal; final label comes from the rule engine,
                                                          ML label is a second opinion)

Chosen model : TF-IDF (word 1-2 grams) + Logistic Regression (L2).
Baselines    : majority class, and a keyword-rule engine (see nlp.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import nlp

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
SEVERITY_ORDER = ["low", "medium", "high"]
CATEGORIES = ["safety", "environment", "esg_compliance"]

# Key hyper-parameters (explained in docs/PROJECT_DOCUMENT.md):
#  C             - inverse regularisation strength: lower = simpler model, less over-fitting on tiny data
#  ngram_range   - (1,2) lets "near miss" / "spill kit" act as single features
#  class_weight  - 'balanced' up-weights rare classes (e.g. high severity)
LOW_CONF = 0.55  # category probability below this => flag for human review
DEFAULT_PARAMS = {"C": 10.0, "ngram_range": (1, 2), "class_weight": "balanced"}


def make_pipeline(params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=nlp.normalize, ngram_range=p["ngram_range"],
                                  sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(C=p["C"], class_weight=p["class_weight"], max_iter=2000)),
    ])


def load_training_frame():
    df = pd.read_csv(DATA / "extended_incidents.csv")
    return df


class IncidentModel:
    """Holds the two fitted pipelines and exposes predict()."""

    def __init__(self, params=None):
        self.params = params
        self.cat_pipe = make_pipeline(params)
        self.fitted = False

    def fit(self, texts, categories):
        self.cat_pipe.fit(texts, categories)
        self.fitted = True
        return self

    # DESIGN DECISION (from evaluation, see docs): severity comes from the rule engine in nlp.py.
    # A TF-IDF severity classifier scored ~34% accuracy on unseen templates (36 templates is too little
    # data to learn severity semantics), and an ML+rule hybrid scored worse than rules alone, so the
    # ML severity model is NOT used here. It is kept only as a documented failed experiment in
    # scripts/evaluate.py.

    def predict_one(self, text: str):
        cat_probs = dict(zip(self.cat_pipe.classes_, self.cat_pipe.predict_proba([text])[0]))
        cat = max(cat_probs, key=cat_probs.get)
        confidence = float(cat_probs[cat])
        sev, evidence = nlp.rule_severity(text)
        no_cue = evidence and evidence[0].startswith("no high or low cue")
        reasons = []
        if confidence < LOW_CONF:
            reasons.append(f"category confidence {confidence:.2f} < {LOW_CONF}")
        if no_cue:
            reasons.append("no severity cue matched (defaulted to medium)")
        return {
            "category": cat,
            "category_probabilities": {k: round(float(v), 3) for k, v in cat_probs.items()},
            "category_confidence": round(confidence, 3),
            "severity": sev,
            "severity_method": "rule_engine",
            "severity_evidence": evidence,
            "needs_human_review": bool(reasons),
            "review_reasons": reasons,
        }


_MODEL = None


def get_model() -> IncidentModel:
    """Train once per process (<1 s on this data). Avoids pickle/sklearn-version mismatches on deploy."""
    global _MODEL
    if _MODEL is None:
        df = load_training_frame()
        _MODEL = IncidentModel().fit(df["narrative"], df["target_category"])
    return _MODEL
