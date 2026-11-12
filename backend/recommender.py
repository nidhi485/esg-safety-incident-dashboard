"""Retrieval-based preventive-action recommender.

This is plain retrieval over a small curated knowledge base (data/preventive_actions_kb.csv),
NOT generative RAG: nothing is generated, the returned text is the stored action verbatim.

score = 0.6 * cosine_tfidf(narrative, kb.applies_when) + 0.4 * tag_match
        tag_match = 1 if kb.hazard_tag is among the narrative's detected hazard tags, else 0
Candidates are also soft-filtered to the predicted category (boost, not hard filter).
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import nlp

KB_PATH = Path(__file__).resolve().parents[1] / "data" / "preventive_actions_kb.csv"
_KB = None


def _load():
    global _KB
    if _KB is None:
        kb = pd.read_csv(KB_PATH)
        vec = TfidfVectorizer(preprocessor=nlp.normalize, ngram_range=(1, 2), sublinear_tf=True)
        mat = vec.fit_transform(kb["applies_when"] + " " + kb["action"])
        _KB = (kb, vec, mat)
    return _KB


def recommend(text: str, category: str, top_k: int = 3):
    kb, vec, mat = _load()
    sims = cosine_similarity(vec.transform([text]), mat)[0]
    tags = set(nlp.hazard_tags(text))
    scored = []
    for i, row in kb.iterrows():
        score = 0.6 * float(sims[i]) + 0.4 * (1.0 if row["hazard_tag"] in tags else 0.0)
        if row["category"] == category:
            score += 0.05
        scored.append((score, i))
    scored.sort(reverse=True)
    out = []
    for score, i in scored[:top_k]:
        r = kb.loc[i]
        if score < 0.15:  # nothing relevant: do not invent a recommendation
            continue
        out.append({"kb_id": r["kb_id"], "action": r["action"], "horizon": r["horizon"], "owner": r["owner"],
                    "hazard_tag": r["hazard_tag"], "score": round(score, 3)})
    return out
