"""Framework-independent service layer. FastAPI (app.py) and the stdlib dev server both call this."""
import json
from pathlib import Path

import pandas as pd

from . import analytics, classifier, llm_explainer, nlp, recommender

BASE = Path(__file__).resolve().parents[1]
MAX_TEXT_LEN = 2000
MAX_BATCH = 200

DISCLAIMER = ("Decision support only. Category and recommendations are model/knowledge-base suggestions and "
              "severity is a keyword-rule estimate; an EHS professional must confirm before action.")


def analyze_text(text: str):
    text = (text or "").strip()
    if not text:
        return {"error": "Empty narrative. Provide the incident description in 'text'."}
    if len(text) > MAX_TEXT_LEN:
        text = text[:MAX_TEXT_LEN]
    model = classifier.get_model()
    p = model.predict_one(text)
    tags = nlp.hazard_tags(text)
    tag = tags[0] if tags else "other"
    result = {
        "input": text,
        "category": p["category"], "category_confidence": p["category_confidence"],
        "category_probabilities": p["category_probabilities"],
        "severity": p["severity"], "severity_method": p["severity_method"], "severity_evidence": p["severity_evidence"],
        "hazard_tags": [{"tag": t, "label": nlp.tag_label(t)} for t in tags],
        "extracted": nlp.extract_structure(text),
        "summary": nlp.summarize_incident(text, p["category"], p["severity"], tag),
        "recommended_actions": recommender.recommend(text, p["category"]),
        "needs_human_review": p["needs_human_review"], "review_reasons": p["review_reasons"],
        "limitations": [DISCLAIMER, "Model trained on a small synthetic set; unfamiliar wording can be mis-classified."],
    }
    # GenAI step runs LAST and only rewrites the verified facts above (template fallback if off/invalid).
    result["explanation"] = llm_explainer.explain(result)
    return result


def genai_status():
    return llm_explainer.status()


def analyze_batch(texts):
    texts = [t for t in (texts or []) if isinstance(t, str) and t.strip()][:MAX_BATCH]
    if not texts:
        return {"error": "Provide a non-empty list of narratives in 'texts'."}
    model = classifier.get_model()
    df = pd.DataFrame({"narrative": texts})
    return analytics.build_dashboard(df, model, "uploaded_batch", in_sample=False)


def dashboard(dataset: str = "extended"):
    model = classifier.get_model()
    if dataset == "bundled":
        df = pd.read_csv(BASE / "data" / "safety_esg_incidents.csv")
        return analytics.build_dashboard(df, model, "bundled (6 rows)", in_sample=False)
    df = pd.read_csv(BASE / "data" / "extended_incidents.csv")
    return analytics.build_dashboard(df, model, "extended synthetic (162 rows)", in_sample=True)


def metrics():
    p = BASE / "artifacts" / "metrics.json"
    if not p.exists():
        return {"error": "metrics.json not found. Run: python scripts/evaluate.py"}
    return json.loads(p.read_text())
