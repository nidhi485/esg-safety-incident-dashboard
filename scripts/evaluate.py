"""Reproducible evaluation. Run from project root:  python scripts/evaluate.py

Protocol (see docs/PROJECT_DOCUMENT.md section 5):
 1. GroupKFold (5 folds) on the synthetic set, grouped by template_id, so every test fold contains
    templates the model never saw in training -> measures generalisation to NEW phrasing patterns.
 2. External check: train on ALL synthetic rows, predict the 6 bundled rows (original task data).
 3. Baselines: majority class, keyword rules.
Outputs artifacts/metrics.json and PNG confusion matrices.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend import nlp, classifier as C  # noqa: E402

ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

ext = pd.read_csv(ROOT / "data" / "extended_incidents.csv")
bun = pd.read_csv(ROOT / "data" / "safety_esg_incidents.csv")
X, groups = ext["narrative"].values, ext["template_id"].values
gkf = GroupKFold(n_splits=5)


def tfidf(ngram=(1, 2)):
    return TfidfVectorizer(preprocessor=nlp.normalize, ngram_range=ngram, sublinear_tf=True)


def candidates():
    return {
        "majority_class": lambda: DummyClassifier(strategy="most_frequent"),
        "tfidf_complement_nb": lambda: Pipeline([("t", tfidf()), ("c", ComplementNB())]),
        "tfidf_linear_svc": lambda: Pipeline([("t", tfidf()), ("c", LinearSVC(class_weight="balanced"))]),
        "tfidf_logreg_C1": lambda: C.make_pipeline({"C": 1.0}),
        "tfidf_logreg_C10 (chosen)": lambda: C.make_pipeline({"C": 10.0}),
        "tfidf_logreg_C10_unigram": lambda: C.make_pipeline({"C": 10.0, "ngram_range": (1, 1)}),
        # experiment: sklearn's English stop list removes cue words such as "not", "without", "beyond"
        "tfidf_logreg_C10_stopwords_removed": lambda: Pipeline([
            ("t", TfidfVectorizer(preprocessor=nlp.normalize, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
            ("c", LogisticRegression(C=10.0, class_weight="balanced", max_iter=2000))]),
    }


def cv_predict(make, y):
    pred = np.empty(len(y), dtype=object)
    for tr, te in gkf.split(X, y, groups):
        m = make()
        if isinstance(m, DummyClassifier):
            m.fit(np.zeros((len(tr), 1)), y[tr]); pred[te] = m.predict(np.zeros((len(te), 1)))
        else:
            m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred


def summarize(y, pred, labels):
    return {
        "accuracy": round(accuracy_score(y, pred), 3),
        "macro_f1": round(f1_score(y, pred, average="macro", labels=labels), 3),
        "per_class_f1": {l: round(f, 3) for l, f in zip(labels, f1_score(y, pred, average=None, labels=labels))},
    }


results = {"data": {"synthetic_rows": len(ext), "templates": int(ext.template_id.nunique()),
                    "bundled_rows": len(bun)}, "category": {}, "severity": {}}

# ------------------------------------------------ category
y_cat = ext["target_category"].values
for name, make in candidates().items():
    results["category"][name] = summarize(y_cat, cv_predict(make, y_cat), C.CATEGORIES)
cat_best_pred = cv_predict(candidates()["tfidf_logreg_C10 (chosen)"], y_cat)

# ------------------------------------------------ severity
y_sev = ext["target_severity"].values
for name, make in candidates().items():
    results["severity"][name] = summarize(y_sev, cv_predict(make, y_sev), C.SEVERITY_ORDER)
rule_pred = np.array([nlp.rule_severity(t)[0] for t in X])
results["severity"]["keyword_rules_only"] = summarize(y_sev, rule_pred, C.SEVERITY_ORDER)
ml_pred = cv_predict(candidates()["tfidf_logreg_C10 (chosen)"], y_sev)
# tried and REJECTED: ML label raised to 'high' whenever a rule fires
hyb_pred = np.array(["high" if nlp.rule_severity(t)[0] == "high" else m for m, t in zip(ml_pred, X)])
results["severity"]["rejected_hybrid_logreg_plus_rule_floor"] = summarize(y_sev, hyb_pred, C.SEVERITY_ORDER)
# chosen: rules decide; ML disagreement => human-review flag
results["severity"]["rules_final_label (chosen)"] = summarize(y_sev, rule_pred, C.SEVERITY_ORDER)
# review-flag quality: is the rule engine really less reliable when the flag fires?
no_cue = np.array([nlp.rule_severity(t)[1][0].startswith("no high or low cue") for t in X])
results["review_flag_severity"] = {
    "share_flagged_no_cue": round(float(no_cue.mean()), 3),
    "rule_error_rate_when_no_cue": round(float((rule_pred[no_cue] != y_sev[no_cue]).mean()), 3) if no_cue.any() else None,
    "rule_error_rate_when_cue": round(float((rule_pred[~no_cue] != y_sev[~no_cue]).mean()), 3),
    "n_no_cue": int(no_cue.sum()),
}
# category confidence flag, using out-of-fold probabilities
oof = np.zeros((len(X), 3))
for tr, te in gkf.split(X, y_cat, groups):
    m = C.make_pipeline(); m.fit(X[tr], y_cat[tr])
    oof[te] = m.predict_proba(X[te])
conf = oof.max(1); low = conf < C.LOW_CONF
oof_pred = m.classes_[oof.argmax(1)]
results["review_flag_category"] = {
    "share_flagged_low_conf": round(float(low.mean()), 3),
    "category_error_rate_when_low_conf": round(float((oof_pred[low] != y_cat[low]).mean()), 3) if low.any() else None,
    "category_error_rate_when_confident": round(float((oof_pred[~low] != y_cat[~low]).mean()), 3),
}
# high-severity recall: the error that matters most operationally
for k, p in [("logreg_only", ml_pred), ("rejected_hybrid", hyb_pred), ("rules_final", rule_pred)]:
    mask = y_sev == "high"
    results.setdefault("high_severity_recall", {})[k] = round(float((p[mask] == "high").mean()), 3)
results["severity_within_one_level_rules"] = round(float(np.mean(
    [abs(C.SEVERITY_ORDER.index(a) - C.SEVERITY_ORDER.index(b)) <= 1 for a, b in zip(y_sev, rule_pred)])), 3)
# fold-to-fold spread for the main category candidates (is NB vs LogReg a real difference?)
def fold_f1(make, y):
    out = []
    for tr, te in gkf.split(X, y, groups):
        m = make(); m.fit(X[tr], y[tr]); out.append(f1_score(y[te], m.predict(X[te]), average="macro"))
    return [round(v, 3) for v in out]
results["category_fold_macro_f1"] = {n: fold_f1(candidates()[n], y_cat) for n in
                                     ["tfidf_complement_nb", "tfidf_logreg_C10 (chosen)"]}

# ------------------------------------------------ external check on 6 bundled rows
m = C.IncidentModel().fit(ext["narrative"], ext["target_category"])
rows = []
for _, r in bun.iterrows():
    p = m.predict_one(r["narrative"])
    rows.append({"id": r["incident_id"], "narrative": r["narrative"],
                 "cat_true": r["target_category"], "cat_pred": p["category"],
                 "sev_true": r["target_severity"], "sev_pred": p["severity"], "review": p["needs_human_review"]})
bdf = pd.DataFrame(rows)
results["bundled_check"] = {
    "category_accuracy": round(float((bdf.cat_true == bdf.cat_pred).mean()), 3),
    "severity_accuracy": round(float((bdf.sev_true == bdf.sev_pred).mean()), 3),
    "rows": rows,
}

# ------------------------------------------------ confusion matrices + report
results["confusion"] = {
    "category": {"labels": C.CATEGORIES, "matrix": confusion_matrix(y_cat, cat_best_pred, labels=C.CATEGORIES).tolist()},
    "severity_rules": {"labels": C.SEVERITY_ORDER, "matrix": confusion_matrix(y_sev, rule_pred, labels=C.SEVERITY_ORDER).tolist()},
}
mis = ext.assign(cat_pred=cat_best_pred, sev_pred=rule_pred)
mis = mis[(mis.cat_pred != mis.target_category) | (mis.sev_pred != mis.target_severity)]
results["misclassified_examples"] = mis[["incident_id", "narrative", "target_category", "cat_pred",
                                         "target_severity", "sev_pred"]].head(12).to_dict("records")
results["misclassified_count"] = int(len(mis))
# recommender sanity: does the top-1 action belong to the TRUE category / a detected hazard tag? (proxy only)
from backend import recommender  # noqa: E402
cov = cat_hit = tag_hit = 0
for t, c in zip(ext["narrative"], ext["target_category"]):
    r = recommender.recommend(t, c, top_k=1)
    if r:
        cov += 1
        cat_hit += int(any(k["category"] == c for k in [recommender._load()[0].set_index("kb_id").loc[r[0]["kb_id"]]]))
        tag_hit += int(r[0]["hazard_tag"] in nlp.hazard_tags(t))
results["recommender_proxy"] = {"coverage_top1": round(cov / len(ext), 3),
                                "top1_same_category_as_true": round(cat_hit / max(cov, 1), 3),
                                "top1_tag_present_in_narrative": round(tag_hit / max(cov, 1), 3),
                                "caveat": "Proxy only. No expert judgement of action quality was collected."}
results["notes"] = [
    "Synthetic data authored by the project author; scores show learnability of label conventions, not real-world accuracy.",
    "GroupKFold by template_id: test templates are unseen, but different templates can still be semantically close.",
    "Severity rules were written by the same author as the generator; rule performance is therefore optimistic.",
    "The 6 bundled rows were read while designing lexicons, so they are a sanity check, not a blind test.",
]
(ART / "metrics.json").write_text(json.dumps(results, indent=2))

# ------------------------------------------------ figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, (title, key) in zip(ax, [("Category (LogReg, GroupKFold)", "category"), ("Severity (rule engine)", "severity_rules")]):
        cm = np.array(results["confusion"][key]["matrix"]); lab = results["confusion"][key]["labels"]
        a.imshow(cm, cmap="Greens")
        a.set_xticks(range(len(lab))); a.set_yticks(range(len(lab)))
        a.set_xticklabels(lab, rotation=20); a.set_yticklabels(lab)
        a.set_xlabel("predicted"); a.set_ylabel("actual"); a.set_title(title, fontsize=10)
        for i in range(len(lab)):
            for j in range(len(lab)):
                a.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max() / 2 else "white")
    plt.tight_layout(); plt.savefig(ART / "confusion_matrices.png", dpi=140); plt.close()
except Exception as e:  # figures are optional
    print("figure skipped:", e)

print(json.dumps({k: results[k] for k in ["category", "severity", "high_severity_recall",
                                           "severity_within_one_level_rules", "review_flag_severity", "review_flag_category", "category_fold_macro_f1", "misclassified_count"]}, indent=1))
print("\nBUNDLED CHECK"); print(bdf.drop(columns=["narrative"]).to_string(index=False))
print("category acc", results["bundled_check"]["category_accuracy"], "severity acc", results["bundled_check"]["severity_accuracy"])
