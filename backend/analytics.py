"""Portfolio-level analytics: counts, trends, recurring patterns, keyphrases, executive summary.

All numbers are computed with pandas; the executive summary is assembled from those numbers with
templates (no LLM), so every sentence can be traced to a computed value.
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from . import nlp, recommender

MIN_RECURRING_COUNT = 3     # a hazard theme must occur at least this often ...
MIN_RECURRING_MONTHS = 2    # ... across at least this many distinct months to count as "recurring"
TREND_RATIO = 1.5           # last-3-months count must be >= 1.5x previous-3-months to be called rising
TREND_MIN = 3               # ... and have at least this many incidents (avoid noise from 1->2)


def enrich(df: pd.DataFrame, model) -> pd.DataFrame:
    rows = []
    for text in df["narrative"]:
        p = model.predict_one(text)
        tags = nlp.hazard_tags(text)
        rows.append({
            "pred_category": p["category"], "pred_severity": p["severity"],
            "category_confidence": p["category_confidence"], "needs_review": p["needs_human_review"],
            "primary_tag": nlp.primary_tag(text), "tags": tags,
        })
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    if "incident_date" in out.columns:
        out["incident_date"] = pd.to_datetime(out["incident_date"], errors="coerce")
        out["month"] = out["incident_date"].dt.to_period("M").astype(str)
    return out


def _trend_label(last3: int, prev3: int) -> str:
    if last3 >= TREND_MIN and last3 >= TREND_RATIO * max(prev3, 1):
        return "rising"
    if prev3 >= TREND_MIN and prev3 >= TREND_RATIO * max(last3, 1):
        return "falling"
    return "stable"


def theme_table(d: pd.DataFrame):
    has_dates = "month" in d.columns and d["month"].notna().any()
    months = sorted(d["month"].dropna().unique()) if has_dates else []
    last3, prev3 = months[-3:], months[-6:-3]
    rows = []
    for tag, g in d.groupby("primary_tag"):
        row = {
            "tag": tag, "label": nlp.tag_label(tag), "count": int(len(g)),
            "share_pct": round(100 * len(g) / len(d), 1),
            "high_severity": int((g["pred_severity"] == "high").sum()),
            "top_category": g["pred_category"].mode().iat[0],
        }
        if has_dates:
            row["distinct_months"] = int(g["month"].nunique())
            row["last3_months"] = int(g["month"].isin(last3).sum())
            row["prev3_months"] = int(g["month"].isin(prev3).sum())
            row["trend"] = _trend_label(row["last3_months"], row["prev3_months"])
            row["first_seen"] = g["incident_date"].min().date().isoformat()
            row["last_seen"] = g["incident_date"].max().date().isoformat()
        if "site_type" in g.columns:
            row["site_types"] = sorted(g["site_type"].dropna().unique().tolist())
        rows.append(row)
    return sorted(rows, key=lambda r: -r["count"])


def recurring_patterns(themes, d):
    out = []
    for t in themes:
        months_ok = t.get("distinct_months", MIN_RECURRING_MONTHS) >= MIN_RECURRING_MONTHS
        if t["tag"] != "other" and t["count"] >= MIN_RECURRING_COUNT and months_ok:
            g = d[d["primary_tag"] == t["tag"]]
            rec = recommender.recommend(g["narrative"].iloc[0], t["top_category"], top_k=1)
            out.append({**t, "examples": g["narrative"].head(2).tolist(),
                        "systemic_action": rec[0]["action"] if rec else None})
    return out


def keyphrases(d, n=10):
    if len(d) < 3:
        return []
    cv = CountVectorizer(preprocessor=nlp.normalize, ngram_range=(2, 3), stop_words="english", min_df=2)
    try:
        m = cv.fit_transform(d["narrative"])
    except ValueError:
        return []
    counts = m.sum(axis=0).A1
    pairs = sorted(zip(cv.get_feature_names_out(), counts), key=lambda x: -x[1])[:n]
    return [{"phrase": p, "count": int(c)} for p, c in pairs]


def executive_summary(d, themes, recurring):
    n = len(d)
    cat = d["pred_category"].value_counts()
    sev = d["pred_severity"].value_counts()
    lines = []
    span = ""
    if "incident_date" in d.columns and d["incident_date"].notna().any():
        span = f" between {d['incident_date'].min().date()} and {d['incident_date'].max().date()}"
    lines.append(f"{n} incident narratives were analysed{span}.")
    mix = ", ".join(f"{k.replace('_', ' ')} {v} ({100 * v / n:.0f}%)" for k, v in cat.items())
    lines.append(f"Category mix (predicted): {mix}.")
    hi = int(sev.get("high", 0))
    lines.append(f"{hi} incidents ({100 * hi / n:.0f}%) were tagged high severity by the rule engine and need priority review.")
    rising = [t for t in themes if t.get("trend") == "rising" and t["tag"] != "other"]
    if rising:
        lines.append("Rising themes (last 3 months vs previous 3): " + "; ".join(
            f"{t['label']} ({t['prev3_months']} -> {t['last3_months']})" for t in rising[:4]) + ".")
    falling = [t for t in themes if t.get("trend") == "falling" and t["tag"] != "other"]
    if falling:
        lines.append("Falling themes: " + "; ".join(
            f"{t['label']} ({t['prev3_months']} -> {t['last3_months']})" for t in falling[:3]) + ".")
    if recurring:
        lines.append(f"{len(recurring)} hazard themes recur (>= {MIN_RECURRING_COUNT} incidents over >= "
                     f"{MIN_RECURRING_MONTHS} months); the largest are " +
                     ", ".join(f"{r['label']} ({r['count']})" for r in recurring[:3]) + ".")
    rv = int(d["needs_review"].sum())
    lines.append(f"{rv} incidents ({100 * rv / n:.0f}%) carry a human-review flag (low category confidence or no severity cue).")
    lines.append("Trends are indicative only: counts per theme are small and no significance test was applied.")
    return " ".join(lines)


def build_dashboard(df: pd.DataFrame, model, dataset_name: str, in_sample: bool):
    d = enrich(df, model)
    themes = theme_table(d)
    recurring = recurring_patterns(themes, d)
    cat_counts = d["pred_category"].value_counts().to_dict()
    sev_counts = {k: int((d["pred_severity"] == k).sum()) for k in ["low", "medium", "high"]}
    matrix = {c: {s: int(((d["pred_category"] == c) & (d["pred_severity"] == s)).sum())
                  for s in ["low", "medium", "high"]} for c in ["safety", "environment", "esg_compliance"]}
    monthly = None
    if "month" in d.columns and d["month"].notna().any():
        mm = d.groupby("month").agg(total=("narrative", "size"),
                                    high=("pred_severity", lambda s: int((s == "high").sum()))).reset_index()
        monthly = mm.to_dict("records")
        by_cat = d.groupby(["month", "pred_category"]).size().unstack(fill_value=0)
        for c in ["safety", "environment", "esg_compliance"]:
            if c not in by_cat.columns:
                by_cat[c] = 0
        for rec in monthly:
            rec.update({c: int(by_cat.loc[rec["month"], c]) for c in ["safety", "environment", "esg_compliance"]})
    top_hi = d[d["pred_severity"] == "high"].sort_values("incident_date", ascending=False) if "incident_date" in d.columns \
        else d[d["pred_severity"] == "high"]
    high_list = []
    for _, r in top_hi.head(8).iterrows():
        rec = recommender.recommend(r["narrative"], r["pred_category"], top_k=1)
        high_list.append({"id": r.get("incident_id"), "narrative": r["narrative"], "category": r["pred_category"],
                          "date": r["incident_date"].date().isoformat() if "incident_date" in d.columns and pd.notna(r["incident_date"]) else None,
                          "action": rec[0]["action"] if rec else None})
    return {
        "dataset": dataset_name,
        "note": ("In-sample demo: the classifier was trained on this same synthetic file, so category counts look cleaner "
                 "than they would on new data." if in_sample else
                 "Out-of-sample: the classifier was not trained on these rows. Only 6 rows, no dates: trends unavailable."),
        "kpis": {"total": int(len(d)), "high": sev_counts["high"],
                 "high_pct": round(100 * sev_counts["high"] / len(d), 1),
                 "needs_review": int(d["needs_review"].sum()), "recurring_themes": len(recurring)},
        "category_counts": cat_counts, "severity_counts": sev_counts, "category_severity_matrix": matrix,
        "monthly": monthly, "themes": themes, "recurring_patterns": recurring,
        "keyphrases": keyphrases(d), "high_severity_incidents": high_list,
        "executive_summary": executive_summary(d, themes, recurring),
    }
