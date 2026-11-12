# ESG / Safety Incident Intelligence Dashboard

Turns free-text safety and ESG incident narratives into management insight: **category**, **severity**,
**hazard tags**, a **structured summary**, **recurring patterns / trends** and **suggested preventive actions**.
Built for the Greenko AI/ML case study (renewable-energy O&M context: solar, wind, hydro, BESS/substation).

> Full write-up: [`docs/PROJECT_DOCUMENT.md`](docs/PROJECT_DOCUMENT.md) · Presentation script (follows the guide): [`docs/PRESENTATION_SCRIPT.md`](docs/PRESENTATION_SCRIPT.md) · Demo script & Q&A prep: [`docs/DEMO_WALKTHROUGH.md`](docs/DEMO_WALKTHROUGH.md) · Railway steps: [`docs/DEPLOYMENT_RAILWAY.md`](docs/DEPLOYMENT_RAILWAY.md)

## What it does (and how)

| Capability | Method | Notes |
|---|---|---|
| Category (safety / environment / esg_compliance) | TF-IDF (1-2 grams) + Logistic Regression | probabilities + low-confidence review flag |
| Severity (low / medium / high) | Transparent keyword **rule engine** | shows the matched phrase as evidence; an ML severity model was tried and rejected (see docs) |
| Hazard tags (20 themes) | Regex lexicon | word-boundary safe, regression-tested |
| Incident summary | Extractive / template (locations, indicators, controls already applied) | no LLM, nothing invented |
| Recurring patterns & trends | pandas: theme counts, ≥3 incidents over ≥2 months, last-3 vs previous-3 months | indicative only, no significance test |
| Preventive actions | TF-IDF **retrieval** over a curated 30-action knowledge base | retrieval itself has no LLM; its top actions feed the optional explanation |
| Plain-English explanation (GenAI, optional) | One LLM call that only rewords the computed facts; output validated; template fallback | off unless `LLM_PROVIDER` + `LLM_API_KEY` are set; not an agent |
| Dashboard | Single HTML page, vanilla JS, no CDN | served by FastAPI at `/` |

**Runs with no secrets.** The LLM layer is optional (see `.env.example`); without a key every explanation uses the deterministic template. No embeddings API, no agent.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --reload                        # open http://127.0.0.1:8000
```

No FastAPI available (offline)? `python scripts/dev_server.py 8000` serves the same routes using only the stdlib + pandas/scikit-learn.

```bash
python scripts/generate_dataset.py     # (optional) regenerate data/extended_incidents.csv (seed 42)
python scripts/evaluate.py             # reproduce artifacts/metrics.json + confusion_matrices.png
python scripts/evaluate_explanations.py  # GenAI layer: llm vs template, validator pass rate -> artifacts/genai_eval.json
python -m unittest discover -s tests -v
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Dashboard UI |
| GET | `/api/health` | Liveness probe (used by Railway) |
| GET | `/api/demo` | Project description |
| POST | `/api/run` `{"text": "..."}` | Analyse one narrative |
| POST | `/api/batch` `{"texts": ["...", "..."]}` | Dashboard payload for up to 200 narratives |
| GET | `/api/dashboard?dataset=extended\|bundled` | Dashboard for bundled data |
| GET | `/api/metrics` | Stored evaluation results |
| GET | `/api/genai` | Is the LLM layer on? (provider/model only, never the key) |

```bash
curl -X POST localhost:8000/api/run -H 'Content-Type: application/json' \
  -d '{"text":"Transformer oil spill reaching soil near substation yard, containment breached"}'
```

## Data

* `data/safety_esg_incidents.csv` - the 6 bundled rows (untouched).
* `data/extended_incidents.csv` - **162 synthetic rows** from 36 templates (`scripts/generate_dataset.py`, seed 42), 9 months of dates. Clearly synthetic.
* `data/preventive_actions_kb.csv` - 30 curated actions (author-written good-practice suggestions; need EHS review).

## Headline results (see docs for the caveats)

* Category, 5-fold **GroupKFold by template** (unseen phrasing): macro-F1 **0.67** (majority baseline 0.14; Complement-NB 0.72 - difference within fold noise).
* Severity rule engine: accuracy **0.93** on synthetic data, **optimistic** (same author wrote rules and generator). ML severity: only 0.34 -> rejected.
* Bundled 6 rows (sanity check, not blind): category 6/6, severity 5/6 (S002 label disagreement).
* GenAI layer: 7 tests with a mocked LLM (hallucinated severity / numbers / root cause are rejected). No live LLM run measured yet.

## Enabling the LLM explanation (optional)

```bash
cp .env.example .env      # then set LLM_PROVIDER=anthropic (or openai) and LLM_API_KEY=...
```
On Railway, add the same two keys under **Variables**. Never commit `.env`.

## Limitations (short list)

Tiny, synthetic training data; scores are not real-world accuracy. Severity rules reflect one author's reading. Trend detection on small counts is noisy. Not yet deployed by the author of this repo - follow `docs/DEPLOYMENT_RAILWAY.md`. Decision support only; an EHS professional must confirm.
