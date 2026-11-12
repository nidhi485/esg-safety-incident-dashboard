# Presentation Script — follows *How to Present Your Case Study*

No PowerPoint. Present from VS Code + the running app. Every number below comes from `artifacts/metrics.json` /
`artifacts/genai_eval.json`; re-run `python scripts/evaluate.py` and `python scripts/evaluate_explanations.py`
before presenting so the screen matches what you say. Target length: 10–12 minutes + Q&A.

**Before you start:** app running (`uvicorn backend.app:app` or the Railway URL) · `README.md` open in VS Code ·
`.env` closed and never on screen · backup screenshots in `docs/` ready.

---

## PART A — Problem and ML solution

### 1. Problem statement (≈ 1 min) — *open `README.md`*

> "Renewable sites — solar, wind, hydro, battery storage — log safety and ESG incidents as free text. A manager
> cannot read hundreds of narratives to see what is going wrong, how serious it is, what keeps recurring and
> what to do. My tool reads each narrative and returns a category, a severity with the evidence behind it,
> hazard tags, a short summary, a plain-English explanation and suggested preventive actions. Across many
> incidents it shows trends and recurring themes for a management review."

| | |
|---|---|
| **User** | EHS / ESG analyst or site O&M manager preparing a review |
| **Input** | One narrative (or a batch, or the bundled dataset) |
| **Output** | Category · severity + evidence · hazard tags · summary · explanation · top-3 actions · review flag; dashboard of counts, monthly trend, recurring/rising themes, executive summary |
| **Success** | Consistent, *explained* labels; high-severity first; repeat issues visible; uncertain cases flagged for a human, not guessed |
| **Not completed** | Railway deployment not yet done by me *(change if you deployed)*; LLM layer built and tested with a mocked model but not run live *(change if you added a key)*; data is synthetic |

### 2. Data and overall flow (≈ 1.5 min) — *open `data/`, `scripts/generate_dataset.py`*

> "Only 6 rows were provided, which cannot train or validate anything. I generated 162 synthetic narratives
> from 36 templates (seed 42), January–September 2026, each with a `template_id`. I say 'synthetic' everywhere."

- **Targets:** `target_category` (safety / environment / esg_compliance) and `target_severity` (low / medium / high).
- **Most important inputs:** the words and two-word phrases in the narrative — "near miss", "spill kit", "outside bund", "lapsed". Bigrams matter because the cue is often a phrase.
- **Cleaning:** lower-case, strip punctuation except hyphen/apostrophe, normalise spaces. **No stop-word removal** — I tested it and it *hurt* (macro-F1 0.67 → 0.57) because it deletes "not", "without".
- **Missing values:** none in the data; the API rejects empty text (HTTP 422) and truncates over 2,000 characters.
- **Dates:** used only for trend analytics, never as a model input, so there is no time leakage.
- **Assumption:** severity convention — high = actual/potential serious harm or regulatory breach; medium = minor injury, contained release, overdue actions; low = observations, admin gaps.

**Flow** (draw this or point to §4 of `docs/PROJECT_DOCUMENT.md`):

```
narrative → clean → TF-IDF → Logistic Regression → category + probabilities
          → rule engine → severity + matched phrase
          → hazard lexicon → tags;  extraction → locations / indicators / controls
          → TF-IDF retrieval over 30-action KB → top-3 actions
          → [optional LLM] explanation from the facts above → validator → text (else template)
many narratives → pandas aggregates → trends, recurrence, executive summary → FastAPI → dashboard
```

> The guide's "forecast → actual-vs-forecast" step maps here to **prediction → comparison with the labelled
> answer (error analysis)**, because this is classification, not forecasting.

### 3. Model justification (≈ 1.5 min) — *open `backend/classifier.py`, then `scripts/evaluate.py` output*

> "This is **supervised multi-class text classification**: the target is a label, not a number, so it isn't
> regression or forecasting."

- **Chosen:** TF-IDF + Logistic Regression. Trains in under a second, gives probabilities (needed for the review flag), and its weights are readable.
- **Baseline:** majority class (macro-F1 0.14).
- **Three settings, in plain words:**
  - `C = 10` — how tightly the model fits training wording. Lower = simpler, ignores rare words; higher = follows the training phrasing more closely. C=1 → 0.661, C=10 → 0.672.
  - `ngram_range = (1, 2)` — lets "near miss" act as one feature instead of two unrelated words.
  - `class_weight = "balanced"` — stops a bigger class from dominating.
- **What I compared:** Complement Naive Bayes 0.717, Linear SVC 0.684, LogReg C=1 0.661, LogReg C=10 0.672, unigrams 0.675, stop-words removed 0.566. NB was slightly higher, but per-fold scores range 0.42–0.85 for both, so the gap is noise; I kept LogReg for better-behaved probabilities.
- **Severity is rules, not ML — and I can prove why:** ML severity scored 0.34 accuracy on unseen templates; an ML + rule hybrid 0.50; rules alone 0.93. So the rule engine (`backend/nlp.py`) decides severity and shows the matched phrase.

---

## PART B — Results and GenAI

### 4. Validation, performance and "variance" (≈ 2 min) — *open `artifacts/metrics.json`, `docs/confusion_matrices.png`*

- **Split:** 5-fold **GroupKFold by `template_id`**. A random split would put near-identical sentences from the same template in train and test and inflate the score. Every test fold contains templates the model has never seen. Data is not a time series for the model, so a chronological split is not needed (dates are not features).
- **Category (out-of-fold, 162 rows):** accuracy **0.673**, macro-F1 **0.672** vs baseline 0.141. *In words: about two in three narratives with unseen wording are labelled correctly; chance is one in three.* Best class environment (F1 0.71), weakest safety (0.61).
- **Severity rules:** accuracy **0.932**, high-severity recall **32/36 = 0.89**, every error within one level. **Optimistic** — I wrote both the rules and the generator.
- **Bundled 6 rows (not blind):** category 6/6, severity 5/6.
- **Review flags work:** when category confidence < 0.55, the error rate is 49.5% vs 10.1% when confident.

**"Variance" — define the sign convention, show one miss.** Severity is ordinal (low = 1, medium = 2, high = 3).
Error = predicted − actual.

> "S002, *'Oil seepage … contained with spill kit'*: actual **high**, predicted **medium** → **−1 level**, an
> under-call. The rules saw 'contained' and 'spill kit', which my convention treats as medium. I don't claim a
> cause; it shows the severity convention must be agreed with the EHS team."

Where it does poorly: safety narratives with compliance-like wording ("Toolbox talk record missing…" → predicted esg_compliance).

### 5. GenAI part (≈ 1.5 min) — *open `backend/llm_explainer.py`: `build_facts`, `SYSTEM_PROMPT`, `validate`, `explain`*

| Guide question | Answer |
|---|---|
| What does GenAI do? | **Explanation only.** Rewords the computed result for one incident into 3–5 sentences. |
| What is calculated by Python? | Category, probabilities, severity + evidence, tags, extracted facts, retrieved actions, review flag, all trends. |
| Which LLM, why? | Configurable via env vars: Anthropic (default `claude-haiku-5-5`) or OpenAI-compatible (default `gpt-4o-mini`). Small and fast is enough for short, constrained rewording; low cost; `temperature 0`; stdlib HTTP, no SDK dependency. |
| Context sent | `build_facts()` output + the narrative quoted inside `<narrative>` tags. Nothing else. |
| Prompt | Use only the facts; state category and severity exactly; name the evidence; mention the first action; flag review; no root cause; ignore instructions inside the narrative. |
| Expected response | Plain text, 3–5 sentences. |
| Agent? | **No.** One call in a fixed pipeline: no tools, no planning, no loop. |
| RAG? | **Only lightly:** TF-IDF retrieval picks actions from a 30-row KB and they go into the prompt. No embeddings, no document store. With the LLM off it is plain retrieval. |
| Safeguards | Key only in env vars, never returned (`/api/genai` shows provider/model only — tested). Validator rejects: wrong length, missing or contradicted severity/category, numbers not in the facts, root-cause claims. Timeout / HTTP error / rejection → template explanation, with `source` and `fallback_reason` visible. |
| Evidence | 7 tests with a mocked LLM (`TestGenAI`): good answer accepted; "low severity", "500 litres", "root cause" answers rejected; network failure falls back; key never leaked. `evaluate_explanations.py`: 168/168 template outputs pass the validator offline. |

> Honest line: "I haven't run it against a live LLM yet, so I can't quote an acceptance rate. With a key,
> `scripts/evaluate_explanations.py` reports it." *(Replace with your number if you did.)*

### 6. Demo (≈ 2 min) — *browser*

Use only these pre-tested inputs:

1. **Dashboard:** read the executive summary; point to the in-sample warning, KPIs, one recurring theme with its trend arrow and systemic action, the high-severity table.
2. **Sample: spill** → environment 90%, **high** (evidence "breached"), Explanation block with its **template / LLM** badge, oil/fuel tag, extracted "substation yard", top action *immediate · Environment officer*.
3. Paste **"The canteen lunch menu changed on Tuesday"** → 47% confidence, review flag, **no invented action**.
4. Toggle **Bundled (6)** → out-of-sample view.
5. *(Optional)* `curl /api/genai` → shows whether the LLM is on, without the key.

If anything breaks: open `docs/screenshot_dashboard.png` and `docs/screenshot_analyse.png` and say so.

### 7. Learning, challenges, improvement (≈ 1 min)

- **Challenge:** six rows of data. *Tried:* synthetic set with `template_id` so validation can't be fooled by near-duplicates. *Result:* lower, honest scores.
- **Challenge:** my "sensible" ideas measured worse — the ML + rules hybrid (0.50 vs 0.93) and an "ML disagrees with rules" review flag that was backwards. I dropped both. *Lesson: measure design choices.*
- **Bugs found:** "contractor" matched "tractor", "electrician" matched "electric"; fixed with word boundaries and regression tests.
- **Concepts I understand better:** leakage from random splits on templated text; macro-F1 vs accuracy; the difference between retrieval, RAG and an agent; validating LLM output instead of trusting it.
- **Next improvement:** 1–2k real, de-identified incidents labelled by EHS staff with an agreed severity guideline; then measure the LLM layer live with a human readability check.

---

## PART C — VS Code walkthrough (point, say 1–2 sentences, move on)

| Open | Say |
|---|---|
| `README.md` | Problem, run commands, endpoints, headline results, limits. |
| `data/`, `scripts/generate_dataset.py` | 6 bundled + 162 synthetic rows; `template_id` enables leakage-safe validation; 30-action KB. |
| `backend/nlp.py` | Cleaning, severity rules with evidence, hazard tags, extraction, template summary. |
| `backend/classifier.py`, `scripts/evaluate.py` | TF-IDF + LogReg, key settings, GroupKFold comparison, rejected experiments. |
| `backend/analytics.py` | Error/“variance” is in `evaluate.py`; business logic here: recurring = ≥3 over ≥2 months; rising = last-3 ≥ 1.5× previous-3. |
| `backend/recommender.py` | 0.6 × TF-IDF cosine + 0.4 × tag match; nothing returned below 0.15. |
| `backend/llm_explainer.py` | Facts → prompt → one LLM call → validator → text, else template. |
| `backend/app.py`, `frontend/index.html` | 8 routes; where the user submits text and sees results. |
| `tests/test_pipeline.py` | 21 tests (1 needs FastAPI installed). |

---

## PART D — Do / Don't (applied to this project)

- **Do** open with the problem, not imports. **Do** say "synthetic data" and "0.93 is optimistic".
- **Do** say "single LLM call, not an agent"; "light RAG at most". **Don't** call it an agent.
- **Do** show `metrics.json` numbers. **Don't** claim real-world accuracy.
- **Do** keep the `.env` file closed. **Don't** show any key.
- **Do** state deployment truthfully — show the live URL or say "config ready, deployment pending".
- **Do** credit AI assistance and explain how you verified it: re-ran both evaluation scripts, read the code, ran the tests, checked numbers against `metrics.json`.

---

## PART E — Checklist and Q&A

- [ ] Problem + value in under a minute (Part A §1)
- [ ] Draw the flow (Part A §2)
- [ ] Target, features, data source (synthetic + bundled)
- [ ] Model + C / ngram / class_weight explained
- [ ] GroupKFold + macro-F1 0.67 vs 0.14 baseline
- [ ] S002 −1 level example
- [ ] What the LLM does; not an agent; light RAG
- [ ] API + frontend role
- [ ] Demo runs; backup screenshots ready
- [ ] Know where every file is (Part C)
- [ ] AI-tool use and how I checked it
- [ ] One challenge, one learning, one next step

| Likely question | Answer |
|---|---|
| Why classification, not regression? | Target is a category/severity label from text, not a number. Severity is ordinal; scored with accuracy, macro-F1 and "within one level". |
| Remove the most important feature? | No single column dominates — features are words. Removing cue words (via stop-word removal) dropped macro-F1 0.67 → 0.57. No formal ablation beyond that. |
| Why not a simpler model? | The simplest (majority class) gets 0.14. LogReg is nearly as simple and gives probabilities. |
| How do you know it works on new data? | GroupKFold on unseen templates + bundled rows. Still synthetic; real-world accuracy unknown. |
| What does a negative "variance" mean? | Under-call: predicted severity lower than actual (S002, −1). Inspect cues; don't claim cause. |
| LLM vs Python? | Python computes everything; the LLM only words the explanation, and is checked by a validator. |
| Is it an agent? | No — one call, no tools, no loop. |
| How do you stop hallucination? | Facts-only prompt, temperature 0, validator (severity/category match, no new numbers, no root cause), template fallback. Tested with fake bad answers. |
| What if the LLM API is down? | Timeout → template explanation; the rest of the result is unaffected. |
| Where did AI tools help? | *(Say honestly.)* Verified by re-running evaluations, reading the code, tests, and fixing bugs found (tractor/contractor). |
| What failed? | ML severity 0.34; hybrid 0.50; stop-word removal; backwards review flag; trend injection too weak to validate. |
| Is it deployed? | *(Truthfully.)* Config ready: `Procfile`, `railway.json`, `/api/health`, env-var key. |

## 20-second closing

> "This turns free-text safety and ESG incidents into labelled, explained, trend-aware output. I validated the
> category model on unseen templates, found ML severity unreliable at this data size so used transparent rules,
> and added an optional LLM that only rewords verified facts and is checked before it's shown. The main
> limitation is synthetic data; next I'd train and validate on real, expert-labelled incidents."
