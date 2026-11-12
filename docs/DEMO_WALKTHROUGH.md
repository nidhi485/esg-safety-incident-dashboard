# Demo Walkthrough & Q&A Preparation

Follows the *How to Present Your Case Study* guide: present from VS Code + the running app (no PowerPoint). You may use AI tools,
but you must be able to **explain, justify and defend** everything. Rehearse once; keep a backup screenshot (`docs/screenshot_*.png`).

## Before you start (2 min)
- [ ] `pip install -r requirements.txt` done; `uvicorn backend.app:app` running; browser tab open on `/` (or the Railway URL once deployed).
- [ ] `python scripts/evaluate.py` re-run so `artifacts/metrics.json` matches the code you present.
- [ ] No `.env` / keys on screen (none are needed). Close unrelated files.
- [ ] Pre-tested demo inputs (below) in a text file.
- [ ] Decide the truthful deployment statement: either show the live Railway URL, or say "deployment config is ready, deployment pending".

## 6-minute script

**1 · Problem (0:00-1:00)** *(open README)*
"Sites log safety and ESG incidents as free text. Managers can't read hundreds of them. This tool labels each incident, explains the severity, finds recurring themes and suggests preventive actions. The user is an EHS/ESG analyst. Success = consistent, explainable labels, high-severity items surfaced, uncertain cases flagged for a human. An optional LLM only rewords the computed result into a readable explanation; it is not an agent. I have not yet deployed it to Railway *(adjust if you have)*."

**2 · Data and flow (1:00-2:00)** *(open `data/`, `generate_dataset.py`)*
"Only 6 bundled rows, so I generated 162 synthetic narratives from 36 templates with dates, and said so everywhere. Target = category and severity. Flow: text → category model + severity rules + hazard tags + summary + retrieved actions → analytics → dashboard."

**3 · Model justification (2:00-3:00)** *(open `classifier.py`, then `evaluate.py` output)*
"This is multi-class text classification. TF-IDF + Logistic Regression, baseline = majority class. Three settings: `C` controls how tightly it fits training wording; `ngram_range=(1,2)` lets 'near miss' be one feature; `class_weight=balanced` stops big classes dominating. I compared Naive Bayes, SVC and variants; NB was slightly higher but within fold noise, and I kept LR for probabilities."

**4 · Validation (3:00-4:00)** *(show `artifacts/metrics.json` / confusion image)*
"Random splits would leak near-duplicate sentences, so I used GroupKFold by template: test templates are unseen. Category macro-F1 0.67 vs 0.14 baseline. Severity: ML scored 0.34, so rules are the authority (0.93, but optimistic because I wrote both). Error example: S002 is labelled high in the source; my rules say medium: a labelling-convention gap."

**5 · GenAI part (4:00-4:30)** *(open `llm_explainer.py`: `build_facts`, `SYSTEM_PROMPT`, `validate`)*
"Python computes every label. The LLM gets only those verified facts plus the retrieved actions and writes 3-5 sentences. A validator rejects answers that change the severity, add numbers or claim a root cause, and then the template is used. It is one call, so not an agent; retrieval is TF-IDF over a table, so at most a light RAG step. Without a key it runs on the template, which is what you see now."

**6 · Live demo (4:30-5:30)** *(browser)*
1. Dashboard: read the executive summary; point at the in-sample warning banner, KPIs, a recurring pattern + systemic action.
2. Click **Sample: spill** → category 90% environment, severity *high* with evidence "breached", the Explanation block with its LLM/template badge, tags, extracted facts, actions.
3. Paste **"The canteen lunch menu changed on Tuesday"** → review flag, no invented action.
4. Toggle **Bundled (6)** → out-of-sample view.

**7 · Challenge, learning, next step (5:30-6:00)**
"Biggest surprise: my 'ML + rules' hybrid and my 'ML disagrees' flag both measured worse than the simple option, so I dropped them. Also found two substring bugs (contractor/tractor, electrician/electric) and added tests. Next: real labelled data and an agreed severity guideline with the EHS team."

## 20-second closing
"This turns free-text incidents into labelled, explained, trend-aware dashboard output. I validated the category model on unseen templates, found ML severity unreliable at this data size, so I used transparent rules and flag uncertain cases. The main limitation is synthetic data; next I'd train and validate on real, expert-labelled incidents."

## Likely questions and honest answers

| Question | Answer |
|---|---|
| Why classification, not regression? | The target is a label from text (category, severity class), not a number. Severity is ordinal; I treat it with rules and score it with accuracy / macro-F1 and "within one level". |
| What if your most important feature were removed? | Features are words/phrases, not one dominant column. Dropping cue words like "oil/spill" would reduce environment recall; I haven't run a formal ablation. Stop-word removal, which removed cue words like "not/without", dropped macro-F1 from 0.67 to 0.57. |
| Why this model over a simpler one? | The simple baseline (majority class) scores 0.14. Logistic Regression is cheap, gives probabilities and weights. NB was slightly better but within noise. |
| How do you know it works on new data? | GroupKFold by template (unseen templates) plus a 6-row bundled check. That's better than a random split but still synthetic; real-world accuracy is unknown. |
| Negative variance / error meaning? | Here: severity under-call, e.g. actual high / predicted medium (S002). I would inspect narrative cues, not claim a cause. |
| What is generated by an LLM vs calculated? | Python computes category, severity, tags, actions, trends. The LLM (optional) only writes the explanation paragraph from those facts; a validator checks it, else a template is used. |
| Is it an agent? | No. One LLM call in a fixed pipeline, no tool choice or loop. |
| Did you use RAG? | Only lightly: TF-IDF retrieval picks actions from a 30-row KB and those go into the prompt. No embeddings, no document store. With the LLM off there is no generation. |
| How do you stop hallucinations? | Facts-only prompt, temperature 0, then a validator: severity/category must match, no new numbers, no root-cause claim. Tested with fake bad answers. |
| Which LLM and why? | Configurable; default a small fast model (Claude Haiku / GPT-4o-mini): short constrained rewording needs speed and low cost, not deep reasoning. |
| Where did AI tools help and how did you verify? | State honestly what you used AI for; verification: re-ran evaluation, read code, tests (20 pass, 1 needs FastAPI), discovered/fixed bugs, reviewed numbers against `metrics.json`. |
| What failed? | ML severity (0.34), hybrid (0.50), stop-word removal, ML-disagreement flag; trend injection too weak to validate detection. |
| Why is 0.93 severity not impressive? | I wrote both the generator and the rules; it's consistency, not independent accuracy. |
| Is it deployed? | Answer truthfully: config ready (`Procfile`, `railway.json`, health check). Say whether you deployed it and show the URL. |
| What would break in production? | New wording, negation ("no fire"), multi-hazard narratives, disputed severity conventions, small-count trends, no auth/audit trail. |

## Do / Don't reminders
- Do start with the problem and outcome; don't start with imports or scroll code.
- Do show actual metrics and say what they mean; don't claim accuracy you haven't measured.
- Do keep the demo to tested inputs; don't fix installs live.
- Do be explicit that data is synthetic, that the LLM only rewords computed facts (single call, not an agent), and whether you ran it live.
