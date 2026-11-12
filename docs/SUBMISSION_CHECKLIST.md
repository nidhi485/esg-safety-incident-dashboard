# Submission Checklist (maps to submission_instructions.md)

| # | Required item | Status | Where / what you must do |
|---|---|---|---|
| 1 | Source code, backend + frontend | DONE | `backend/`, `frontend/` |
| 2 | Dataset + preprocessing scripts | DONE | `data/`, `scripts/generate_dataset.py`, `backend/nlp.py` |
| 3 | README with setup/run commands | DONE | `README.md` |
| 4 | Screenshots / demo notes | DONE | `docs/screenshot_*.png`, `docs/DEMO_WALKTHROUGH.md` |
| 5 | Evaluation results + limitations | DONE | `artifacts/metrics.json`, `docs/PROJECT_DOCUMENT.md` sections 6 and 11 |
| 6 | Railway link or deployment evidence | **YOU MUST DO** | Follow `docs/DEPLOYMENT_RAILWAY.md` (about 10 min) |
| 7 | No hard-coded secrets | DONE | LLM key only via env vars; `.env` git-ignored; key never returned (tested) |
| 8 | GenAI explanation (guide Part B §5) | DONE (offline) | `backend/llm_explainer.py`; optional: add a key and run `scripts/evaluate_explanations.py` to measure live acceptance |

## Your 15-minute finish (do in this order)
1. `pip install -r requirements.txt`
2. `python -m unittest discover -s tests -v`  -> expect 21 tests, 0 skipped (the API test now runs). If any fails, fix before deploying.
3. `uvicorn backend.app:app --reload` -> open http://127.0.0.1:8000, click "Sample: spill", toggle "Bundled (6)".
4. `python scripts/evaluate.py` -> confirm numbers still match the project document.
5. Push to GitHub, deploy on Railway, open `/api/health` and `/`.
6. Paste the live URL here: **Live URL: ______________________** and add it to README.md.
7. Save a screenshot of the live dashboard + Railway "Active" status into `docs/`.
8. If the Railway build fails on pinned versions, change `==` to `>=` in requirements.txt and redeploy.
9. Update the "Not completed" and "Deployment" lines in the project document to reflect reality.
10. Rehearse `docs/PRESENTATION_SCRIPT.md` (section by section, matches the guide) once, out loud.
11. *(Optional)* set `LLM_PROVIDER`/`LLM_API_KEY`, run `python scripts/evaluate_explanations.py`, and put the real `llm_acceptance_rate` in the project document §7.
