# Deploying to Railway

**Status:** the code is deployment-ready (`Procfile`, `railway.json`, `/api/health`, `$PORT` binding), but it has **not** been
deployed by the author of this package. Deploying and recording the live URL is a required submission step - do it yourself using
the steps below and paste the URL + a screenshot into the README / submission.

## Option A - GitHub (recommended)
1. Create a GitHub repo and push this folder (`.env` is git-ignored; the project needs no secrets).
   ```bash
   git init && git add . && git commit -m "ESG incident dashboard"
   git branch -M main && git remote add origin <your-repo-url> && git push -u origin main
   ```
2. Railway dashboard -> **New Project -> Deploy from GitHub repo** -> select the repo.
3. Railway detects Python from `requirements.txt` and uses the start command in `railway.json` / `Procfile`:
   `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
4. **Settings -> Networking -> Generate Domain** to get a public URL.
   *(Optional)* **Variables** -> add `LLM_PROVIDER` (`anthropic` or `openai`) and `LLM_API_KEY` to switch on the LLM explanation. Skip this and the app still works with template explanations. Check `/api/genai` afterwards.
5. Verify: `https://<your-domain>/api/health` returns `{"status":"ok"}`; open `/` for the dashboard; test `/api/run`.
6. Save evidence: screenshot of the deployment (green "Active"), the live dashboard, and the URL in the README.

## Option B - Railway CLI
```bash
npm i -g @railway/cli && railway login
railway init && railway up && railway domain
```

## Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Build fails on pandas/numpy | Pinned versions need Python 3.9-3.12. `.python-version` pins 3.11. Keep it. |
| `ModuleNotFoundError: backend` | Start command must run from the repo root (it does by default). |
| 502 / app not reachable | App must bind `0.0.0.0:$PORT` (already in start command). |
| First request slow | Model trains at startup (<1 s locally); give the health check up to 60 s (set in `railway.json`). |
| Dashboard shows "Could not load dashboard" | You opened `frontend/index.html` as a file. Open the deployed `/` URL instead. |

## Known risks (not verified by the author)
* Pinned library versions (`requirements.txt`) were **not** test-installed in the authoring sandbox (no network); the code was
  tested on newer pandas / scikit-learn, and uses only basic APIs. If the build fails, relax pins to `>=` and redeploy.
* `backend/app.py` was syntax-checked, but run only through the framework-independent `service.py` layer (tests) and
  `scripts/dev_server.py`, because FastAPI could not be installed in the sandbox. Run `python -m unittest discover -s tests -v`
  locally after `pip install -r requirements.txt` - the API tests un-skip automatically and will confirm the real app.
* Free-tier services sleep or restart; training at startup keeps the app stateless.
