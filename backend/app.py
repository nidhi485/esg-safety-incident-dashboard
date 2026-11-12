"""FastAPI entry point (Railway: see Procfile).  Run locally:  uvicorn backend.app:app --reload

Endpoints
  GET  /                  dashboard UI (frontend/index.html)
  GET  /api/health        liveness probe
  GET  /api/demo          project description (kept from the starter)
  POST /api/run           analyse ONE narrative  -> category, severity, tags, summary, actions
  POST /api/batch         analyse a list of narratives -> full dashboard payload
  GET  /api/dashboard     dashboard for the bundled datasets (?dataset=extended|bundled)
  GET  /api/metrics       stored evaluation results (artifacts/metrics.json)
  GET  /api/genai         whether the optional LLM explanation layer is enabled (never exposes the key)
"""
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import service

app = FastAPI(title="ESG / Safety Incident Intelligence Dashboard", version="1.0.0")
BASE = Path(__file__).resolve().parents[1]
FRONTEND = BASE / "frontend" / "index.html"


class Query(BaseModel):
    text: str = Field("", max_length=5000)


class BatchQuery(BaseModel):
    texts: List[str] = Field(default_factory=list)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo")
def demo():
    return {
        "case_study": "ESG / Safety Incident Intelligence Dashboard",
        "status": "running",
        "expected_outcome": "Classifies incident category and severity, extracts recurring patterns and trends, "
                            "summarises incidents and recommends preventive actions.",
        "method": "TF-IDF + logistic regression (category), keyword rule engine (severity, hazard tags), "
                  "TF-IDF retrieval over a curated action knowledge base, optional grounded LLM explanation "
                  "(single call, validated, template fallback; not an agent).",
        "genai": service.genai_status(),
        "endpoints": ["/api/run", "/api/batch", "/api/dashboard", "/api/metrics"],
    }


@app.post("/api/run")
def run_case(query: Query):
    result = service.analyze_text(query.text)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.post("/api/batch")
def run_batch(q: BatchQuery):
    result = service.analyze_batch(q.texts)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/api/dashboard")
def get_dashboard(dataset: str = "extended"):
    if dataset not in ("extended", "bundled"):
        raise HTTPException(status_code=422, detail="dataset must be 'extended' or 'bundled'")
    return service.dashboard(dataset)


@app.get("/api/metrics")
def get_metrics():
    return service.metrics()


@app.get("/api/genai")
def genai():
    return service.genai_status()
