"""
FastAPI backend for the Drug Interaction Checker Agent.

Run locally:
    uvicorn app.api.main:app --reload
Then open http://localhost:8000

Endpoints:
    GET  /                -> serves the frontend (static index.html)
    POST /check           -> { "medications": ["warfarin","aspirin"] }
    GET  /health          -> health probe
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agents_sdk import run_agent
from app.crew.agents import DISCLAIMER

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"

app = FastAPI(
    title="Drug Interaction Checker Agent",
    version="1.0.0",
    description="Multi-agent demo (OpenAI Agents SDK + CrewAI). " + DISCLAIMER,
)


class CheckRequest(BaseModel):
    medications: list[str] = Field(
        ..., min_length=1, description="Patient's medication list (free-text names)."
    )


class CheckResponse(BaseModel):
    report: dict[str, Any]
    disclaimer: str = DISCLAIMER
    source: str = "bundled offline knowledge base (demo)"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "disclaimer": DISCLAIMER,
    }


@app.post("/check", response_model=CheckResponse)
async def check(req: CheckRequest):
    meds = [m.strip() for m in req.medications if m.strip()]
    if not meds:
        raise HTTPException(status_code=400, detail="No medications provided.")
    report = await run_agent(meds)
    return CheckResponse(report=report.model_dump())


# --- Serve the frontend ---
if (FRONTEND / "index.html").exists():
    @app.get("/")
    async def index():
        return FileResponse(FRONTEND / "index.html")

    # Mount static assets (css/js) if present
    if (FRONTEND / "static").is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")
else:
    @app.get("/")
    async def index_missing():
        return JSONResponse(
            {
                "message": "API is running. Frontend not found. Use POST /check.",
                "disclaimer": DISCLAIMER,
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
