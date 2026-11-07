"""FastAPI service.

GET  /health          liveness + model metadata
POST /predict         {"samples": [...2048+ floats...], "fs": 12000, "model": "cnn"|"gbm"}
POST /predict/csv     upload a CSV (one column of samples, or the first numeric column)
GET  /                minimal dashboard (same UI as the GitHub Pages demo, served locally)
"""

from __future__ import annotations

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import FS, WINDOW
from .predictor import Predictor

ROOT = Path(__file__).resolve().parents[2]
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["predictor"] = Predictor()
    yield
    STATE.clear()


app = FastAPI(title="VibeSense bearing fault diagnosis", version="1.0.0", lifespan=lifespan)


class PredictIn(BaseModel):
    samples: list[float] = Field(..., min_length=WINDOW, max_length=200_000)
    fs: float = Field(FS, gt=1000, le=200_000)
    model: Literal["cnn", "gbm"] = "cnn"


def _predict(samples, fs, model) -> dict:
    t0 = time.perf_counter()
    try:
        out = STATE["predictor"].predict(samples, fs=fs, model=model)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return out


@app.get("/health")
def health() -> dict:
    p = STATE.get("predictor")
    return {
        "status": "ok" if p else "loading",
        "window_samples": WINDOW,
        "fs_hz": FS,
        "classes": p.classes if p else None,
        "trained_on": p.meta.get("trained_on") if p else None,
    }


@app.post("/predict")
def predict(body: PredictIn) -> dict:
    return _predict(body.samples, body.fs, body.model)


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), fs: float = Query(FS), model: str = Query("cnn")) -> dict:
    raw = (await file.read()).decode("utf-8", errors="replace")
    try:
        samples = parse_csv(raw)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return _predict(samples, fs, model)


def parse_csv(text: str) -> np.ndarray:
    """First numeric column of a CSV/TSV/plain list; header rows are skipped."""
    vals = []
    for line in io.StringIO(text):
        cell = line.replace(";", ",").replace("\t", ",").split(",")[0].strip()
        if not cell:
            continue
        try:
            vals.append(float(cell))
        except ValueError:
            continue
    if len(vals) < WINDOW:
        raise ValueError(f"CSV has {len(vals)} numeric rows; need >= {WINDOW}")
    return np.asarray(vals, dtype=np.float32)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return FileResponse(ROOT / "index.html")


if (ROOT / "demo").exists():
    app.mount("/demo", StaticFiles(directory=ROOT / "demo"), name="demo")
