"""Clean FastAPI entry point for LexisMark NER Studio."""

from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ner_engine import ENTITY_SCHEMA, NEREngine, load_benchmark, load_report_text

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="LexisMark — Named Entity Recognition Studio")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

engine: NEREngine | None = None
load_error: str | None = None

try:
    engine = NEREngine(BASE_DIR / "saved_ner_model")
except Exception as exc:
    load_error = str(exc)


class ExtractRequest(BaseModel):
    text: str


@app.get("/")
def serve_home():
    return FileResponse(BASE_DIR / "templates" / "index.html")


@app.get("/api/meta")
def api_meta():
    return {
        "ready": engine is not None,
        "model_name": engine.model_name if engine else "Not Loaded",
        "error": load_error,
        "schema": ENTITY_SCHEMA,
        "benchmark": load_benchmark(BASE_DIR / "saved_ner_model"),
        "report": load_report_text(BASE_DIR / "saved_ner_model"),
    }


@app.post("/api/extract")
def api_extract(payload: ExtractRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {load_error}")
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Please enter text to annotate.")
    return engine.extract(payload.text.strip())


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7862, reload=False)
