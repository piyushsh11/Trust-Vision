"""
Unified ASGI entrypoint for Hugging Face Spaces / Docker.
Serves FastAPI backend under /api and all static assets (HTML/JS/CSS) from repo root.
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import app as api_app

app = FastAPI(title="Trust Vision - Unified")

# Backend API available at /api/*
app.mount("/api", api_app)

# Serve static files (HTML, JS, CSS, images) from repository root
app.mount("/", StaticFiles(directory=".", html=True), name="static")


@app.get("/")
def root():
    # default to overview page
    return FileResponse("index1.html")


@app.get("/healthz")
def healthcheck():
    """Lightweight health endpoint for Render/uptime checks."""
    return {"status": "ok"}
