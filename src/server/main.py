"""FastAPI application entry point for the DnS Trading RAG chatbot.

Exposes a health-check endpoint and mounts the KakaoTalk skill router.
Designed for deployment on Render free tier with uvicorn.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.server.admin import router as admin_router
from src.server.skill_handler import router as skill_router
from src.server.warmup import (
    ensure_rag_warmup_started,
    get_rag_warmup_status,
    start_rag_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler -- log startup, warm dependencies, shutdown."""
    logger.info("Starting DnS Trading RAG Bot...")
    warmup_task = start_rag_warmup()
    app.state.rag_warmup_task = warmup_task

    try:
        yield
    finally:
        if not warmup_task.done():
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task
    logger.info("Shutting down...")


app = FastAPI(
    title="DnS Trading RAG Bot",
    description="KakaoTalk channel chatbot backed by a RAG pipeline",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health-check endpoint (also used by UptimeRobot to prevent Render sleep)."""
    return {"status": "ok"}


@app.get("/warmup")
async def warmup():
    """Readiness endpoint that triggers background warmup and returns status."""
    ensure_rag_warmup_started(app.state)
    rag_status = get_rag_warmup_status(app.state)

    if rag_status in {"ready", "warming"}:
        return {"status": "ok", "rag": rag_status}

    status_code = 503 if rag_status in {"failed", "skipped"} else 500
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "rag": rag_status},
    )


app.include_router(skill_router)
app.include_router(admin_router)
