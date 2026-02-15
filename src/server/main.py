"""FastAPI application entry point for the DnS Trading RAG chatbot.

Exposes a health-check endpoint and mounts the KakaoTalk skill router.
Designed for deployment on Render free tier with uvicorn.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.server.skill_handler import router as skill_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler -- log startup and shutdown."""
    logger.info("Starting DnS Trading RAG Bot...")
    yield
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


app.include_router(skill_router)
