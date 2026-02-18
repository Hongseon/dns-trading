#!/usr/bin/env python3
"""Interactive RAG testing tool.

Provides a REPL for testing the RAG pipeline against real data in Zilliz Cloud.

Usage::

    python scripts/test_rag.py

Commands:
    /search <query>     -- Vector search only (no LLM)
    /embed <text>       -- Embedding only
    /llm <prompt>       -- LLM directly (no retrieval)
    /briefing [type]    -- Briefing generation (daily/weekly/monthly)
    /stats              -- Zilliz collection stats
    /quit               -- Exit
"""

from __future__ import annotations

import sys
import asyncio
import logging
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.rag.chain import RAGChain
from src.rag.retriever import Retriever
from src.rag.embedder import Embedder
from src.rag.generator import Generator
from src.db.zilliz_client import get_client
from src.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def cmd_search(retriever: Retriever, query: str) -> None:
    results, context, sources = retriever.search_and_prepare(query, top_k=5)
    print(f"\n  Found {len(results)} results")
    for i, r in enumerate(results):
        sim = r.get("similarity", 0)
        name = r.get("filename") or r.get("email_subject") or "?"
        print(f"  [{i+1}] sim={sim:.4f} | {r.get('source_type')} | {name}")
        print(f"       {(r.get('content') or '')[:120]}...")
    print(f"\n  Sources: {sources}")


async def cmd_embed(embedder: Embedder, text: str) -> None:
    vec = embedder.embed(text)
    print(f"\n  Dimension: {len(vec)}")
    print(f"  First 5: {vec[:5]}")


async def cmd_llm(generator: Generator, prompt: str) -> None:
    answer = await generator._call_llm(settings.gemini_model, prompt)
    print(f"\n  Response ({len(answer)} chars):")
    print(f"  {answer}")


async def cmd_briefing(briefing_type: str) -> None:
    from src.briefing.generator import BriefingGenerator
    bg = BriefingGenerator()
    content = await bg.generate(briefing_type)
    print(f"\n  Briefing ({len(content)} chars):")
    print(content)


async def cmd_stats() -> None:
    client = get_client()
    for name in ["documents", "sync_state", "briefings"]:
        stats = client.get_collection_stats(name)
        print(f"  {name}: {stats.get('row_count', 0)} rows")


async def cmd_rag(chain: RAGChain, query: str) -> None:
    answer = await chain.run(query)
    print(f"\n  Answer ({len(answer)} chars):")
    print(answer)


async def main() -> None:
    chain = RAGChain()
    retriever = chain.retriever
    embedder = retriever.embedder
    generator = chain.generator

    print("RAG Interactive Test Tool")
    print("Commands: /search, /embed, /llm, /briefing, /stats, /quit")
    print("Or type any question for full RAG.\n")

    while True:
        try:
            user_input = input("RAG> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            print("Bye!")
            break

        t0 = time.time()
        try:
            if user_input.startswith("/search "):
                await cmd_search(retriever, user_input[8:].strip())
            elif user_input.startswith("/embed "):
                await cmd_embed(embedder, user_input[7:].strip())
            elif user_input.startswith("/llm "):
                await cmd_llm(generator, user_input[5:].strip())
            elif user_input.startswith("/briefing"):
                btype = user_input[9:].strip() or "daily"
                await cmd_briefing(btype)
            elif user_input == "/stats":
                await cmd_stats()
            else:
                await cmd_rag(chain, user_input)
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            logger.exception("Command failed")

        elapsed = time.time() - t0
        print(f"\n  [{elapsed:.1f}s]\n")


if __name__ == "__main__":
    asyncio.run(main())
