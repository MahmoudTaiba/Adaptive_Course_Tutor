"""
src/config.py — Shared configuration for both Part A and Part B.
Import constants and helpers from here; never hardcode values elsewhere.
"""

from __future__ import annotations

# ─── Embedding / Qdrant (Part A) ───────────────────────────────────────────────

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
COLLECTION = "course"
QDRANT_PATH = "./qdrant_data"


def get_client():
    """Return a QdrantClient pointed at the local on-disk store."""
    from qdrant_client import QdrantClient
    return QdrantClient(path=QDRANT_PATH)


def get_embed_model():
    """Load the sentence-transformer embedding model (cached after first call)."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


# ─── LLM (Part B) ─────────────────────────────────────────────────────────────

LLM_MODEL = "llama3.1"          # Ollama model — run: ollama pull llama3.1
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"   # Free API fallback


def get_llm(temperature: float = 0.3, max_tokens: int = 1024):
    """
    Instantiate the shared LLM.

    Priority:
    1. Ollama (production — matches partner's stack, needs `ollama pull llama3.1`)
    2. Groq (dev fallback — free API, set GROQ_API_KEY in .env)

    Both Part A and Part B import from here — one place to swap models.
    """
    import os

    # Try Ollama first (production stack)
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        if r.status_code == 200:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=LLM_MODEL,
                temperature=temperature,
                num_predict=max_tokens,
            )
    except Exception:
        pass

    # Fallback: Groq (dev environment / no local Ollama)
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=GROQ_FALLBACK_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
