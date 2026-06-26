"""
src/ingest.py — PDF ingestion pipeline (Part A).
Run via: python scripts/run_ingest.py
"""

from __future__ import annotations

import uuid

from pypdf import PdfReader
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import COLLECTION, EMBED_DIM, QDRANT_PATH, get_client, get_embed_model
from src.schemas import Chunk


def extract_pages(path: str) -> list[tuple[int, str]]:
    """Extract (page_number, text) pairs from a PDF."""
    reader = PdfReader(path)
    return [(i, p.extract_text() or "") for i, p in enumerate(reader.pages)]


def chunk_page(
    text: str,
    source: str,
    page: int,
    size: int = 512,
    overlap: int = 75,
) -> list[Chunk]:
    """Split a page's text into overlapping word-window chunks."""
    words = text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    for idx, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        if not window:
            break
        chunks.append(
            Chunk(
                chunk_id=f"{source}_p{page}_{idx}",
                text=" ".join(window),
                source_doc=source,
                page=page,
            )
        )
    return chunks


def build_index(chunks: list[Chunk]) -> None:
    """Embed chunks and upsert into Qdrant."""
    model = get_embed_model()
    vectors = model.encode([c.text for c in chunks], normalize_embeddings=True)
    client = get_client()

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                vector=v.tolist(),
                payload=c.model_dump(),
            )
            for c, v in zip(chunks, vectors)
        ],
    )
    print(f"Upserted {len(chunks)} chunks into '{COLLECTION}'")
