"""
src/retrieve.py — Vector retrieval (Part A).
Called by the graph's retrieve node in graph.py.
"""

from __future__ import annotations

from src.config import COLLECTION, get_client, get_embed_model
from src.schemas import RetrievedChunk


def retrieve_chunks(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Embed the query and fetch the top-k most similar chunks from Qdrant.

    Args:
        query:  The student's question / search string.
        top_k:  Number of chunks to return.

    Returns:
        List of RetrievedChunk (Chunk + similarity score), sorted by score desc.
    """
    model = get_embed_model()
    client = get_client()

    qv = model.encode(query, normalize_embeddings=True)

    res = client.query_points(
        collection_name=COLLECTION,
        query=qv.tolist(),
        limit=top_k,
        with_payload=True,
    )

    return [
        RetrievedChunk(**pt.payload, similarity=pt.score)
        for pt in res.points
    ]
