"""
src/graph.py — LangGraph graph (Stage 2 + Part A integration).

Graph topology: entry → retrieve → generate → END

Retriever:
    Uses the REAL retrieve_chunks() from src.retrieve (Qdrant + sentence-transformers).
    Falls back to a clearly-marked mock ONLY if the Qdrant index is empty,
    so the app still runs before ingestion.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.config import get_llm
from src.graph_state import TutorState
from src.profiles import build_system_prompt
from src.schemas import Citation, Chunk, RetrievedChunk


# ─── Mock fallback (used only when Qdrant index is empty) ─────────────────────
# ⚠️  MOCK FALLBACK — remove once ingestion has been run.

def _mock_chunks() -> list[RetrievedChunk]:
    """Returns canned RetrievedChunks when the real index has no data."""
    return [
        RetrievedChunk(
            chunk_id="mock-001",
            text=(
                "Gradient descent is an optimisation algorithm used to minimise a loss function. "
                "It works by iteratively moving in the direction of the negative gradient. "
                "The learning rate controls the step size at each iteration."
            ),
            source_doc="intro_to_ml.pdf",
            page=42,
            similarity=1.0,
        ),
        RetrievedChunk(
            chunk_id="mock-002",
            text=(
                "Overfitting occurs when a model learns the training data too well, "
                "including its noise, and performs poorly on unseen data. "
                "Regularisation techniques such as L1/L2 and dropout help mitigate overfitting."
            ),
            source_doc="intro_to_ml.pdf",
            page=87,
            similarity=0.95,
        ),
        RetrievedChunk(
            chunk_id="mock-003",
            text=(
                "A confusion matrix is a table used to evaluate classification model performance. "
                "Rows represent actual classes; columns represent predicted classes. "
                "Key metrics derived from it: precision, recall, and F1-score."
            ),
            source_doc="evaluation_metrics.pdf",
            page=15,
            similarity=0.90,
        ),
    ]


def _real_retrieve(query: str) -> list[RetrievedChunk]:
    """
    Call the real Qdrant retriever.
    Returns empty list if the index doesn't exist yet (pre-ingestion).
    """
    try:
        from src.retrieve import retrieve_chunks
        from src.config import get_client, COLLECTION
        client = get_client()
        if not client.collection_exists(COLLECTION):
            return []
        results = retrieve_chunks(query, top_k=5)
        return results
    except Exception:
        return []


# ─── Node: retrieve ────────────────────────────────────────────────────────────

def retrieve(state: TutorState) -> dict[str, Any]:
    """
    Retrieval node — calls real Qdrant retriever with mock fallback.
    Falls back to mock ONLY when the index is empty (pre-ingestion).
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = last_human.content if last_human else ""

    chunks = _real_retrieve(query)

    if not chunks:
        # ⚠️  MOCK FALLBACK — index is empty, using canned chunks
        chunks = _mock_chunks()

    return {"retrieved_chunks": chunks}


# ─── Node: generate ────────────────────────────────────────────────────────────

def generate(state: TutorState) -> dict[str, Any]:
    """
    Generation node — builds context-augmented prompt, calls LLM, returns response.

    Uses RetrievedChunk.similarity to optionally sort/filter chunks.
    """
    llm = get_llm()

    profile = state.get("profile")
    memory_block = state.get("memory_block", "")
    system_prompt = (
        build_system_prompt(profile, memory_block)
        if profile
        else "You are a helpful course tutor. Answer clearly and cite your sources."
    )

    # Build <context> block — sorted by similarity (best first)
    chunks: list[RetrievedChunk] = sorted(
        state.get("retrieved_chunks", []),
        key=lambda c: c.similarity,
        reverse=True,
    )

    if chunks:
        context_parts = [
            f"[chunk_id={c.chunk_id} | source={c.source_doc} | page={c.page} | score={c.similarity:.3f}]\n{c.text}"
            for c in chunks
        ]
        context_block = "<context>\n" + "\n\n".join(context_parts) + "\n</context>"
    else:
        context_block = "<context>\nNo relevant chunks retrieved.\n</context>"

    # Inject context into the last human message
    messages = list(state["messages"])
    if messages and isinstance(messages[-1], HumanMessage):
        augmented = HumanMessage(content=f"{messages[-1].content}\n\n{context_block}")
        messages = messages[:-1] + [augmented]

    full_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(full_messages)

    citations = [
        Citation(chunk_id=c.chunk_id, source=c.source_doc, page=c.page)
        for c in chunks
    ]

    return {"messages": [response], "citations": citations}


# ─── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> Any:
    builder = StateGraph(TutorState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()


tutor_graph = build_graph()
