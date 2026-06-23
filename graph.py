"""
graph.py — LangGraph skeleton for Stage 1.

Graph topology (Stage 1):
    entry → retrieve → generate → END

Nodes
-----
retrieve : Calls the retriever. In Stage 1 this is a clearly-marked MOCK that
           returns canned Chunk objects. My partner (Part A) will replace the
           internals of this node in later stages — the node signature stays the same.

generate : Builds the system prompt from the active profile, injects retrieved
           chunks as <context>, calls Claude, and returns the AI message + citations.

How LangGraph works (quick mental model):
    - You define a StateGraph[TutorState].
    - Each node is a plain function: (state) → partial_state_update (dict).
    - LangGraph merges the returned dict into the current state using reducers.
    - Edges define execution order. END is a sentinel from langgraph.graph.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from graph_state import TutorState
from profiles import build_system_prompt
from schemas import Chunk, Citation

load_dotenv()

# ─── LLM ──────────────────────────────────────────────────────────────────────

def _get_llm() -> ChatGroq:
    """
    Instantiate Llama-3.3-70b via Groq (free tier).
    ⚠️  TEMPORARY — swap back to ChatAnthropic(model='claude-sonnet-4-6')
    once Anthropic credits are loaded before final submission.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        max_tokens=1024,
    )


# ─── MOCK Retriever ────────────────────────────────────────────────────────────
# ⚠️  MOCK — Replace internals with Part A's retriever in Stage 2.
# The function signature (takes query string, returns List[Chunk]) is the
# agreed Part A ↔ Part B interface. DO NOT change the signature.

def _mock_retrieve(query: str) -> list[Chunk]:
    """
    MOCK retriever — returns canned chunks regardless of query.

    In production (Stage 2+), Part A will replace this body with a real
    vector-store lookup. The return type (List[Chunk]) is the contract.
    """
    return [
        Chunk(
            chunk_id="mock-001",
            text=(
                "Gradient descent is an optimisation algorithm used to minimise a loss function. "
                "It works by iteratively moving in the direction of the negative gradient. "
                "The learning rate controls the step size at each iteration."
            ),
            source_doc="intro_to_ml.pdf",
            page=42,
        ),
        Chunk(
            chunk_id="mock-002",
            text=(
                "Overfitting occurs when a model learns the training data too well, "
                "including its noise, and performs poorly on unseen data. "
                "Regularisation techniques such as L1/L2 and dropout help mitigate overfitting."
            ),
            source_doc="intro_to_ml.pdf",
            page=87,
        ),
        Chunk(
            chunk_id="mock-003",
            text=(
                "A confusion matrix is a table used to evaluate classification model performance. "
                "Rows represent actual classes; columns represent predicted classes. "
                "Key metrics derived from it: precision, recall, and F1-score."
            ),
            source_doc="evaluation_metrics.pdf",
            page=15,
        ),
    ]


# ─── Node: retrieve ────────────────────────────────────────────────────────────

def retrieve(state: TutorState) -> dict[str, Any]:
    """
    Retrieval node — extracts the latest user query and fetches relevant chunks.

    Returns a partial state update: sets `retrieved_chunks`.
    """
    # Pull the most recent human message as the query
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    query = last_human.content if last_human else ""

    # ⚠️  MOCK call — Part A replaces body of _mock_retrieve, not this line.
    chunks = _mock_retrieve(query)

    return {"retrieved_chunks": chunks}


# ─── Node: generate ────────────────────────────────────────────────────────────

def generate(state: TutorState) -> dict[str, Any]:
    """
    Generation node — builds a context-augmented prompt and calls Claude.

    Steps:
    1. Build the system prompt from the active profile (prompt-level adaptation).
    2. Serialise retrieved chunks into a <context> block.
    3. Call Claude with [SystemMessage, *conversation_history].
    4. Extract citations from the chunks used.
    5. Return the AI message (appended via add_messages reducer) + citations.
    """
    llm = _get_llm()

    # 1. System prompt (profile-adaptive)
    profile = state.get("profile")
    memory_block = state.get("memory_block", "")
    system_prompt = build_system_prompt(profile, memory_block) if profile else (
        "You are a helpful course tutor. Answer clearly and cite your sources."
    )

    # 2. Build <context> block from retrieved chunks
    chunks: list[Chunk] = state.get("retrieved_chunks", [])
    if chunks:
        context_parts = []
        for c in chunks:
            context_parts.append(
                f"[chunk_id={c.chunk_id} | source={c.source_doc} | page={c.page}]\n{c.text}"
            )
        context_block = "<context>\n" + "\n\n".join(context_parts) + "\n</context>"
    else:
        context_block = "<context>\nNo relevant chunks retrieved.\n</context>"

    # 3. Inject context into the last human message (non-destructively)
    messages = list(state["messages"])
    if messages and isinstance(messages[-1], HumanMessage):
        augmented_last = HumanMessage(
            content=f"{messages[-1].content}\n\n{context_block}"
        )
        messages = messages[:-1] + [augmented_last]

    # 4. Call Claude
    full_messages = [SystemMessage(content=system_prompt)] + messages
    response = llm.invoke(full_messages)

    # 5. Build citations from the chunks we used
    citations = [
        Citation(chunk_id=c.chunk_id, source=c.source_doc, page=c.page)
        for c in chunks
    ]

    # Return partial state — messages uses add_messages reducer (appends)
    return {
        "messages": [response],
        "citations": citations,
    }


# ─── Graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> Any:
    """
    Wire up the Stage 1 LangGraph:  entry → retrieve → generate → END
    """
    builder = StateGraph(TutorState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# Singleton graph (compiled once, reused across Streamlit reruns via st.cache_resource)
tutor_graph = build_graph()
