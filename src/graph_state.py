"""
src/graph_state.py — LangGraph TutorState TypedDict.
Uses add_messages reducer so messages are appended, not replaced.
"""

from __future__ import annotations

from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from src.schemas import Citation, RetrievedChunk, StudentProfile


class TutorState(TypedDict):
    """
    Shared state threaded through every LangGraph node.

    messages          : conversation history (add_messages reducer — appends)
    profile           : active StudentProfile (set at session start)
    memory_block      : compressed prior session context for system prompt
    retrieved_chunks  : RetrievedChunk list from retrieve node (has .similarity)
    citations         : Citation list built after retrieval
    route             : optional routing signal (reserved for Stage 3)
    """
    messages: Annotated[list, add_messages]
    profile: Optional[StudentProfile]
    memory_block: str
    retrieved_chunks: List[RetrievedChunk]
    citations: List[Citation]
    route: Optional[str]
