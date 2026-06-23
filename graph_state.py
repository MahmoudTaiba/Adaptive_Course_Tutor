"""
graph_state.py — LangGraph state definition for the Adaptive Course Tutor.

LangGraph passes state through every node in the graph.
We define it as a TypedDict so LangGraph can introspect and validate it.

Key concept — Reducers:
    Normal TypedDict fields get REPLACED on each update.
    The `messages` field uses the special `add_messages` reducer from LangGraph,
    which APPENDS new messages instead of replacing the whole list.
    This is how conversational history is maintained across graph steps.
"""

from __future__ import annotations

from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from schemas import Chunk, Citation, StudentProfile


class TutorState(TypedDict):
    """
    The shared state object threaded through every node in the LangGraph.

    Fields
    ------
    messages : list[BaseMessage]  (reducer: add_messages — APPENDS, not replaces)
        Full conversation history (HumanMessage, AIMessage, SystemMessage, etc.)
        LangGraph will call add_messages(existing, new) on each update,
        so nodes just return {"messages": [new_message]} to extend the list.

    profile : StudentProfile
        The active student profile. Set once at session start; read by generate
        node to build the system prompt.

    memory_block : str
        Summary of prior sessions injected into the system prompt.
        Empty string when no prior history exists.

    retrieved_chunks : list[Chunk]
        Chunks returned by the retriever node (Part A interface).
        In Stage 1 this is populated by a MOCK retriever.

    citations : list[Citation]
        Citations extracted from retrieved_chunks after generation.
        Populated by the generate node.

    route : str | None
        Optional routing signal. Reserved for Stage 2 when we add a router node
        to direct queries to quiz generation vs. explanation vs. memory replay.
        Unused in Stage 1.
    """

    messages: Annotated[list, add_messages]
    profile: Optional[StudentProfile]
    memory_block: str
    retrieved_chunks: List[Chunk]
    citations: List[Citation]
    route: Optional[str]
