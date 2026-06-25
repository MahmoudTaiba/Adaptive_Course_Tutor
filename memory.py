"""
memory.py — Session-log memory system (Stage 2).

Storage: one JSON file per student at data/memory/{student_name}.json
No vector DB — just append-only JSON logs, last N sessions loaded as context.

Flow:
    End of session  → summarize_session() → persist_log()
    Start of session → load_memory_block()  → injected into system prompt
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

from schemas import SessionLog

load_dotenv()

# ─── Storage path ──────────────────────────────────────────────────────────────

MEMORY_DIR = Path(__file__).parent / "data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _memory_path(student_name: str) -> Path:
    safe_name = student_name.lower().replace(" ", "_")
    return MEMORY_DIR / f"{safe_name}.json"


# ─── LLM for summarisation ─────────────────────────────────────────────────────

def _get_summary_llm():
    """
    LLM used for session summarisation.
    ⚠️  TEMPORARY: Groq/Llama — swap to ChatAnthropic(model='claude-sonnet-4-6')
    once Anthropic credits are loaded.
    """
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
        max_tokens=512,
    )


# ─── Core functions ────────────────────────────────────────────────────────────

def summarize_session(student_name: str, messages: list[BaseMessage]) -> SessionLog:
    """
    Compress a conversation into a structured SessionLog via one LLM call.

    Uses .with_structured_output(SessionLog) — LangChain forces the model to
    return JSON that maps directly onto the SessionLog Pydantic schema.
    This is structured output, not manual parsing.

    Args:
        student_name: Name of the student (for the log).
        messages:     Full conversation (HumanMessage + AIMessage list).

    Returns:
        A populated SessionLog ready to persist.
    """
    llm = _get_summary_llm()
    structured_llm = llm.with_structured_output(SessionLog)

    # Format conversation for the summariser
    convo_lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            convo_lines.append(f"Student: {m.content}")
        elif isinstance(m, AIMessage):
            # Truncate long AI messages to keep the prompt small
            content = m.content[:400] + "..." if len(m.content) > 400 else m.content
            convo_lines.append(f"Tutor: {content}")

    convo_text = "\n".join(convo_lines) if convo_lines else "No conversation yet."

    prompt = f"""You are analysing a tutoring session to create a memory log.

Conversation:
{convo_text}

Extract:
- topics_covered: list of topics the student asked about or learned
- weak_topics: list of topics where the student seemed confused, asked follow-ups, or made errors
- summary: 1-2 sentence plain-English summary of what happened in this session

Be concise. weak_topics should only include genuine struggles, not every topic covered.
Student name is: {student_name}"""

    log: SessionLog = structured_llm.invoke(prompt)

    # Ensure student_name and timestamp are set correctly
    log.student_name = student_name
    log.timestamp = datetime.utcnow()

    return log


def persist_log(log: SessionLog) -> None:
    """
    Append a SessionLog to the student's JSON file.

    File format: a JSON array of SessionLog dicts (append-only).
    Creates the file if it doesn't exist.
    """
    path = _memory_path(log.student_name)

    # Load existing logs
    if path.exists():
        with open(path, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    # Append new log (serialize datetime to ISO string)
    log_dict = log.model_dump()
    log_dict["timestamp"] = log.timestamp.isoformat()
    logs.append(log_dict)

    with open(path, "w") as f:
        json.dump(logs, f, indent=2)


def load_memory_block(student_name: str, max_sessions: int = 3) -> str:
    """
    Load the last N sessions and compress them into a SHORT context string.

    Kept deliberately compact to avoid bloating the system prompt.
    Returns an empty string if no prior history exists.

    Args:
        student_name:  Student whose history to load.
        max_sessions:  How many past sessions to include (default 3).

    Returns:
        A multi-line string like:
            Past session (2024-01-10): Covered: gradient descent, overfitting.
            Struggled with: backpropagation.
    """
    path = _memory_path(student_name)

    if not path.exists():
        return ""

    with open(path, "r") as f:
        logs = json.load(f)

    if not logs:
        return ""

    # Take the most recent N sessions
    recent = logs[-max_sessions:]

    lines = []
    for entry in recent:
        date_str = entry.get("timestamp", "")[:10]  # YYYY-MM-DD only
        covered = ", ".join(entry.get("topics_covered", [])) or "none"
        weak = ", ".join(entry.get("weak_topics", [])) or "none"
        summary = entry.get("summary", "")

        lines.append(
            f"Session ({date_str}): Covered: {covered}. "
            f"Struggled with: {weak}. "
            f"Summary: {summary}"
        )

    return "\n".join(lines)
