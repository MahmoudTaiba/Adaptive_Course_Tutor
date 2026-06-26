"""
src/memory.py — Session-log memory system (Stage 2).
Storage: data/memory/{student_name}.json (append-only JSON array)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import get_llm
from src.schemas import SessionLog

MEMORY_DIR = Path(__file__).parent.parent / "data" / "memory"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _memory_path(student_name: str) -> Path:
    safe_name = student_name.lower().replace(" ", "_")
    return MEMORY_DIR / f"{safe_name}.json"


def summarize_session(student_name: str, messages: list[BaseMessage]) -> SessionLog:
    """
    Compress a conversation into a structured SessionLog via one LLM call.
    Uses .with_structured_output(SessionLog) — structured JSON output.
    """
    llm = get_llm(temperature=0, max_tokens=512)
    structured_llm = llm.with_structured_output(SessionLog)

    convo_lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            convo_lines.append(f"Student: {m.content}")
        elif isinstance(m, AIMessage):
            content = m.content[:400] + "..." if len(m.content) > 400 else m.content
            convo_lines.append(f"Tutor: {content}")

    convo_text = "\n".join(convo_lines) if convo_lines else "No conversation."

    prompt = f"""Analyse this tutoring session and extract a structured memory log.

Conversation:
{convo_text}

Extract:
- topics_covered: list of topics the student asked about or learned
- weak_topics: topics where the student seemed confused or asked follow-ups repeatedly
- summary: 1-2 sentence plain-English summary of the session

Be concise. Student name: {student_name}"""

    log: SessionLog = structured_llm.invoke(prompt)
    log.student_name = student_name
    log.timestamp = datetime.utcnow()
    return log


def persist_log(log: SessionLog) -> None:
    """Append a SessionLog to the student's JSON file."""
    path = _memory_path(log.student_name)
    logs = []
    if path.exists():
        with open(path, "r") as f:
            logs = json.load(f)

    log_dict = log.model_dump()
    log_dict["timestamp"] = log.timestamp.isoformat()
    logs.append(log_dict)

    with open(path, "w") as f:
        json.dump(logs, f, indent=2)


def load_memory_block(student_name: str, max_sessions: int = 3) -> str:
    """
    Load the last N sessions and compress into a short context string.
    Returns empty string if no prior history.
    """
    path = _memory_path(student_name)
    if not path.exists():
        return ""

    with open(path, "r") as f:
        logs = json.load(f)

    if not logs:
        return ""

    recent = logs[-max_sessions:]
    lines = []
    for entry in recent:
        date_str = entry.get("timestamp", "")[:10]
        covered = ", ".join(entry.get("topics_covered", [])) or "none"
        weak = ", ".join(entry.get("weak_topics", [])) or "none"
        summary = entry.get("summary", "")
        lines.append(
            f"Session ({date_str}): Covered: {covered}. "
            f"Struggled with: {weak}. Summary: {summary}"
        )

    return "\n".join(lines)
